#!/usr/bin/env python
"""
Synthetic benchmark harness for HierarchicalClassifier.

Generates a balanced class hierarchy (optionally with extra DAG edges), a sparse feature matrix
whose rows carry a per-leaf signature so local classifiers have something to learn, and times
fit / predict / predict_proba along with the pickled model size. A cheap, fully vectorized
centroid base estimator is used by default so that the numbers reflect the framework's own
overhead rather than the base estimator's; pass --base logreg to see the realistic mix.

Examples:

    uv run python benchmarks/bench.py                          # default medium run
    uv run python benchmarks/bench.py --n-samples 50000 --depth 4 --branching 4
    uv run python benchmarks/bench.py --profile fit            # cProfile top functions for fit
    uv run python benchmarks/bench.py --profile predict --dag-extra-edges 10

"""

import argparse
import cProfile
import io
import pickle
import pstats
import time

import numpy as np
import scipy.sparse as sp
from networkx import DiGraph
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted, validate_data

from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT


class CentroidClassifier(ClassifierMixin, BaseEstimator):
    """Nearest-centroid classifier with a softmax over dot-product scores.

    Fit is one sparse matmul and predict_proba is one more, so timings are dominated by whatever
    the meta-estimator does around it. Not `sklearn.neighbors.NearestCentroid`: that scores by
    euclidean distance (a dense pass over X) and its binary `decision_function` is 1-D, whereas
    this one is a single sparse product and always returns one column per class.
    """

    def fit(self, X, y):
        X, y = validate_data(self, X, y, accept_sparse="csr")
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        indicator = sp.csr_matrix(
            (np.ones(len(y_idx)), (y_idx, np.arange(len(y_idx)))),
            shape=(len(self.classes_), len(y_idx)),
        )
        counts = np.asarray(indicator.sum(axis=1)).ravel()
        self.centroids_ = np.asarray((indicator @ X).todense() if sp.issparse(X) else indicator @ X) / counts[:, None]
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse="csr", reset=False)
        return np.asarray(X @ self.centroids_.T)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        proba = np.exp(scores)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


def make_hierarchy(depth, branching, dag_extra_edges=0, seed=0):
    """Balanced tree of given depth/branching under ROOT; leaves are the classes.

    With dag_extra_edges > 0, that many extra parent->leaf edges are added between random
    non-adjacent nodes so that some leaves have two parents (a DAG).
    """
    rng = np.random.default_rng(seed)
    graph = DiGraph()
    level = [ROOT]
    internal = []
    for d in range(depth):
        next_level = []
        for parent in level:
            for b in range(branching):
                child = f"L{len(next_level)}" if d == depth - 1 else f"n{d}_{parent}_{b}".replace(ROOT, "r")
                graph.add_edge(parent, child)
                next_level.append(child)
        if d < depth - 1:
            internal.extend(next_level)
        level = next_level
    leaves = level
    candidates = [(parent, leaf) for parent in internal for leaf in leaves if not graph.has_edge(parent, leaf)]
    if dag_extra_edges > len(candidates):
        raise ValueError(f"at most {len(candidates)} extra DAG edges are possible for this hierarchy")
    for pick in rng.choice(len(candidates), size=dag_extra_edges, replace=False):
        graph.add_edge(*candidates[pick])
    return graph, leaves


def make_dataset(leaves, n_samples, n_features, density, signature_size=8, seed=0):
    """Sparse CSR features with per-leaf signature columns, and leaf labels as strings."""
    rng = np.random.default_rng(seed)
    y = np.asarray(leaves)[rng.integers(len(leaves), size=n_samples)]
    X = sp.random(n_samples, n_features, density=density, format="csr", random_state=seed, dtype=np.float64)
    signature = {leaf: rng.choice(n_features, size=signature_size, replace=False) for leaf in leaves}
    rows = np.repeat(np.arange(n_samples), signature_size)
    cols = np.concatenate([signature[label] for label in y])
    X = X + sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=X.shape)
    X.sum_duplicates()
    return X, y


def make_classifier(base, graph, **kwargs):
    base_estimator = {
        "centroid": CentroidClassifier(),
        "logreg": LogisticRegression(max_iter=200),
    }[base]
    return HierarchicalClassifier(base_estimator=base_estimator, class_hierarchy=graph, **kwargs)


def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


def run(args):
    graph, leaves = make_hierarchy(args.depth, args.branching, args.dag_extra_edges, seed=args.seed)
    X, y = make_dataset(leaves, args.n_samples + args.n_test, args.n_features, args.density, seed=args.seed)
    X_train, y_train = X[: args.n_samples], y[: args.n_samples]
    X_test, y_test = X[args.n_samples :], y[args.n_samples :]
    if args.dense:
        X_train, X_test = X_train.toarray(), X_test.toarray()

    print(
        f"hierarchy: depth={args.depth} branching={args.branching} nodes={graph.number_of_nodes()} "
        f"leaves={len(leaves)} dag_extra_edges={args.dag_extra_edges}"
    )
    print(
        f"data: n_train={args.n_samples} n_test={args.n_test} n_features={args.n_features} "
        f"nnz/row={X_train.nnz / args.n_samples if sp.issparse(X_train) else 'dense'} base={args.base}"
    )

    clf = make_classifier(args.base, graph)

    if args.profile:
        profiler = cProfile.Profile()
        if args.profile == "fit":
            profiler.runcall(clf.fit, X_train, y_train)
        else:
            clf.fit(X_train, y_train)
            profiler.runcall(getattr(clf, args.profile), X_test)
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats(args.sort).print_stats(args.top)
        print(stream.getvalue())
        return

    _, t_fit = timed(clf.fit, X_train, y_train)
    y_pred, t_predict = timed(clf.predict, X_test)
    _, t_proba = timed(clf.predict_proba, X_test)
    model_bytes = len(pickle.dumps(clf))
    accuracy = float(np.mean(y_pred == y_test))

    print(f"fit:            {t_fit:8.2f} s")
    print(f"predict:        {t_predict:8.2f} s  ({t_predict / args.n_test * 1e3:.2f} ms/sample)")
    print(f"predict_proba:  {t_proba:8.2f} s")
    print(f"accuracy:       {accuracy:8.3f}")
    print(f"pickled model:  {model_bytes / 1e6:8.1f} MB  (training X: {X_train.data.nbytes / 1e6:.1f} MB of values)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-samples", type=int, default=20_000)
    parser.add_argument("--n-test", type=int, default=2_000)
    parser.add_argument("--n-features", type=int, default=5_000)
    parser.add_argument("--density", type=float, default=0.01)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--branching", type=int, default=4)
    parser.add_argument("--dag-extra-edges", type=int, default=0)
    parser.add_argument("--base", choices=["centroid", "logreg"], default="centroid")
    parser.add_argument("--dense", action="store_true", help="benchmark the dense-input code path")
    parser.add_argument("--profile", choices=["fit", "predict", "predict_proba"], default=None)
    parser.add_argument("--sort", default="cumulative")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
