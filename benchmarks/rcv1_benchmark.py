#!/usr/bin/env python
"""
RCV1-v2 hierarchical multi-label benchmark (Lewis et al., JMLR 2004).

Reuters Corpus Volume 1 as distributed by scikit-learn: 804,414 newswire stories as TF-IDF vectors
(47,236 features) labelled with 103 topic codes arranged in a 4-root hierarchy (CCAT, ECAT, GCAT,
MCAT), with labels already expanded to include every ancestor. The LYRL2004 split is 23,149
training / 781,265 test documents. The hierarchy is recovered from the topic codes themselves:
a numeric code hangs under its longest proper prefix that is also a code (C1511 -> C151 -> C15),
otherwise under its category root; the alphabetic G codes (GCRIM, GPOL, ...) hang under GCAT.

Published reference (Lewis et al. 2004, Table 6, Topics, full test set): SVM with per-category
SCut thresholds, micro-averaged F1 0.816 and macro-averaged F1 0.607.

With --tune, per-class decision thresholds are tuned on 5-fold out-of-fold scores of the training
set (SCut, local to each parent: see `sklearn_hierarchical_classification.thresholds`) and passed
to the classifier as `mlb_prediction_threshold`; the test set is still scored once per model.

Example:

    uv run python benchmarks/rcv1_benchmark.py                 # full test set
    uv run python benchmarks/rcv1_benchmark.py --n-test 100000  # quicker
    uv run python benchmarks/rcv1_benchmark.py --tune           # + per-class thresholds from CV

"""

import argparse
import time
import warnings

import numpy as np
from networkx import DiGraph, ancestors, relabel_nodes
from sklearn.datasets import fetch_rcv1
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.metrics import h_fbeta_score, h_precision_score, h_recall_score
from sklearn_hierarchical_classification.thresholds import scut_thresholds


CATEGORY_ROOTS = {"C": "CCAT", "E": "ECAT", "G": "GCAT", "M": "MCAT"}


def parent_of(code, codes):
    if code in CATEGORY_ROOTS.values():
        return ROOT
    if not code[1:].isdigit():
        return CATEGORY_ROOTS[code[0]]
    for length in range(len(code) - 1, 1, -1):
        if code[:length] in codes:
            return code[:length]
    return CATEGORY_ROOTS[code[0]]


def make_topic_hierarchy(codes):
    codes = set(codes)
    return DiGraph([(parent_of(code, codes), code) for code in codes])


def make_hierarchical(graph, mlb, C, threshold=0.0):
    return HierarchicalClassifier(
        base_estimator=OneVsRestClassifier(LinearSVC(C=C)),
        class_hierarchy=graph,
        mlb=mlb,
        use_decision_function=True,
        mlb_prediction_threshold=threshold,
    )


def tuned_thresholds(graph, mlb, C, X, y, n_folds=5):
    """Per-class SCut thresholds from out-of-fold all-node scores on the training set."""
    scores, scored = np.zeros(y.shape), np.zeros(y.shape, dtype=bool)
    for fit_rows, score_rows in KFold(n_folds, shuffle=True, random_state=0).split(X):
        clf = make_hierarchical(graph, mlb, C, threshold=-np.inf).fit(X[fit_rows], y[fit_rows])
        scores[score_rows] = clf.predict_proba(X[score_rows])
        # A class without positives in this fold's training part is not learned, hence not scored
        scored[score_rows] = y[fit_rows].any(axis=0)
    return scut_thresholds(scores, y, graph=graph, classes=mlb.classes_, scored=scored)


def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


def report(name, y_true, y_pred, graph_by_column, t_fit, t_predict):
    print(
        f"{name:<28} micro-F1 {f1_score(y_true, y_pred, average='micro', zero_division=0):.3f}   "
        f"macro-F1 {f1_score(y_true, y_pred, average='macro', zero_division=0):.3f}   "
        f"hP {h_precision_score(y_true, y_pred, graph_by_column):.3f}  "
        f"hR {h_recall_score(y_true, y_pred, graph_by_column):.3f}  "
        f"hF1 {h_fbeta_score(y_true, y_pred, graph_by_column):.3f}   "
        f"fit {t_fit:6.1f}s  predict {t_predict:6.1f}s"
    )


def main():
    # At internal nodes most indicator columns are constant; OneVsRestClassifier warns about each one
    warnings.filterwarnings("ignore", message="Label .* is present in all training examples", category=UserWarning)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-test", type=int, default=None, help="use only the first N test documents")
    parser.add_argument("--C", type=float, default=1.0, help="LinearSVC regularisation")
    parser.add_argument(
        "--tune", action="store_true", help="tune per-class thresholds with 5-fold CV on the training set"
    )
    args = parser.parse_args()

    train, test = fetch_rcv1(subset="train"), fetch_rcv1(subset="test")
    codes = list(train.target_names)
    X_train, y_train = train.data, train.target
    X_test, y_test = test.data, test.target
    if args.n_test:
        X_test, y_test = X_test[: args.n_test], y_test[: args.n_test]
    y_test = y_test.toarray()

    graph = make_topic_hierarchy(codes)
    graph_by_column = relabel_nodes(graph, {code: column for column, code in enumerate(codes)})
    mlb = MultiLabelBinarizer(classes=codes).fit([codes])
    print(
        f"RCV1-v2: train {X_train.shape[0]:,} docs, test {X_test.shape[0]:,} docs, {X_train.shape[1]:,} features, "
        f"{len(codes)} topics, max depth {max(len(ancestors(graph, code)) for code in codes)}, "
        f"{sum(1 for _, degree in graph.out_degree() if degree > 0)} internal nodes"
    )

    flat = OneVsRestClassifier(LinearSVC(C=args.C))
    _, t_fit = timed(flat.fit, X_train, y_train.toarray())
    y_flat, t_predict = timed(flat.predict, X_test)
    report("flat OneVsRest(LinearSVC)", y_test, y_flat, graph_by_column, t_fit, t_predict)

    clf = make_hierarchical(graph, mlb, args.C)
    _, t_fit = timed(clf.fit, X_train, y_train)
    y_hier, t_predict = timed(clf.predict, X_test)
    report("hierarchical (LCPN, LinearSVC)", y_test, y_hier, graph_by_column, t_fit, t_predict)

    if args.tune:
        y_train_dense = y_train.toarray()
        thresholds, t_tune = timed(tuned_thresholds, graph, mlb, args.C, X_train, y_train_dense)
        clf = make_hierarchical(graph, mlb, args.C, threshold=thresholds)
        _, t_fit = timed(clf.fit, X_train, y_train)
        y_hier, t_predict = timed(clf.predict, X_test)
        report("hierarchical + local SCut (CV)", y_test, y_hier, graph_by_column, t_fit + t_tune, t_predict)

    print("published (Lewis et al. 2004)    micro-F1 0.816   macro-F1 0.607   (SVM, per-category tuned thresholds)")
    n_true, n_flat, n_hier = y_test.sum(1).mean(), y_flat.sum(1).mean(), y_hier.sum(1).mean()
    print(f"labels/doc: true {n_true:.2f}  flat {n_flat:.2f}  hierarchical {n_hier:.2f}")
    empty_flat, empty_hier = np.mean(y_flat.sum(1) == 0), np.mean(y_hier.sum(1) == 0)
    print(f"docs with no predicted label: flat {empty_flat:.3f}  hierarchical {empty_hier:.3f}")


if __name__ == "__main__":
    main()
