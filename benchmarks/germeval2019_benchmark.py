#!/usr/bin/env python
"""
GermEval 2019 Task 1 benchmark: hierarchical classification of German book blurbs.

Remus, Aly and Biemann (2019). 343 genre labels in a 4-level tree with 8 root genres; 14,548
training, 2,079 development and 4,157 test blurbs. Subtask A scores the root genres, subtask B the
full label set of each blurb (micro-F1 in both). The winning subtask-B system (TwistBytes, Benites
2019, micro-F1 0.6767) used this library with TF-IDF + LinearSVC local classifiers and a negative
decision threshold to trade precision for recall.

Features are TF-IDF views fitted once on the training text and shared by every node (the winning
system fitted its vocabularies per node, on the node's own subtree; its word 1-7-gram views were
measured here and do not beat the word 1-2 + character 2-3 grams used). The "text" set has those
two views of the title + blurb; "text+metadata" adds three views of the record's other fields,
which every participant had at test time (the publisher URL was withheld and is not used): the
title on its own, the author names as tokens, and ISBN publisher prefixes, which identify the
imprint. Authors and imprints are strong genre predictors, and the task report notes several
teams used such metadata. Local classifiers use the "inclusive" training strategy by default
(every out-of-subtree blurb is a negative at every node); --training-strategy siblings trains
each node on its own subtree only.

Protocol: the configuration (feature set, decision threshold, root fallback) is chosen on the
development split with models fitted on the training split only; the model is then refitted on
train + dev and the test set is scored once for the default configuration (text features,
threshold 0) and once for the dev-selected one. Per-class thresholds are not tuned here: most of
the 343 labels have too few development positives for that (they hurt on dev in 2-fold
cross-tuning), so one global threshold is selected.

The official data package (CC BY-NC 4.0, University of Hamburg Language Technology group) is
downloaded on first use into ~/scikit_learn_data/germeval2019.

Example:

    uv run python benchmarks/germeval2019_benchmark.py

"""

import argparse
import re
import time
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
from networkx import DiGraph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer
from sklearn.svm import LinearSVC

from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.thresholds import route


PACKAGE_URL = (
    "https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc/germeval2019t1-public-data-final.zip"
)
FILES = {"train": "blurbs_train.txt", "dev": "blurbs_dev.txt", "test": "blurbs_test.txt", "hierarchy": "hierarchy.txt"}
BOOK = re.compile(r"<book .*?</book>", re.DOTALL)
FIELD = {name: re.compile(rf"<{name}>(.*?)</{name}>", re.DOTALL) for name in ("title", "body", "authors", "isbn")}
TOPIC = re.compile(r"<topic d=\"\d\"[^>]*>(.*?)</topic>")


def fetch(name, cache_dir):
    path = cache_dir / FILES[name]
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        archive, _ = urllib.request.urlretrieve(PACKAGE_URL)
        with zipfile.ZipFile(archive) as package:
            package.extractall(cache_dir)
    return path.read_text(encoding="utf-8")


def parse_books(text):
    """Return (books, label sets): each book a dict of its text fields; label sets are empty for test books."""
    books, labels = [], []
    for book in BOOK.findall(text):
        books.append({name: (pattern.search(book) or [None, ""])[1].strip() for name, pattern in FIELD.items()})
        labels.append(sorted(set(TOPIC.findall(book))))
    return books, labels


def make_hierarchy(text):
    graph = DiGraph([tuple(line.split("\t")) for line in text.splitlines() if line.strip()])
    for node in [node for node, degree in graph.in_degree() if degree == 0]:
        graph.add_edge(ROOT, node)
    return graph


def field(fn):
    """Transformer mapping each record (a dict of the book's fields) to the string `fn` computes from it."""
    return FunctionTransformer(lambda records: [fn(record) for record in records])


def text(record):
    return f"{record['title']} {record['body']}"


def title(record):
    return record["title"]


def author_tokens(record):
    names = re.split(r",| und |&", record["authors"])
    return " ".join("author_" + re.sub(r"\W+", "_", name.strip().lower()) for name in names if name.strip())


def isbn_prefix_tokens(record):
    # ISBN-13 publisher prefixes vary in length; one of these identifies the imprint
    isbn = record["isbn"]
    return " ".join(f"isbn{length}_{isbn[:length]}" for length in (7, 8, 9) if len(isbn) >= length)


def tfidf(**kwargs):
    return TfidfVectorizer(sublinear_tf=True, **kwargs)


def tokens(fn):
    return make_pipeline(field(fn), tfidf(token_pattern=r"\S+", lowercase=False))


VIEWS = {
    "word": lambda: make_pipeline(field(text), tfidf(ngram_range=(1, 2), max_features=70_000)),
    "char": lambda: make_pipeline(field(text), tfidf(analyzer="char_wb", ngram_range=(2, 3), max_features=70_000)),
    "title": lambda: make_pipeline(field(title), tfidf(ngram_range=(1, 2))),
    "authors": lambda: tokens(author_tokens),
    "isbn": lambda: tokens(isbn_prefix_tokens),
}
FEATURE_SETS = {
    "text": ["word", "char"],
    "text+metadata": ["word", "char", "title", "authors", "isbn"],
}


def make_classifier(graph, mlb, C, strategy, threshold, min_root=0):
    return HierarchicalClassifier(
        base_estimator=OneVsRestClassifier(LinearSVC(C=C)),
        class_hierarchy=graph,
        mlb=mlb,
        use_decision_function=True,
        training_strategy=strategy,
        mlb_prediction_threshold=threshold,
        mlb_min_root_predictions=min_root,
    )


def vectorize(features, fit_books, *books):
    """TF-IDF views of a feature set, fitted on `fit_books` only; returns one matrix per argument."""
    vectorizer = FeatureUnion([(view, VIEWS[view]()) for view in FEATURE_SETS[features]])
    return [vectorizer.fit_transform(fit_books), *(vectorizer.transform(b) for b in books)]


def micro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="micro", zero_division=0)


def main():
    warnings.filterwarnings("ignore", message="Label .* is present in all training examples", category=UserWarning)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--C", type=float, default=1.5, help="LinearSVC regularisation (the paper used 1.5)")
    parser.add_argument(
        "--training-strategy", choices=("inclusive", "siblings"), default="inclusive", help="local training sets"
    )
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "scikit_learn_data" / "germeval2019")
    args = parser.parse_args()
    strategy = args.training_strategy

    graph = make_hierarchy(fetch("hierarchy", args.cache_dir))
    nodes = [node for node in graph.nodes if node != ROOT]
    mlb = MultiLabelBinarizer(classes=nodes).fit([nodes])
    columns = {node: column for column, node in enumerate(nodes)}
    root_columns = [columns[node] for node in graph.successors(ROOT)]

    X_train, y_train = parse_books(fetch("train", args.cache_dir))
    X_dev, y_dev = parse_books(fetch("dev", args.cache_dir))
    X_test, y_test = parse_books(fetch("test", args.cache_dir))
    Y_train, Y_dev, Y_test = mlb.transform(y_train), mlb.transform(y_dev), mlb.transform(y_test)
    print(
        f"GermEval 2019 Task 1: {len(X_train):,} train / {len(X_dev):,} dev / {len(X_test):,} test blurbs, "
        f"{len(nodes)} labels ({len(root_columns)} roots, {sum(1 for _, d in graph.out_degree() if d > 0) - 1} "
        f"internal), labels/blurb {Y_train.sum(1).mean():.2f}"
    )

    # --- choose feature set, threshold and root fallback on dev (models fitted on train only), using every
    #     node's score (threshold -inf visits all nodes) and emulating the walk (`thresholds.route`) per candidate
    grid = np.round(np.arange(-0.5, 0.21, 0.05), 2)
    dev_f1 = {}
    for features in FEATURE_SETS:
        F_train, F_dev = vectorize(features, X_train, X_dev)
        start = time.perf_counter()
        clf = make_classifier(graph, mlb, args.C, strategy, -np.inf).fit(F_train, Y_train)
        scores_dev = clf.predict_proba(F_dev)
        t_dev = time.perf_counter() - start
        print(f"{features} features: classifier fit on train + score dev {t_dev:.0f}s", flush=True)
        for min_root in (0, 1):
            for t in grid:
                dev_f1[(features, t, min_root)] = micro_f1(Y_dev, route(scores_dev, t, graph, nodes, min_root=min_root))
        best = max((k for k in dev_f1 if k[0] == features), key=dev_f1.get)
        print(f"  best on dev: threshold {best[1]:+.2f}, min_root {best[2]}: subtask-B micro-F1 {dev_f1[best]:.4f}")
    chosen = max(dev_f1, key=dev_f1.get)
    print(f"chosen: {chosen[0]} features, threshold {chosen[1]:+.2f}, min_root {chosen[2]} (dev {dev_f1[chosen]:.4f})")

    # --- refit on train + dev, score the test set once per pre-registered configuration
    X_all, Y_all = X_train + X_dev, np.vstack([Y_train, Y_dev])
    configurations = {"default (text, threshold 0)": ("text", 0.0, 0), "dev-selected": chosen}
    matrices = {features: vectorize(features, X_all, X_test) for features in {f for f, _, _ in configurations.values()}}
    for name, (features, threshold, min_root) in configurations.items():
        F_all, F_test = matrices[features]
        start = time.perf_counter()
        clf = make_classifier(graph, mlb, args.C, strategy, threshold, min_root).fit(F_all, Y_all)
        t_fit = time.perf_counter() - start
        start = time.perf_counter()
        Y_pred = clf.predict(F_test)
        t_predict = time.perf_counter() - start
        print(
            f"TEST {name:<28} ({features}, t={threshold:+.2f}, min_root={min_root})   "
            f"subtask B micro-F1 {micro_f1(Y_test, Y_pred):.4f}   "
            f"subtask A micro-F1 {micro_f1(Y_test[:, root_columns], Y_pred[:, root_columns]):.4f}   "
            f"labels/blurb {Y_pred.sum(1).mean():.2f}  no-label {np.mean(Y_pred.sum(1) == 0):.3f}   "
            f"classifier fit {t_fit:.0f}s predict {t_predict:.1f}s (TF-IDF excluded)",
            flush=True,
        )
    print("published test scores: TwistBytes (this library, t=-0.25) subtask B 0.6767 (1st of 10);")
    print("                       TwistBytes flat model subtask A 0.8634 (2nd)")


if __name__ == "__main__":
    main()
