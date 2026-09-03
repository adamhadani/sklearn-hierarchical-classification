"""
Shared code of the book-blurb benchmarks: GermEval 2019 Task 1 (German) and the Blurb Genre
Collection (English), both from the Hamburg Language Technology group and both distributed as
one XML-like record per book with a genre hierarchy file.

Features are TF-IDF views fitted once on the training records and shared by every node. The
"text" set has word 1-2 grams and character 2-3 grams of the title + blurb; "text+metadata"
adds three views of the record's other fields: the title on its own, the author names as tokens,
and ISBN publisher prefixes, which identify the imprint. Every view is L2-normalised and the views
are concatenated with equal weight. Local classifiers are LinearSVC one-vs-rest models trained
with the "inclusive" strategy by default (every out-of-subtree blurb is a negative at every node).

Protocol: the configuration (feature set, decision threshold, root fallback) is chosen on the
development split with models fitted on the training split only; the model is then refitted on
train + dev and the test set is scored once for the default configuration (text features,
threshold 0) and once per feature set for the configuration selected on dev for it. Per-class
thresholds are not tuned: most labels have too few development positives for that, so one global
threshold is selected. Decision scores of every node come from one `-inf` prediction and the
walk is emulated per candidate with `thresholds.route`.

"""

import argparse
import re
import time
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass, replace
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


BOOK = re.compile(r"<book .*?</book>", re.DOTALL)


@dataclass(frozen=True)
class Blurbs:
    """One blurb dataset: where to get it, how its records are tagged, and what to compare against."""

    name: str
    package_url: str
    cache_dir: Path
    files: dict[str, str]  # "train", "dev", "test", "hierarchy" -> file name inside the package
    topic: re.Pattern  # matches one genre label of a record
    tags: dict[str, str]  # record field ("title", "body", "authors", "isbn") -> XML tag name
    C: float  # LinearSVC regularisation
    published: list[str]  # lines describing published results, printed after the test scores

    def fetch(self, name):
        path = self.cache_dir / self.files[name]
        if not path.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            archive, _ = urllib.request.urlretrieve(self.package_url)
            with zipfile.ZipFile(archive) as package:
                package.extractall(self.cache_dir)
        return path.read_text(encoding="utf-8")

    def parse_books(self, text):
        """Return (books, label sets): each book a dict of its text fields."""
        fields = {key: re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL) for key, tag in self.tags.items()}
        books, labels = [], []
        for book in BOOK.findall(text):
            books.append({key: (pattern.search(book) or [None, ""])[1].strip() for key, pattern in fields.items()})
            labels.append(sorted(set(self.topic.findall(book))))
        return books, labels


def make_hierarchy(text):
    """Graph of a tab-separated parent/child file; a line with a single name is a root genre without children."""
    graph = DiGraph()
    for line in text.splitlines():
        parts = [part.strip() for part in line.split("\t") if part.strip()]
        if len(parts) == 1:
            graph.add_node(parts[0])
        elif len(parts) == 2:
            graph.add_edge(*parts)
        elif parts:
            raise ValueError(f"Unexpected hierarchy line: {line!r}")
    for node in [node for node, degree in graph.in_degree() if degree == 0]:
        graph.add_edge(ROOT, node)
    return graph


def field(fn):
    """Transformer mapping each record (a dict of the book's fields) to the string `fn` computes from it."""
    return FunctionTransformer(lambda records: [fn(record) for record in records])


def record_text(record):
    return f"{record['title']} {record['body']}"


def title(record):
    return record["title"]


def author_tokens(record):
    names = re.split(r",| und | and |&", record["authors"])
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
    "word": lambda: make_pipeline(field(record_text), tfidf(ngram_range=(1, 2), max_features=70_000)),
    "char": lambda: make_pipeline(
        field(record_text), tfidf(analyzer="char_wb", ngram_range=(2, 3), max_features=70_000)
    ),
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


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def parse_args(spec, description):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--C", type=float, default=spec.C, help="LinearSVC regularisation")
    parser.add_argument(
        "--training-strategy", choices=("inclusive", "siblings"), default="inclusive", help="local training sets"
    )
    parser.add_argument("--cache-dir", type=Path, default=spec.cache_dir)
    return parser.parse_args()


def run(spec, args):
    warnings.filterwarnings("ignore", message="Label .* is present in all training examples", category=UserWarning)
    spec = replace(spec, cache_dir=args.cache_dir)
    strategy = args.training_strategy

    X_train, y_train = spec.parse_books(spec.fetch("train"))
    X_dev, y_dev = spec.parse_books(spec.fetch("dev"))
    X_test, y_test = spec.parse_books(spec.fetch("test"))
    graph = make_hierarchy(spec.fetch("hierarchy"))
    # The label space is the genres that occur in the data: hierarchy leaves that never do would only add
    # empty columns to the F1 averages (BGC lists six, all absent from the training split alone as well)
    used = {label for labels in y_train + y_dev for label in labels}
    graph.remove_nodes_from([node for node in graph.nodes if graph.out_degree(node) == 0 and node not in used])
    nodes = [node for node in graph.nodes if node != ROOT]
    mlb = MultiLabelBinarizer(classes=nodes).fit([nodes])
    columns = {node: column for column, node in enumerate(nodes)}
    root_columns = [columns[node] for node in graph.successors(ROOT)]
    Y_train, Y_dev, Y_test = mlb.transform(y_train), mlb.transform(y_dev), mlb.transform(y_test)
    print(
        f"{spec.name}: {len(X_train):,} train / {len(X_dev):,} dev / {len(X_test):,} test blurbs, "
        f"{len(nodes)} labels ({len(root_columns)} roots, {sum(1 for _, d in graph.out_degree() if d > 0) - 1} "
        f"internal), labels/blurb {Y_train.sum(1).mean():.2f}"
    )

    # --- choose feature set, threshold and root fallback on dev (models fitted on train only), using every
    #     node's score (threshold -inf visits all nodes) and emulating the walk (`thresholds.route`) per candidate
    grid = np.round(np.arange(-0.5, 0.21, 0.05), 2)
    dev_f1, best = {}, {}
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
        best[features] = max((k for k in dev_f1 if k[0] == features), key=dev_f1.get)
        _, t, min_root = best[features]
        f1 = dev_f1[best[features]]
        print(f"  best on dev: threshold {t:+.2f}, min_root {min_root}: micro-F1 {f1:.4f}")
    chosen = max(dev_f1, key=dev_f1.get)
    print(f"chosen: {chosen[0]} features, threshold {chosen[1]:+.2f}, min_root {chosen[2]} (dev {dev_f1[chosen]:.4f})")

    # --- refit on train + dev, score the test set once per pre-registered configuration: the default and,
    #     for each feature set, the configuration selected on dev for it (the overall winner among them)
    X_all, Y_all = X_train + X_dev, np.vstack([Y_train, Y_dev])
    configurations = {"default (text, threshold 0)": ("text", 0.0, 0)}
    configurations.update({f"dev-selected ({features})": best[features] for features in FEATURE_SETS})
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
            f"micro-F1 {micro_f1(Y_test, Y_pred):.4f}   macro-F1 {macro_f1(Y_test, Y_pred):.4f}   "
            f"root genres micro-F1 {micro_f1(Y_test[:, root_columns], Y_pred[:, root_columns]):.4f}   "
            f"labels/blurb {Y_pred.sum(1).mean():.2f}  no-label {np.mean(Y_pred.sum(1) == 0):.3f}   "
            f"classifier fit {t_fit:.0f}s predict {t_predict:.1f}s (TF-IDF excluded)",
            flush=True,
        )
    for line in spec.published:
        print(line)
