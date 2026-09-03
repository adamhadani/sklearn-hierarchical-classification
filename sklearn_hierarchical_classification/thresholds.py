"""Decision-threshold tuning for multi-label prediction scores."""

import numpy as np
from networkx import descendants, topological_sort

from sklearn_hierarchical_classification.array import top_k_mask
from sklearn_hierarchical_classification.constants import ROOT


def scut_thresholds(scores, y, graph=None, classes=None, root=ROOT, scored=None):
    """
    Per-class decision thresholds maximising F1 on held-out scores (SCut, Yang 1999).

    Parameters
    ----------
    scores : array-like, shape = [n_samples, n_classes]
        Held-out (e.g. out-of-fold) scores, one column per class, as returned by
        `HierarchicalClassifier.predict_proba` with `mlb_prediction_threshold=-np.inf` (which visits
        every node learned at fit; a class without a positive example at its parent is never
        scored and its column stays zero, which matters in cross-validation folds that drop a rare
        class).

    y : array-like, shape = [n_samples, n_classes]
        Binary indicator of the true labels, columns aligned with `scores`.

    graph : networkx.DiGraph, optional
        The class hierarchy. When given (together with `classes`), each class's threshold is
        tuned only on the samples that truly lie under the class's parent: that is the population
        the local classifier at the parent was trained on, and scores it gives to other samples are
        not meaningful. Without a graph every threshold is tuned on all samples.

    classes : sequence, optional
        The hierarchy node of each column of `scores` (e.g. `mlb.classes_`); required with `graph`.

    scored : array-like of bool, shape = [n_samples, n_classes], optional
        Which cells of `scores` were actually produced. Cells never scored keep a zero placeholder
        (e.g. a class with no positive example in a cross-validation fold's training part is not
        learned there, so the fold's held-out rows carry no score for it) and are excluded from
        that class's tuning.

    Returns
    -------
    thresholds : ndarray, shape = [n_classes]
        Predict class j where `scores[:, j] > thresholds[j]`; `inf` for classes without positives.

    """
    scores, y, scored = _validated(scores, y, scored)
    thresholds = []
    for j, rows in enumerate(_populations(y, graph, classes, root)):
        rows = rows[scored[rows, j]]
        thresholds.append(best_f1_threshold(scores[rows, j], y[rows, j]))
    return np.array(thresholds)


def routed_thresholds(scores, y, graph, classes, root=ROOT, scored=None, min_root=0):
    """
    Per-class decision thresholds tuned sequentially, top-down, on the samples the hierarchy routes.

    Like `scut_thresholds` with a graph, except that each class is tuned on the samples its parent
    *predicts* under the parent's own, already tuned threshold rather than on the samples truly under
    the parent. That is the population the class's threshold faces at prediction time: the parent's
    false positives are present as negatives and the parent's false negatives are absent, so the
    threshold maximises F1 under the actual routing instead of under perfect routing above it.
    Children of a root are tuned on every sample, and `min_root` replicates the classifier's
    `mlb_min_root_predictions` fallback before descending. A class below one that is never predicted
    (or whose parents are not among `classes`) gets `inf`, as does a class without positives among the
    samples routed to it.

    The walk is emulated with `route` on the score matrix, so it is exact on a tree. On a DAG the
    matrix holds each class's maximum score over *all* its parents (the `-inf` prediction visits them
    all), while at prediction time only parents that accept a sample score it, so a sample can be
    routed here through a parent that would reject it there.

    Parameters are as for `scut_thresholds` (`graph` and `classes` required) plus `min_root`.

    Returns
    -------
    thresholds : ndarray, shape = [n_classes]
        Predict class j where `scores[:, j] > thresholds[j]`.

    """
    scores, y, scored = _validated(scores, y, scored)
    thresholds = np.full(y.shape[1], np.inf)

    def tune(j, rows):
        thresholds[j] = best_f1_threshold(scores[rows, j], y[rows, j])
        return thresholds[j]

    _route(scores, scored, graph, classes, root, min_root, tune)
    return thresholds


def route(scores, thresholds, graph, classes, root=ROOT, min_root=0, scored=None):
    """
    Emulate `HierarchicalClassifier`'s multi-label walk on an all-node score matrix.

    Given the scores of a `-inf` prediction (every learned node visited), predicts class j for the
    samples with `scores[:, j] > thresholds[j]` that a predicted parent routes there. Children of
    the root (including a class that is also some other node's child) are considered for every
    sample, and `min_root` forces each sample's best scoring root children when fewer are
    predicted, as `mlb_min_root_predictions` does. Cells not in `scored` are placeholders for
    classes that were not learned and are never predicted nor ranked, matching the classifier,
    which routes only to children learned at fit. Nodes unreachable from `root` are never predicted.

    On a tree this is exactly what the classifier predicts with the same thresholds, which lets
    threshold policies be compared on held-out scores without refitting (tests pin it to `predict`).
    On a DAG it predicts a superset: the matrix holds each class's best score over all its parents,
    so a sample can be routed to a class here through a parent that rejects it at prediction time.

    Parameters
    ----------
    scores : array-like, shape = [n_samples, n_classes]
    thresholds : float or array-like, shape = [n_classes]
    graph : networkx.DiGraph
    classes : sequence
        The hierarchy node of each column of `scores`.
    root : node, optional
    min_root : int, optional
    scored : array-like of bool, shape = [n_samples, n_classes], optional

    Returns
    -------
    y_pred : ndarray of int, shape = [n_samples, n_classes]

    """
    scores = np.asarray(scores, dtype=np.float64)
    scored = np.ones(scores.shape, dtype=bool) if scored is None else np.asarray(scored, dtype=bool)
    thresholds = np.broadcast_to(np.asarray(thresholds, dtype=np.float64), (scores.shape[1],))
    return _route(scores, scored, graph, classes, root, min_root, lambda j, rows: thresholds[j]).astype(int)


def label_cardinality_threshold(scores, target_cardinality, candidates=None):
    """
    A single decision threshold matching the average number of labels per sample (label
    cardinality adjustment, Read et al. 2009): among `candidates` (default: the distinct scores),
    the threshold whose predicted label cardinality on `scores` is closest to `target_cardinality`,
    typically the cardinality of the training set.

    """
    scores = np.asarray(scores, dtype=np.float64)
    ranked = np.sort(scores, axis=None)[::-1]  # all cells, descending
    n_samples = scores.shape[0]
    if candidates is None:
        # Predicting the top k cells overall gives cardinality k / n_samples: pick k directly
        k = int(np.clip(round(target_cardinality * n_samples), 1, ranked.size))
        return float(np.nextafter(ranked[k - 1], -np.inf))
    thresholds = np.nextafter(np.asarray(candidates, dtype=np.float64), -np.inf)
    predicted = ranked.size - np.searchsorted(-ranked, -thresholds, side="right")  # cells strictly above
    return float(thresholds[np.argmin(np.abs(predicted / n_samples - target_cardinality))])


def best_f1_threshold(scores, y):
    """The threshold on `scores` maximising F1 against binary `y`; `inf` if `y` has no positives."""
    n_positive = int(np.sum(y))
    if n_positive == 0 or len(scores) == 0:
        return np.inf
    order = np.argsort(-scores, kind="stable")
    ranked_scores, true_positives = scores[order], np.cumsum(y[order])
    f1 = 2 * true_positives / (np.arange(1, len(scores) + 1) + n_positive)
    # A cut can only fall between distinct scores: tied scores are predicted together or not at all
    last_of_tie = np.append(ranked_scores[1:] != ranked_scores[:-1], True)
    best = int(np.argmax(np.where(last_of_tie, f1, -np.inf)))
    if best + 1 < len(ranked_scores):
        return float((ranked_scores[best] + ranked_scores[best + 1]) / 2)
    return float(np.nextafter(ranked_scores[best], -np.inf))


def _validated(scores, y, scored):
    scores, y = np.asarray(scores, dtype=np.float64), np.asarray(y)
    scored = np.ones(scores.shape, dtype=bool) if scored is None else np.asarray(scored, dtype=bool)
    if scores.ndim != 2 or scores.shape != y.shape or scored.shape != y.shape:
        raise ValueError(
            "`scores`, `y` and `scored` must share one shape (n_samples, n_classes); got "
            f"{scores.shape}, {y.shape} and {scored.shape}"
        )
    return scores, y, scored


def _column_index(classes, n_classes):
    if classes is None or len(classes) != n_classes:
        raise ValueError("`classes` must name the hierarchy node of every column of `scores`")
    return {node: j for j, node in enumerate(classes)}


def _route(scores, scored, graph, classes, root, min_root, threshold_for):
    """The top-down walk shared by `route` and `routed_thresholds`: classes are visited parents first,
    `threshold_for(column, rows)` is asked for each one's threshold given the scored rows its parents route
    to it, and the rows above that threshold are predicted. The root's children come first, all of them
    before the fallback forces `min_root` of them per sample. Returns the boolean prediction matrix."""
    column = _column_index(classes, scores.shape[1])
    predicted = np.zeros(scores.shape, dtype=bool)
    top_nodes = set(graph.successors(root)) if root in graph else {n for n, degree in graph.in_degree() if degree == 0}
    order = [node for node in topological_sort(graph) if node in column]
    top = sorted(column[node] for node in order if node in top_nodes)  # column order: how ties are broken
    for j in top:
        rows = np.flatnonzero(scored[:, j])
        predicted[rows, j] = scores[rows, j] > threshold_for(j, rows)
    if min_root and top:
        short = np.flatnonzero(predicted[:, top].sum(axis=1) < min_root)
        block = np.ix_(short, top)
        predicted[block] |= top_k_mask(np.where(scored[block], scores[block], -np.inf), min_root) & scored[block]
    for node in order:
        if node in top_nodes:
            continue
        parents = [column[parent] for parent in graph.predecessors(node) if parent in column]
        j = column[node]
        rows = np.flatnonzero(predicted[:, parents].any(axis=1) & scored[:, j])
        predicted[rows, j] = scores[rows, j] > threshold_for(j, rows)
    return predicted


def _populations(y, graph, classes, root):
    """Per column, the row indices to tune on: all rows, or the rows truly under the class's parent."""
    if graph is None:
        return [np.arange(y.shape[0])] * y.shape[1]
    column = _column_index(classes, y.shape[1])
    populations = []
    for node in classes:
        if node not in graph:
            # Not a hierarchy node: never scored, no threshold to tune
            populations.append(np.empty(0, dtype=np.intp))
            continue
        parents = _parents(graph, node, root)
        if not parents:
            populations.append(np.arange(y.shape[0]))
            continue
        # A sample is under the parent if it carries the parent or anything below it (labels may not be
        # ancestor-closed); on a DAG, under any of the parents.
        under = set()
        for parent in parents:
            under |= {parent} | descendants(graph, parent)
        under_columns = [column[n] for n in under if n in column]
        populations.append(np.flatnonzero(y[:, under_columns].any(axis=1)))
    return populations


def _parents(graph, node, root):
    """The parents that route to `node`: a root (the given sentinel, or any node without parents) does not."""
    return [p for p in graph.predecessors(node) if p != root and graph.in_degree(p) > 0]
