"""Decision-threshold tuning for multi-label prediction scores."""

import numpy as np
from networkx import descendants

from sklearn_hierarchical_classification.constants import ROOT


def scut_thresholds(scores, y, graph=None, classes=None, root=ROOT):
    """
    Per-class decision thresholds maximising F1 on held-out scores (SCut, Yang 1999).

    Parameters
    ----------
    scores : array-like, shape = [n_samples, n_classes]
        Held-out (e.g. out-of-fold) scores, one column per class, as returned by
        `HierarchicalClassifier.predict_proba` with `mlb_prediction_threshold=-np.inf`.

    y : array-like, shape = [n_samples, n_classes]
        Binary indicator of the true labels, columns aligned with `scores`.

    graph : networkx.DiGraph, optional
        The class hierarchy. When given (together with `classes`), each class's threshold is
        tuned only on the samples that truly lie under the class's parent: that is the population
        the local classifier at the parent was trained on, and scores it gives to other samples are
        not meaningful. Without a graph every threshold is tuned on all samples.

    classes : sequence, optional
        The hierarchy node of each column of `scores` (e.g. `mlb.classes_`); required with `graph`.

    Returns
    -------
    thresholds : ndarray, shape = [n_classes]
        Predict class j where `scores[:, j] > thresholds[j]`; `inf` for classes without positives.

    """
    scores, y = np.asarray(scores, dtype=np.float64), np.asarray(y)
    population = _populations(y, graph, classes, root)
    return np.array([best_f1_threshold(scores[rows, j], y[rows, j]) for j, rows in enumerate(population)])


def label_cardinality_threshold(scores, target_cardinality, candidates=None):
    """
    A single decision threshold matching the average number of labels per sample (label
    cardinality adjustment, Read et al. 2009): among `candidates` (default: the distinct scores),
    the threshold whose predicted label cardinality on `scores` is closest to `target_cardinality`,
    typically the cardinality of the training set.

    """
    scores = np.asarray(scores, dtype=np.float64)
    candidates = np.unique(scores) if candidates is None else np.asarray(candidates, dtype=np.float64)
    # Slightly below each candidate, so that a candidate score itself counts as predicted
    thresholds = np.nextafter(candidates, -np.inf)
    cardinality = np.array([(scores > t).sum(axis=1).mean() for t in thresholds])
    return float(thresholds[np.argmin(np.abs(cardinality - target_cardinality))])


def best_f1_threshold(scores, y):
    """The threshold on `scores` maximising F1 against binary `y`; `inf` if `y` has no positives."""
    n_positive = int(np.sum(y))
    if n_positive == 0 or len(scores) == 0:
        return np.inf
    order = np.argsort(-scores, kind="stable")
    ranked_scores, true_positives = scores[order], np.cumsum(y[order])
    f1 = 2 * true_positives / (np.arange(1, len(scores) + 1) + n_positive)
    best = int(np.argmax(f1))
    if best + 1 < len(ranked_scores):
        return float((ranked_scores[best] + ranked_scores[best + 1]) / 2)
    return float(np.nextafter(ranked_scores[best], -np.inf))


def _populations(y, graph, classes, root):
    """Per column, the row indices to tune on: all rows, or the rows truly under the class's parent."""
    if graph is None:
        return [np.arange(y.shape[0])] * y.shape[1]
    if classes is None or len(classes) != y.shape[1]:
        raise ValueError("`classes` must name the hierarchy node of every column of `scores` when `graph` is given")
    column = {node: j for j, node in enumerate(classes)}
    populations = []
    for node in classes:
        parents = [parent for parent in graph.predecessors(node) if parent != root]
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
