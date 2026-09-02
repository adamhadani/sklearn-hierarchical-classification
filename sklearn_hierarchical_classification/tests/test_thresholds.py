"""Unit-tests for the decision-threshold tuning helpers."""

import numpy as np
from hamcrest import assert_that, close_to, equal_to, is_
from networkx import DiGraph

from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.thresholds import (
    label_cardinality_threshold,
    scut_thresholds,
)


def test_scut_threshold_separates_positives_from_negatives():
    scores = np.array([[0.9], [0.8], [0.7], [0.2], [0.1]])
    y = np.array([[1], [1], [1], [0], [0]])

    thresholds = scut_thresholds(scores, y)

    assert_that(thresholds.shape, is_(equal_to((1,))))
    assert_that(0.2 < thresholds[0] < 0.7, is_(True))


def test_scut_threshold_is_infinite_for_a_class_without_positives():
    scores = np.array([[0.9, 0.3], [0.1, 0.4]])
    y = np.array([[1, 0], [0, 0]])

    thresholds = scut_thresholds(scores, y)

    assert_that(np.isinf(thresholds[1]), is_(True))


def test_scut_thresholds_can_be_tuned_locally_under_each_parent():
    r"""With a hierarchy, class c's threshold is tuned only on the samples that truly lie under c's
    parent, since that is the population the local classifier at the parent was trained on.

            ROOT
           /    \
          A      B
          |
          a1
    """
    graph = DiGraph([(ROOT, "A"), (ROOT, "B"), ("A", "a1")])
    classes = ["A", "B", "a1"]
    # a1's classifier gives meaningless scores to the B samples it never saw; here they fall between
    # the two true a1 samples, so a globally tuned threshold gives up the second positive
    scores = np.array(
        [
            [0.9, 0.1, 0.8],  # A, a1
            [0.9, 0.1, 0.6],  # A, a1
            [0.9, 0.1, 0.3],  # A only
            [0.9, 0.1, 0.2],  # A only
            [0.1, 0.9, 0.70],  # B: out-of-subtree noise for a1
            [0.1, 0.9, 0.66],  # B
            [0.1, 0.9, 0.64],  # B
        ]
    )
    y = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])

    local = scut_thresholds(scores, y, graph=graph, classes=classes)
    global_ = scut_thresholds(scores, y)

    assert_that(0.3 < local[2] < 0.6, is_(True))  # tuned on the four A samples only: both positives kept
    assert_that(global_[2] > 0.6, is_(True))  # polluted by the B scores: the 0.6 positive is dropped


def test_label_cardinality_threshold_matches_target_cardinality():
    scores = np.array([[0.9, 0.6, 0.1], [0.8, 0.3, 0.2], [0.7, 0.5, 0.4], [0.95, 0.05, 0.0]])

    threshold = label_cardinality_threshold(scores, target_cardinality=1.0)

    assert_that((scores > threshold).sum(1).mean(), is_(close_to(1.0, delta=0.01)))
    assert_that(
        (scores > label_cardinality_threshold(scores, target_cardinality=2.0)).sum(1).mean(),
        is_(close_to(2.0, delta=0.01)),
    )
