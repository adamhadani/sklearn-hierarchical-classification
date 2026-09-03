"""Unit-tests for the decision-threshold tuning helpers."""

import time

import numpy as np
from hamcrest import assert_that, close_to, equal_to, is_
from networkx import DiGraph

from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.thresholds import (
    best_f1_threshold,
    label_cardinality_threshold,
    routed_thresholds,
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


def test_best_f1_threshold_keeps_tied_positives():
    scores = np.array([0.9, 0.5, 0.5, 0.1])
    y = np.array([1, 1, 0, 0])

    threshold = best_f1_threshold(scores, y)

    assert_that((scores > threshold).tolist(), is_(equal_to([True, True, True, False])))


def test_scut_thresholds_ignore_classes_missing_from_the_hierarchy():
    graph = DiGraph([(ROOT, "A"), (ROOT, "B")])
    scores = np.array([[0.9, 0.1, 0.5], [0.2, 0.8, 0.5]])
    y = np.array([[1, 0, 0], [0, 1, 0]])

    thresholds = scut_thresholds(scores, y, graph=graph, classes=["A", "B", "zzz"])

    assert_that(np.isinf(thresholds[2]), is_(True))
    assert_that(0.2 < thresholds[0] < 0.9, is_(True))


def test_scut_thresholds_treat_any_in_degree_zero_node_as_a_root():
    """A hierarchy rooted at a custom node: its children are tuned on all samples, not only on the
    samples carrying some label."""
    graph = DiGraph([("top", "A"), ("top", "B")])
    scores = np.array([[0.9, 0.1], [0.2, 0.1], [0.3, 0.1]])
    y = np.array([[1, 0], [0, 0], [0, 0]])

    thresholds = scut_thresholds(scores, y, graph=graph, classes=["A", "B"])

    assert_that(0.5 < thresholds[0] < 0.7, is_(True))  # midpoint of 0.9 and 0.3: the unlabeled rows count


def test_label_cardinality_threshold_is_fast_on_distinct_scores():
    """All-distinct scores (e.g. SVM decision values) must not turn every cell into a candidate."""
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(2_000, 50))

    start = time.perf_counter()
    threshold = label_cardinality_threshold(scores, target_cardinality=3.0)

    assert_that(time.perf_counter() - start < 2.0, is_(True))
    assert_that((scores > threshold).sum(1).mean(), is_(close_to(3.0, delta=0.01)))


def test_scut_thresholds_ignore_unscored_cells():
    """Cells that were never scored (e.g. a class absent from a CV fold's training part, so its
    held-out rows keep the zero placeholder) must not take part in that class's tuning."""
    scores = np.array([[0.9], [0.2], [0.1], [0.0], [0.0], [0.0]])
    y = np.array([[1], [0], [0], [1], [1], [1]])
    scored = np.array([[True], [True], [True], [False], [False], [False]])

    with_mask = scut_thresholds(scores, y, scored=scored)
    without_mask = scut_thresholds(scores, y)

    assert_that(0.2 < with_mask[0] < 0.9, is_(True))  # tuned on the three scored rows
    assert_that(without_mask[0] < 0.0, is_(True))  # the zero-scored positives drag the cut below the zeros


def test_routed_thresholds_tune_a_child_on_the_rows_its_parent_routes_to_it():
    r"""Sequential top-down tuning: B's threshold is tuned on the rows that pass A's threshold, which
    includes a false positive of A that the true-population SCut never sees.

            ROOT
             |
             A
             |
             B
    """
    graph = DiGraph([(ROOT, "A"), ("A", "B")])
    classes = ["A", "B"]
    scores = np.array(
        [
            [2.0, 3.0],  # A, B
            [2.0, 3.0],  # A, B
            [2.0, 1.0],  # A only
            [2.0, 1.0],  # A only
            [2.0, 2.5],  # nothing: A's false positive, routed on to B where it scores between B's classes
            [-1.0, 0.0],  # nothing
        ]
    )
    y = np.array([[1, 1], [1, 1], [1, 0], [1, 0], [0, 0], [0, 0]])

    routed = routed_thresholds(scores, y, graph=graph, classes=classes)
    local = scut_thresholds(scores, y, graph=graph, classes=classes)

    assert_that(routed[0], is_(close_to(0.5, delta=1e-9)))  # A is tuned on every row either way
    assert_that(local[0], is_(close_to(0.5, delta=1e-9)))
    assert_that(local[1], is_(close_to(2.0, delta=1e-9)))  # midpoint of B's scores among true A rows
    assert_that(routed[1], is_(close_to(2.75, delta=1e-9)))  # lifted above the routed false positive


def test_routed_thresholds_ignore_unscored_cells():
    graph = DiGraph([(ROOT, "A"), ("A", "B")])
    scores = np.array([[2.0, 3.0], [2.0, 3.0], [2.0, 1.0], [2.0, 1.0], [2.0, 2.5], [-1.0, 0.0]])
    y = np.array([[1, 1], [1, 1], [1, 0], [1, 0], [0, 0], [0, 0]])
    scored = np.ones(scores.shape, dtype=bool)
    scored[4, 1] = False

    routed = routed_thresholds(scores, y, graph=graph, classes=["A", "B"], scored=scored)

    assert_that(routed[1], is_(close_to(2.0, delta=1e-9)))


def test_routed_thresholds_are_infinite_below_a_class_that_is_never_predicted():
    graph = DiGraph([(ROOT, "A"), ("A", "B")])
    scores = np.array([[0.9, 0.9], [0.8, 0.8]])
    y = np.array([[0, 1], [0, 0]])  # B without A is possible in the data, but A is never predicted

    routed = routed_thresholds(scores, y, graph=graph, classes=["A", "B"])

    assert_that(np.isinf(routed).tolist(), is_(equal_to([True, True])))
