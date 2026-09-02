"""Unit-tests for the graph helpers."""

from hamcrest import assert_that, equal_to, is_
from networkx import DiGraph

from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.graph import rollup_nodes


def make_dag():
    r"""
            ROOT
           /    \
          A      B
         / \    / \
        1   C  C   2
            |
            3
    """
    return DiGraph([(ROOT, "A"), (ROOT, "B"), ("A", "1"), ("A", "C"), ("B", "C"), ("B", "2"), ("C", "3")])


def test_rollup_nodes_on_tree():
    graph = DiGraph([(ROOT, "A"), (ROOT, "B"), ("A", "1"), ("A", "2"), ("B", "3")])

    assert_that(rollup_nodes(graph, ROOT, ["1", "2", "3", "A"]), is_(equal_to([["A"], ["A"], ["B"], ["A"]])))
    assert_that(rollup_nodes(graph, "A", ["1", "2"]), is_(equal_to([["1"], ["2"]])))


def test_rollup_nodes_target_not_below_source_is_empty():
    graph = DiGraph([(ROOT, "A"), (ROOT, "B"), ("A", "1"), ("B", "3")])

    # the source itself, a sibling subtree, and an unknown label all roll up to nothing
    assert_that(rollup_nodes(graph, "A", ["A", "3", "zzz"]), is_(equal_to([[], [], []])))


def test_rollup_nodes_on_dag_returns_each_child_once():
    graph = make_dag()

    # "3" is reachable from ROOT through both A and B: both children, each exactly once
    assert_that(rollup_nodes(graph, ROOT, ["3", "C", "1"]), is_(equal_to([["A", "B"], ["A", "B"], ["A"]])))
    # from A there is a single child on the way to "3", even though A->C->3 is the only route
    assert_that(rollup_nodes(graph, "A", ["3"]), is_(equal_to([["C"]])))
