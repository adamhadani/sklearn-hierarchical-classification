"""
Graph processing helpers.

"""

from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from itertools import chain
from typing import Any

from networkx import DiGraph, descendants
from sklearn.preprocessing import MultiLabelBinarizer


def make_flat_hierarchy(targets: Iterable[Hashable], root: Hashable) -> dict[Hashable, list[Hashable]]:
    """Create a trivial (flat) hierarchy, linking all given targets to given root node."""
    adjacency: defaultdict[Hashable, list[Hashable]] = defaultdict(list)
    for target in targets:
        adjacency[root].append(target)
    return adjacency


def rollup_nodes(
    graph: DiGraph, source: Hashable, targets: Iterable[Any], mlb: MultiLabelBinarizer | None = None
) -> list[list[Hashable]]:
    """Perform a "roll-up" of given target nodes up to the nodes immediately below
    given source node in given graph.

    For each target, the result is the list of children of `source` under which the target lies
    (one child on a tree, possibly several on a DAG, each listed once). A target that is not a
    strict descendant of `source`, including `source` itself, rolls up to an empty list.

    When `mlb` is given, each target is a binary indicator row over `mlb.classes_` and the result
    is the concatenation of the roll-ups of every label set in it.

    """
    return rollup_targets(children_by_descendant(graph, source), targets, mlb=mlb)


def rollup_targets(
    child_of: Mapping[Hashable, Sequence[Hashable]], targets: Iterable[Any], mlb: MultiLabelBinarizer | None = None
) -> list[list[Hashable]]:
    """Roll `targets` up through a precomputed descendant -> children map (see `rollup_nodes`)."""
    if mlb is None:
        return [list(child_of.get(target, ())) for target in targets]
    return [[child for label in row.nonzero()[0] for child in child_of.get(mlb.classes_[label], ())] for row in targets]


def children_by_descendant(graph: DiGraph, source: Hashable) -> dict[Hashable, list[Hashable]]:
    """Map every strict descendant of `source` to the children of `source` it lies under."""
    child_of: dict[Hashable, list[Hashable]] = {}
    for child in graph.successors(source):
        for node in chain([child], descendants(graph, child)):
            child_of.setdefault(node, []).append(child)
    return child_of


def root_nodes(graph: DiGraph) -> Iterator[Hashable]:
    return (node for node, in_degree in graph.in_degree() if in_degree == 0)


def terminal_nodes(graph: DiGraph) -> Iterator[Hashable]:
    return (node for node, out_degree in graph.out_degree() if out_degree == 0)
