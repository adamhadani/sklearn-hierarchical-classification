"""
Graph processing helpers.

"""

from collections import defaultdict
from itertools import chain

from networkx import descendants
from numpy import ndarray


def make_flat_hierarchy(targets, root):
    """Create a trivial (flat) hiearchy, linking all given targets to given root node."""
    adjacency = defaultdict(list)
    for target in targets:
        adjacency[root].append(target)
    return adjacency


def rollup_nodes(graph, source, targets, mlb=None):
    """Perform a "roll-up" of given target nodes up to the nodes immediately below
    given source node in given graph.

    For each target, the result is the list of children of `source` under which the target lies
    (one child on a tree, possibly several on a DAG, each listed once). A target that is not a
    strict descendant of `source`, including `source` itself, rolls up to an empty list.

    When `mlb` is given, each target is a binary indicator row over `mlb.classes_` and the result
    is the concatenation of the roll-ups of every label set in it.

    """
    child_of = children_by_descendant(graph, source)
    resultset = []
    for node_id in targets:
        if mlb is not None and isinstance(node_id, ndarray):
            result_row = []
            for label in node_id.nonzero()[0]:
                result_row.extend(child_of.get(mlb.classes_[label], ()))
            resultset.append(result_row)
        else:
            resultset.append(list(child_of.get(node_id, ())))

    return resultset


def children_by_descendant(graph, source):
    """Map every strict descendant of `source` to the children of `source` it lies under."""
    child_of = {}
    for child in graph.successors(source):
        for node in chain([child], descendants(graph, child)):
            child_of.setdefault(node, []).append(child)
    return child_of


def root_nodes(graph):
    return (node for node, in_degree in graph.in_degree() if in_degree == 0)


def terminal_nodes(graph):
    return (node for node, out_degree in graph.out_degree() if out_degree == 0)
