"""
Hamcrest matchers for unit-tests.

"""

from hamcrest.core.base_matcher import BaseMatcher
from networkx import is_isomorphic


class GraphMatcher(BaseMatcher):
    def __init__(self, graph):
        self.graph = graph

    def _matches(self, graph):
        return is_isomorphic(graph, self.graph)

    def describe_to(self, description):
        description.append_text("graph isomorphic to ").append_text(
            f"{type(self.graph).__name__} with {self.graph.number_of_nodes()} nodes "
            f"and {self.graph.number_of_edges()} edges"
        )


def matches_graph(graph):
    return GraphMatcher(graph)
