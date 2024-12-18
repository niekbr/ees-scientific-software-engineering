"""
This is for the graph processing assignment
"""

from typing import List, Tuple

import networkx as nx


class IDNotFoundError(Exception):
    pass


class InputLengthDoesNotMatchError(Exception):
    pass


class IDNotUniqueError(Exception):
    pass


class GraphNotFullyConnectedError(Exception):
    pass


class GraphCycleError(Exception):
    pass


class EdgeAlreadyDisabledError(Exception):
    pass


class GraphProcessor:
    """
    A Graph Processor to check the graph validity, find downstream vertices and find alternative edges
    """

    def __init__(
        self,
        vertex_ids: List[int],
        edge_ids: List[int],
        edge_vertex_id_pairs: List[Tuple[int, int]],
        edge_enabled: List[bool],
        source_vertex_id: int,
    ) -> None:
        """
        Initialize a graph processor object with an undirected graph.
        Only the edges which are enabled are taken into account.
        Check if the input is valid and raise exceptions if not.

        Args:
            vertex_ids: list of vertex ids
            edge_ids: list of edge ids
            edge_vertex_id_pairs: list of tuples of two integer
                Each tuple is a vertex id pair of the edge.
            edge_enabled: list of bools indicating of an edge is enabled or not
            source_vertex_id: vertex id of the source in the graph
        """
        if len(vertex_ids) > len(set(vertex_ids)):
            raise IDNotUniqueError("Not all vertex_ids are unique")

        if len(edge_ids) > len(set(edge_ids)):
            raise IDNotUniqueError("Not all edge_ids are unique")

        if len(edge_ids) != len(edge_vertex_id_pairs):
            raise InputLengthDoesNotMatchError("edge_ids should be the same length as edge_vertex_id_pairs")

        edge_vertex_ids_1, edge_vertex_ids_2 = zip(*edge_vertex_id_pairs)
        edge_vertex_ids_all = edge_vertex_ids_1 + edge_vertex_ids_2
        x = set(edge_vertex_ids_all)
        if not set(edge_vertex_ids_all).issubset(vertex_ids):
            raise IDNotFoundError("One of values in edge_vertex_id_pairs not found in vertex_ids")

        if len(edge_ids) != len(edge_enabled):
            raise InputLengthDoesNotMatchError("edge_ids should be the same length as edge_enabled")

        if source_vertex_id not in vertex_ids:
            raise IDNotFoundError("source_vertex_id not found in vertex_ids")

        self.G = nx.Graph()
        self.G.add_nodes_from(vertex_ids)

        for i, vertex_pair in enumerate(edge_vertex_id_pairs):
            if not edge_enabled[i]:
                continue

            self.G.add_edge(vertex_pair[0], vertex_pair[1], id=edge_ids[i])

        if not nx.is_connected(self.G):
            raise GraphNotFullyConnectedError()

        if len(list(nx.simple_cycles(self.G))) > 0:
            raise GraphCycleError()

    def find_downstream_vertices(self, edge_id: int) -> List[int]:
        """
        Given an edge id, return all the vertices which are in the downstream of the edge,
            with respect to the source vertex.
            Including the downstream vertex of the edge itself!

        Only enabled edges should be taken into account in the analysis.
        If the given edge_id is a disabled edge, it should return empty list.
        If the given edge_id does not exist, it should raise IDNotFoundError.


        For example, given the following graph (all edges enabled):

            vertex_0 (source) --edge_1-- vertex_2 --edge_3-- vertex_4

        Call find_downstream_vertices with edge_id=1 will return [2, 4]
        Call find_downstream_vertices with edge_id=3 will return [4]

        Args:
            edge_id: edge id to be searched

        Returns:
            A list of all downstream vertices.
        """
        # put your implementation here
        pass

    def find_alternative_edges(self, disabled_edge_id: int) -> List[int]:
        """
        Given an enabled edge, do the following analysis:
            If the edge is going to be disabled,
                which (currently disabled) edge can be enabled to ensure
                that the graph is again fully connected and acyclic?
            Return a list of all alternative edges.
        If the disabled_edge_id is not a valid edge id, it should raise IDNotFoundError.
        If the disabled_edge_id is already disabled, it should raise EdgeAlreadyDisabledError.
        If there are no alternative to make the graph fully connected again, it should return empty list.


        For example, given the following graph:

        vertex_0 (source) --edge_1(enabled)-- vertex_2 --edge_9(enabled)-- vertex_10
                 |                               |
                 |                           edge_7(disabled)
                 |                               |
                 -----------edge_3(enabled)-- vertex_4
                 |                               |
                 |                           edge_8(disabled)
                 |                               |
                 -----------edge_5(enabled)-- vertex_6

        Call find_alternative_edges with disabled_edge_id=1 will return [7]
        Call find_alternative_edges with disabled_edge_id=3 will return [7, 8]
        Call find_alternative_edges with disabled_edge_id=5 will return [8]
        Call find_alternative_edges with disabled_edge_id=9 will return []

        Args:
            disabled_edge_id: edge id (which is currently enabled) to be disabled

        Returns:
            A list of alternative edge ids.
        """
        # put your implementation here
        pass
