import numpy as np
import pytest

from ees_scientific_software_engineering.graph_processing import *

###############
# Constructor #
###############


def test_constructor_simple_network():
    graph_processor = GraphProcessor(
        [1, 2, 3, 4, 5],
        [12, 23, 24, 45, 51],
        [(1, 2), (2, 3), (2, 4), (4, 5), (5, 1)],
        [True, True, True, True, False],
        1,
    )

    assert len(graph_processor._network.nodes) == 5
    assert len(graph_processor._network.edges) == 4


def test_constructor_unique_vertex():
    with pytest.raises(IDNotUniqueError, match="Not all vertex_ids are unique"):
        graph_processor = GraphProcessor(
            [1, 2, 3, 4, 4],
            [12, 23, 34, 45, 51],
            [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)],
            [True, True, True, True, False],
            1,
        )


def test_constructor_unique_edge():
    with pytest.raises(IDNotUniqueError, match="Not all edge_ids are unique"):
        graph_processor = GraphProcessor(
            [1, 2, 3, 4, 5],
            [12, 23, 34, 45, 45],
            [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)],
            [True, True, True, True, False],
            1,
        )


def test_constructor_edge_ids_and_pairs_length():
    with pytest.raises(
        InputLengthDoesNotMatchError, match="edge_ids should be the same length as edge_vertex_id_pairs"
    ):
        graph_processor = GraphProcessor(
            [1, 2, 3, 4, 5],
            [12, 23, 34, 45],
            [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)],
            [True, True, True, True, False],
            1,
        )


def test_constructor_edge_ids_and_enabled_length():
    with pytest.raises(InputLengthDoesNotMatchError, match="edge_ids should be the same length as edge_enabled"):
        graph_processor = GraphProcessor(
            [1, 2, 3, 4, 5],
            [12, 23, 34, 45, 51],
            [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)],
            [
                True,
                True,
                True,
                True,
            ],
            1,
        )


def test_constructor_pairs_vertex_not_exist():
    with pytest.raises(IDNotFoundError, match="One of values in edge_vertex_id_pairs not found in vertex_ids"):
        graph_processor = GraphProcessor(
            [1, 2, 3, 4, 5],
            [12, 23, 34, 45, 51],
            [(1, 2), (2, 3), (3, 4), (4, 5), (6, 1)],
            [True, True, True, True, False],
            1,
        )


def test_constructor_source_vertex_id_not_found():
    with pytest.raises(IDNotFoundError, match="source_vertex_id not found in vertex_ids"):
        graph_processor = GraphProcessor(
            [1, 2, 3, 4, 5],
            [12, 23, 34, 45, 51],
            [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)],
            [True, True, True, True, False],
            6,
        )


def test_constructor_fully_connected():
    with pytest.raises(GraphNotFullyConnectedError):
        graph_processor = GraphProcessor(
            [1, 2, 3, 4, 5],
            [12, 23, 34],
            [(1, 2), (2, 3), (3, 4)],
            [True, True, True],
            1,
        )


def test_constructor_cycles():
    with pytest.raises(GraphCycleError):
        graph_processor = GraphProcessor(
            [1, 2, 3, 4, 5],
            [12, 23, 34, 45, 51],
            [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)],
            [True, True, True, True, True],
            1,
        )


##############
# Downstream #
##############
def test_find_downstream_vertices_simple_network():
    graph_processor = GraphProcessor(
        [1, 2, 3, 4, 5],
        [12, 23, 24, 45, 51],
        [(1, 2), (2, 3), (2, 4), (4, 5), (5, 1)],
        [True, True, True, True, False],
        1,
    )

    result = graph_processor.find_downstream_vertices(24)

    assert result == [4, 5]


def test_find_downstream_vertices_edge_not_exist():
    graph_processor = GraphProcessor(
        [1, 2, 3, 4, 5],
        [12, 23, 24, 45, 51],
        [(1, 2), (2, 3), (2, 4), (4, 5), (5, 1)],
        [True, True, True, True, False],
        1,
    )

    with pytest.raises(IDNotFoundError, match="edge_id not found in edge_ids"):
        result = graph_processor.find_downstream_vertices(88)


def test_find_downstream_vertices_edge_disabled():
    graph_processor = GraphProcessor(
        [1, 2, 3, 4, 5],
        [12, 23, 24, 45, 51],
        [(1, 2), (2, 3), (2, 4), (4, 5), (5, 1)],
        [True, True, True, True, False],
        1,
    )

    result = graph_processor.find_downstream_vertices(51)

    assert result == []


###############
# Alternative #
###############
def test_find_alternative_vertices_simple_network():
    graph_processor = GraphProcessor(
        [0, 2, 10, 4, 6],
        [1, 9, 7, 3, 8, 5],
        [(0, 2), (2, 10), (2, 4), (0, 4), (4, 6), (0, 6)],
        [True, True, False, True, False, True],
        0,
    )

    assert graph_processor.find_alternative_edges(1) == [7]
    assert graph_processor.find_alternative_edges(3) == [7, 8]
    assert graph_processor.find_alternative_edges(5) == [8]
    assert graph_processor.find_alternative_edges(9) == []


def test_find_alternative_vertices_edge_not_found():
    graph_processor = GraphProcessor(
        [0, 2, 10, 4, 6],
        [1, 9, 7, 3, 8, 5],
        [(0, 2), (2, 10), (2, 4), (0, 4), (4, 6), (0, 6)],
        [True, True, False, True, False, True],
        0,
    )

    with pytest.raises(IDNotFoundError, match="disabled_edge_id not found in edge_ids"):
        result = graph_processor.find_alternative_edges(88)


def test_find_alternative_vertices_already_disabled():
    graph_processor = GraphProcessor(
        [0, 2, 10, 4, 6],
        [1, 9, 7, 3, 8, 5],
        [(0, 2), (2, 10), (2, 4), (0, 4), (4, 6), (0, 6)],
        [True, True, False, True, False, True],
        0,
    )

    with pytest.raises(EdgeAlreadyDisabledError):
        result = graph_processor.find_alternative_edges(7)
