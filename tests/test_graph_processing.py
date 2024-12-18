import numpy as np
import pytest

from ees_scientific_software_engineering.graph_processing import *


def test_constructor_simple_network():
    graph_processor = GraphProcessor(
        [1, 2, 3, 4, 5],
        [12, 23, 34, 45, 51],
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)],
        [True, True, True, True, False],
        1,
    )

    assert len(graph_processor.G.nodes) == 5
    assert len(graph_processor.G.edges) == 4

    graph_processor.find_alternative_edges(1)
    graph_processor.find_downstream_vertices(1)


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
    with pytest.raises(InputLengthDoesNotMatchError, match="edge_ids should be the same length as edge_vertex_id_pairs"):
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
            [True, True, True, True,],
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
