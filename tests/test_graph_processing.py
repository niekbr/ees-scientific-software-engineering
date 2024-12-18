import numpy as np
import pytest

from ees_scientific_software_engineering.graph_processing import GraphProcessor


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
