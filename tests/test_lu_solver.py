import numpy as np
import pytest

from ees_scientific_software_engineering.solvers import LUSolver


def test_lu_solver_correct():
    A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    b = np.array([1, 1, 1, 1])

    solver = LUSolver(A)

    x = solver.solve(b)

    np.allclose(A @ x - b, np.zeros((4,)))  # A == x - b (or close)


def test_lu_solver_matrix_not_array():
    with pytest.raises(TypeError, match="Argument should be a numpy array!"):
        solver = LUSolver("string")


def test_lu_solver_matrix_not_square_matrix():
    with pytest.raises(ValueError, match="Argument should be a square matrix!"):
        solver = LUSolver(np.array([[1, 1, 1], [1, 1, 1]]))
