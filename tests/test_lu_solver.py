import numpy as np
import pytest

from ees_scientific_software_engineering.solvers import LUSolver


def test_lu_solver_correct():
    A = np.array([[2.0, 5.0, 8.0, 7.0], [5.0, 2.0, 2.0, 8.0], [7.0, 5.0, 6.0, 6.0], [5.0, 4.0, 4.0, 8.0]])
    b = np.array([1.0, 1.0, 1.0, 1.0])

    solver = LUSolver(A)

    x = solver.solve(b)

    np.allclose(A @ x - b, np.zeros((4,)))  # A == x - b (or close)


def test_lu_solver_matrix_not_array():
    with pytest.raises(TypeError, match="Argument should be a numpy array!"):
        solver = LUSolver("string")


def test_lu_solver_matrix_not_square_matrix():
    with pytest.raises(ValueError, match="Argument should be a square matrix!"):
        solver = LUSolver(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))


def test_lu_solver_matrix_not_float():
    with pytest.raises(ValueError, match="Argument should contain float64 values!"):
        solver = LUSolver(np.array([[1, 1], [1, 1]]))
