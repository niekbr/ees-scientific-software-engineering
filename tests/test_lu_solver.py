import numpy as np
import pytest

from ees_scientific_software_engineering.solvers import LUSolver


def correct_matrix():
    return np.array([[2.0, 5.0, 8.0, 7.0], [5.0, 2.0, 2.0, 8.0], [7.0, 5.0, 6.0, 6.0], [5.0, 4.0, 4.0, 8.0]])


def test_lu_solver_correct():
    A = correct_matrix()
    b = np.array([1.0, 1.0, 1.0, 1.0])

    A = correct_matrix()
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


def test_lu_solver_matrix_not_empty():
    with pytest.raises(ValueError, match="Argument should contain at least 1 value!"):
        solver = LUSolver(np.array([], dtype=np.float64))


def test_lu_solver_matrix_not_inf():
    with pytest.raises(ValueError, match="Argument should not contain inf values!"):
        solver = LUSolver(np.array([[1.0, 1.0, 1.0], [1.0, np.inf, 1.0], [1.0, 1.0, 1.0]]))


def test_lu_solver_matrix_not_nan():
    with pytest.raises(ValueError, match="Argument should not contain nan values!"):
        solver = LUSolver(np.array([[1.0, 1.0, 1.0], [1.0, np.nan, 1.0], [1.0, 1.0, 1.0]]))


def test_lu_solver_matrix_not_singular():
    with pytest.raises(ValueError, match="Argument should not be a singular matrix!"):
        solver = LUSolver(np.array([[1.0, 2.0], [2.0, 4.0]]))


def test_lu_solver_vector_type_not_array():
    A = correct_matrix()
    solver = LUSolver(A)

    with pytest.raises(TypeError, match="Argument should be a numpy array!"):
        result = solver.solve("not a np array")


def test_lu_solver_vector_type_not_1d():
    A = correct_matrix()
    solver = LUSolver(A)

    with pytest.raises(TypeError, match="Argument should be one dimensional array!"):
        result = solver.solve(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))


def test_lu_solver_vector_not_float64():
    A = correct_matrix()
    solver = LUSolver(A)

    with pytest.raises(TypeError, match="Argument numpy array should contain float64 values!"):
        result = solver.solve(np.array([1, 1, 1]))


def test_lu_solver_vector_not_same_size_as_matrix():
    A = correct_matrix()
    solver = LUSolver(A)

    with pytest.raises(ValueError, match="Argument should be the same size as the matrix!"):
        result = solver.solve(np.array([1.0, 1.0, 1.0]))


def test_lu_solver_vector_has_inf():
    A = correct_matrix()
    solver = LUSolver(A)

    with pytest.raises(ValueError, match="Argument array should not contain inf!"):
        result = solver.solve(np.array([1.0, np.inf, 1.0, 1.0]))


def test_lu_solver_vector_has_nan():
    A = correct_matrix()
    solver = LUSolver(A)

    with pytest.raises(ValueError, match="Argument array should not contain nan!"):
        result = solver.solve(np.array([1.0, np.nan, 1.0, 1.0]))
