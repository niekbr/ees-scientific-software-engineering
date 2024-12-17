from ees_scientific_software_engineering.LUSolver import LUSolver
import numpy as np

def test_lu_solver():
    A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    b = np.array([1, 1, 1, 1])

    solver = LUSolver(A)

    x = LUSolver.solve(b)

    np.allclose(A @ x - b, np.zeros((4,)))  # A == x - b (or close)
