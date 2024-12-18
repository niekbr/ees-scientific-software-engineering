"""Module to solve Ax=b"""

import numpy as np
import scipy


# pylint: disable=too-few-public-methods
class LUSolver:
    """Class to solve Ax=b using matrix factorization"""

    def __init__(self, input_matrix: np.ndarray):
        """
        Constructor of the class. It takes the input matrix and decompose it into LU factorization.
        Store the factorization and permutation into class members.
        """
        if not isinstance(input_matrix, np.ndarray):
            raise TypeError("Argument should be a numpy array!")

        if len(input_matrix) == 0:
            raise ValueError("Argument should contain at least 1 value!")

        if input_matrix.shape[0] != input_matrix.shape[1]:
            raise ValueError("Argument should be a square matrix!")

        if input_matrix.dtype != np.float64:
            raise ValueError("Argument should contain float64 values!")

        if np.isinf(input_matrix).any():
            raise ValueError("Argument should not contain inf values!")

        if np.isnan(input_matrix).any():
            raise ValueError("Argument should not contain nan values!")

        if scipy.linalg.det(input_matrix) == 0:
            raise ValueError("Argument should not be a singular matrix!")

        self._lu, self._piv = scipy.linalg.lu_factor(input_matrix)

    def solve(self, b: np.ndarray) -> np.ndarray:
        """
        Solve the linear equation with the input matrix and the given vector b.
        """
        if not isinstance(b, np.ndarray):
            raise TypeError("Argument should be a numpy array!")

        if len(b.shape) != 1:
            raise TypeError("Argument should be one dimensional array!")

        if b.dtype != np.float64:
            raise TypeError("Argument numpy array should contain float64 values!")

        if b.shape[0] != self._lu.shape[0]:
            raise ValueError("Argument should be the same size as the matrix!")

        if np.isinf(b).any():
            raise ValueError("Argument array should not contain inf!")

        if np.isnan(b).any():
            raise ValueError("Argument array should not contain nan!")

        return scipy.linalg.lu_solve((self._lu, self._piv), b)
