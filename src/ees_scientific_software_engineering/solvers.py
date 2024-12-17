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

        if input_matrix.shape[0] != input_matrix.shape[1]:
            raise ValueError("Argument should be a square matrix!")

        self._lu, self._piv = scipy.linalg.lu_factor(input_matrix)

    def solve(self, b: np.ndarray) -> np.ndarray:
        """
        Solve the linear equation with the input matrix and the given vector b.
        """
        return scipy.linalg.lu_solve((self._lu, self._piv), b)

    # def x(self):
    #     """Fixes min of 2 methods"""
    #     return 'x'
