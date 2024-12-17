import numpy as np
import scipy


class LUSolver:
    def __init__(self, input_matrix: np.ndarray):
        """
        Constructor of the class. It takes the input matrix and decompose it into LU factorization.
        Store the factorization and permutation into class members.
        """
        if not isinstance(input_matrix, np.ndarray):
            raise TypeError("Argument should be a numpy array!")

        self._lu, self._piv = scipy.linalg.lu_factor(input_matrix)

    def solve(self, b: np.ndarray) -> np.ndarray:
        """
        Solve the linear equation with the input matrix and the given vector b.
        """
        return scipy.linalg.lu_solve((self._lu, self._piv), b)
