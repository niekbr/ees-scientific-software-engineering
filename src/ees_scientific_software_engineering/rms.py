"""A module that calculates the root mean square
"""

import numpy as np


def rms(input_array: np.ndarray) -> float:
    """calculating rms

    Args:
        input_array: numpy array containing floats

    Returns:
        rms of numbers in array
    """
    if not isinstance(input_array, np.ndarray):
        raise TypeError("Argument should be a numpy array!")

    input_array_squared = input_array**2
    return np.sqrt(np.mean(input_array_squared))
