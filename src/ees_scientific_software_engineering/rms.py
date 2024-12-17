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

    if len(input_array.shape) != 1:
        raise TypeError("Argument should be one dimensional array!")

    if input_array.dtype != np.float64:
        raise TypeError("Argument numpy array should contain float64 values!")

    if len(input_array) == 0:
        raise ValueError("Argument numpy array should contain at least one value!")
        
    if np.isinf(input_array).any():
        raise ValueError("Argument array should not contain inf!")

    input_array_squared = input_array**2
    return np.sqrt(np.mean(input_array_squared))
