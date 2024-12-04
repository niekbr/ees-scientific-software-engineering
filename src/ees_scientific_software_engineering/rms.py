"""A module that calculates the root mean square
"""

import numpy as np


def rms(input_array: np.ndarray) -> float:
    input_array_squared = input_array**2
    return np.sqrt(np.mean(input_array_squared))
