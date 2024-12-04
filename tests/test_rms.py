import numpy as np
import pytest

from ees_scientific_software_engineering.rms import rms


def test_rms():
    assert rms(np.array([2, 2, 2])) == 2
