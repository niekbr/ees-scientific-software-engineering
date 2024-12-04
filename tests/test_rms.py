import numpy as np
import pytest

from ees_scientific_software_engineering.rms import rms


def test_rms():
    assert np.isclose(rms(np.array([4.0, 1.0, 8.0])), 5.1962)
