import json
from pathlib import Path

import numpy as np
import pytest

from ees_scientific_software_engineering.rms import rms

DATA_PATH = Path(__file__).parent / "data"


def test_rms():
    assert np.isclose(rms(np.array([4.0, 1.0, 8.0])), 5.1962)


def test_rms_bad_argument():
    with pytest.raises(TypeError, match="Argument should be a numpy array!"):
        rms("string instead of numpy array")


def test_rms_error_not_float():
    with pytest.raises(TypeError, match="Argument numpy array should contain float64 values!"):
        rms(np.array(["one", "two", "three"]))


def test_rms_from_data():
    with open(DATA_PATH / "test_rms.json") as f:
        data = json.load(f)
    input_array = np.array(data["array"])
    assert np.isclose(rms(input_array), data["expected"])


def test_rms_error_multi_dimensional():
    with pytest.raises(TypeError, match="Argument should be one dimensional array!"):
        rms(np.array([[4.0, 1.0, 8.0], [2.1, 1.2, 2.2]]))


def test_rms_error_contains_inf():
    with pytest.raises(ValueError, match="Argument array should not contain inf!"):
        rms(np.array([4.0, np.inf, 8.0]))


def test_rms_error_contains_minus_inf():
    with pytest.raises(ValueError, match="Argument array should not contain inf!"):
        rms(np.array([4.0, -np.inf, 8.0]))


def test_atleat_one_value():
    with pytest.raises(ValueError, match="Argument numpy array should contain at least one value!"):
        rms(np.array([], dtype=np.float64))


def test_rms_error_contains_nan():
    with pytest.raises(ValueError, match="Argument array should not contain nan!"):
        rms(np.array([4.0, np.nan, 8.0]))
