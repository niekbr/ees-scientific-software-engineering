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


def test_rms_from_data():
    with open(DATA_PATH / "test_rms.json") as f:
        data = json.load(f)
    input_array = np.array(data["array"])
    assert np.isclose(rms(input_array), data["expected"])
