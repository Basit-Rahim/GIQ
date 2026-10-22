"""Checks on the trained model.  Run with: python -m pytest tests -q"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C
from src.data import load_training_table
from src.models import predict


def _bundle():
    return joblib.load(C.MODEL_BUNDLE)


def test_predictions_are_probabilities():
    out = predict(_bundle(), load_training_table())
    assert out["enrollment_prob"].between(0, 1).all()
    assert out["enrollment_prob"].std() > 0


def test_income_is_in_rupees():
    bundle = _bundle()
    income = bundle["feature_columns"].index("monthly_income")
    assert bundle["stage1"]["mean"][income] > 1_000


def test_unsafe_route_adjustment():
    row = load_training_table().iloc[[0]].assign(route_safe=0)
    out = predict(_bundle(), row).iloc[0]
    assert np.isclose(out["enrollment_prob"], out["model_prob"] * C.UNSAFE_ROUTE_MULTIPLIER)


def test_training_table_has_no_identifiers():
    columns = set(pd.read_csv(C.TRAINING_TABLE, nrows=1).columns)
    assert columns == set(C.FEATURE_COLUMNS) | set(C.STAGE1_TARGETS) | {
        C.TARGET_COLUMN, "n_school_age", "school_info_imputed"}
