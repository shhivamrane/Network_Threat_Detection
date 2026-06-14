"""Smoke tests for data loading and preprocessing."""
import numpy as np
from ml.data import make_synthetic
from ml.preprocessing.pipeline import prepare, _clean


def test_synthetic_shape():
    df = make_synthetic(500)
    assert "attack_class" in df.columns
    assert len(df) == 500
    assert df["attack_class"].nunique() <= 5


def test_clean_removes_constant_columns():
    df = make_synthetic(200)
    df["constant_col"] = 1
    cleaned = _clean(df)
    assert "constant_col" not in cleaned.columns


def test_prepare_outputs_aligned():
    df = make_synthetic(1000)
    X_tr, X_val, X_te, y_tr, y_val, y_te, pre = prepare(df, use_smote=False)
    assert X_tr.shape[0] == len(y_tr)
    assert X_te.shape[0] == len(y_te)
    assert X_tr.shape[1] == X_te.shape[1]  # same feature count
    assert not np.isnan(X_tr).any()