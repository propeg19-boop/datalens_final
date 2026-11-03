"""
tests/test_model_trainer.py
Tests for datalens.model_trainer automated baseline machine learning.
"""
import pytest
import numpy as np
import pandas as pd

from datalens.column_types import classify_all
from datalens.model_trainer import train_baseline_model, detect_target_task


def test_detect_target_task():
    task_num, err1 = detect_target_task("numeric", pd.Series([1.0, 2.0, 3.0]))
    assert task_num == "regression"
    assert err1 is None

    task_cat, err2 = detect_target_task("categorical", pd.Series(["A", "B", "A", "B"]))
    assert task_cat == "classification"
    assert err2 is None

    task_bad, err3 = detect_target_task("id", pd.Series(["ID1", "ID2"]))
    assert task_bad is None
    assert "not suitable" in err3


def test_train_regression_model():
    np.random.seed(42)
    n = 100
    x1 = np.linspace(1, 50, n)
    x2 = np.random.choice(["High", "Medium", "Low"], n)
    y = x1 * 3.5 + np.random.normal(0, 2, n)

    df = pd.DataFrame({"feat1": x1, "cat_feat": x2, "target": y})
    col_types = {"feat1": "numeric", "cat_feat": "categorical", "target": "numeric"}

    res = train_baseline_model(df, col_types, "target")
    assert "error" not in res
    assert res["task"] == "regression"
    assert len(res["metrics_table"]["models"]) == 3
    assert len(res["feature_importances"]) > 0
    assert "verdict" in res


def test_train_classification_model():
    np.random.seed(42)
    n = 100
    feat = np.linspace(10, 100, n)
    classes = ["Active" if f > 50 else "Inactive" for f in feat]

    df = pd.DataFrame({"feature": feat, "status": classes})
    col_types = {"feature": "numeric", "status": "categorical"}

    res = train_baseline_model(df, col_types, "status")
    assert "error" not in res
    assert res["task"] == "classification"
    assert len(res["metrics_table"]["models"]) == 3
    assert res["metrics_table"]["Accuracy"][2] > 0.80
