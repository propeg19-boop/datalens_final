"""
tests/test_cleaning_assistant.py
Tests for datalens.cleaning_assistant matching PRD Section 5.4 decision table.
"""
import pytest
import numpy as np
import pandas as pd

from datalens.column_types import classify_all
from datalens.data_quality import run_data_detective
from datalens.cleaning_assistant import run_cleaning_assistant


def test_missing_decision_table_full_coverage():
    # Build dataframe exercising every branch of the decision matrix
    n = 100
    df = pd.DataFrame({
        # <= 5% missing numeric -> impute_median
        "num_low": [np.nan if i < 3 else float(i) for i in range(n)],
        # 5-15% missing numeric -> impute_knn
        "num_mid": [np.nan if i < 10 else float(i * 2) for i in range(n)],
        # > 15% missing numeric -> manual
        "num_high": [np.nan if i < 25 else float(i * 3) for i in range(n)],
        # <= 5% missing categorical -> impute_mode
        "cat_low": [None if i < 4 else "Category_A" for i in range(n)],
        # > 5% missing categorical -> manual
        "cat_high": [None if i < 12 else "Category_B" for i in range(n)],
        # <= 5% missing id -> manual (never auto-fill id)
        "id_col": [None if i < 2 else f"ID_{i}" for i in range(n)],
    })

    col_types = {
        "num_low": "numeric",
        "num_mid": "numeric",
        "num_high": "numeric",
        "cat_low": "categorical",
        "cat_high": "categorical",
        "id_col": "id",
    }

    report = run_data_detective(df, col_types)
    recs = run_cleaning_assistant(df, col_types, report)

    missing_map = {r["column"]: r for r in recs["missing_recommendations"]}

    assert missing_map["num_low"]["action"] == "impute_median"
    assert missing_map["num_mid"]["action"] == "impute_knn"
    assert missing_map["num_high"]["action"] == "manual"
    assert missing_map["cat_low"]["action"] == "impute_mode"
    assert missing_map["cat_high"]["action"] == "manual"
    assert missing_map["id_col"]["action"] == "manual"

    # Verify every recommendation has plain-English reasoning
    for r in recs["missing_recommendations"]:
        assert len(r["reasoning"]) > 10


def test_duplicate_and_outlier_rules():
    # Duplicates present
    base = pd.DataFrame({"id": range(20), "val": range(20)})
    df = pd.concat([base, base.iloc[:3]], ignore_index=True)
    col_types = {"id": "numeric", "val": "numeric"}

    report = run_data_detective(df, col_types)
    recs = run_cleaning_assistant(df, col_types, report)

    assert recs["duplicate_recommendation"] is not None
    assert recs["duplicate_recommendation"]["action"] == "drop_duplicates"

    # Outliers should always have action == 'review_manual'
    for o_rec in recs["outlier_recommendations"]:
        assert o_rec["action"] == "review_manual"
        assert "IQR" in o_rec["issue"]
