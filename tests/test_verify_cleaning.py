"""
tests/test_verify_cleaning.py
Tests for datalens.verify_cleaning matching PRD Section 5.5 specifications.
"""
import pytest
import numpy as np
import pandas as pd

from datalens.column_types import classify_all
from datalens.data_quality import run_data_detective
from datalens.cleaning_assistant import run_cleaning_assistant
from datalens.verify_cleaning import apply_recommendations, verify_cleaning


def test_original_dataframe_immutability():
    df_orig = pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0, 5.0] * 5,
        "cat": ["A", "B", None, "A", "A"] * 5,
    })
    df_snapshot = df_orig.copy()

    actions = [
        {"column": "num", "action": "impute_median"},
        {"column": "cat", "action": "impute_mode"},
    ]

    cleaned_df, log = apply_recommendations(df_orig, actions)

    # Assert original df was NOT mutated
    pd.testing.assert_frame_equal(df_orig, df_snapshot)
    assert df_orig["num"].isna().sum() > 0
    assert df_orig["cat"].isna().sum() > 0

    # Assert cleaned df has 0 missing
    assert cleaned_df["num"].isna().sum() == 0
    assert cleaned_df["cat"].isna().sum() == 0


def test_real_knn_imputation():
    # 2 correlated numeric columns with missing entries in one
    np.random.seed(42)
    x = np.arange(50, dtype=float)
    y = x * 2.0 + np.random.normal(0, 1, 50)
    y[10:15] = np.nan  # 5 missing entries (10%)

    df = pd.DataFrame({"x": x, "y": y})
    actions = [{"column": "y", "action": "impute_knn"}]

    cleaned_df, log = apply_recommendations(df, actions)

    assert cleaned_df["y"].isna().sum() == 0
    # Imputed values should be close to 2 * x
    for idx in range(10, 15):
        expected = x[idx] * 2.0
        assert abs(cleaned_df.loc[idx, "y"] - expected) < 15.0

    knn_log = next(l for l in log if l["target"] == "y")
    assert knn_log["action"] == "impute_knn"
    assert "KNN" in knn_log["note"]


def test_end_to_end_verification_coffee_shop(coffee_shop_fixture):
    col_types = classify_all(coffee_shop_fixture)
    before_report = run_data_detective(coffee_shop_fixture, col_types)
    recs = run_cleaning_assistant(coffee_shop_fixture, col_types, before_report)

    # Approve all safe auto recommendations
    approved = [r for r in recs["missing_recommendations"] if r["is_safe_auto"]]
    cleaned_df, log = apply_recommendations(coffee_shop_fixture, approved)

    after_types = classify_all(cleaned_df)
    after_report = run_data_detective(cleaned_df, after_types)

    diff = verify_cleaning(before_report, after_report, log)

    assert diff["score_after"] >= diff["score_before"]
    assert diff["missing_resolved"] >= 0
    assert len(diff["log"]) == len(approved)
