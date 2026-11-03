"""
tests/test_data_quality.py
Tests for datalens.data_quality (Data Detective) matching PRD Section 5.3 specifications.
"""
import pytest
import numpy as np
import pandas as pd

from datalens.column_types import classify_all
from datalens.data_quality import run_data_detective


def test_perfect_dataset_scores_100(clean_numeric_df):
    col_types = classify_all(clean_numeric_df)
    report = run_data_detective(clean_numeric_df, col_types)

    assert report["quality_score"] == 100.0
    assert report["duplicate_rows"] == 0
    assert report["total_missing_cells"] == 0


def test_missing_values_and_score_breakdown():
    # 100 rows, 10 missing in num1 (10%), unique row_id so 0 duplicate rows
    df = pd.DataFrame({
        "row_id": range(100),
        "num1": [np.nan if i < 10 else float(i) for i in range(100)],
        "cat1": ["A" if i % 2 == 0 else "B" for i in range(100)],
    })
    col_types = {"row_id": "id", "num1": "numeric", "cat1": "categorical"}
    report = run_data_detective(df, col_types)

    assert report["missing_by_column"]["num1"]["count"] == 10
    assert report["missing_by_column"]["num1"]["pct"] == 10.0
    assert report["duplicate_rows"] == 0

    # Avg missing = (10% + 0% + 0%) / 3 = 3.333%
    # Missing penalty = 3.333% * 1.5 = 5.0 pts
    assert report["score_breakdown"]["missing_penalty"] == 5.0
    assert report["quality_score"] == 95.0


def test_duplicate_rows_penalty():
    # 5 duplicate rows in 50 total (10% duplicates)
    base = pd.DataFrame({"x": range(45), "y": ["val"] * 45})
    dups = base.head(5)
    df = pd.concat([base, dups], ignore_index=True)
    col_types = classify_all(df)
    report = run_data_detective(df, col_types)

    assert report["duplicate_rows"] == 5
    assert report["duplicate_pct"] == 10.0
    # Duplicate penalty = 10% * 2.0 = 20.0 pts
    assert report["score_breakdown"]["duplicate_penalty"] == 20.0


def test_skewness_annotation():
    # Heavily right-skewed data (e.g. exponential distribution with high outliers)
    np.random.seed(42)
    normal_data = list(np.random.normal(10, 1, 95))
    extreme_premiums = [150.0, 200.0, 300.0, 400.0, 500.0]  # Skewed items
    df = pd.DataFrame({"price": normal_data + extreme_premiums})
    col_types = {"price": "numeric"}

    report = run_data_detective(df, col_types)
    outlier_info = report["outliers_by_column"]["price"]

    assert outlier_info["count"] > 0
    assert outlier_info["is_skewed"] is True
    assert "skewed" in outlier_info["note"].lower()
