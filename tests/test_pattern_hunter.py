"""
tests/test_pattern_hunter.py
Tests for datalens.pattern_hunter matching PRD Section 5.6 specifications.
"""
import pytest
import numpy as np
import pandas as pd

from datalens.column_types import classify_all
from datalens.pattern_hunter import (
    find_correlations,
    find_time_trends,
    find_category_breakdowns,
    run_pattern_hunter,
)


def test_correlations_detection(clean_numeric_df):
    col_types = classify_all(clean_numeric_df)
    corrs = find_correlations(clean_numeric_df, col_types)

    assert len(corrs) >= 1
    top_corr = corrs[0]
    assert set([top_corr["col1"], top_corr["col2"]]) == {"metric_x", "metric_y"}
    assert top_corr["r"] > 0.8
    assert top_corr["strength"] == "strong"
    assert top_corr["direction"] == "positive"


def test_no_correlations_when_uncorrelated():
    np.random.seed(42)
    df = pd.DataFrame({
        "a": np.random.normal(0, 1, 100),
        "b": np.random.normal(0, 1, 100),
    })
    col_types = {"a": "numeric", "b": "numeric"}
    corrs = find_correlations(df, col_types)
    # Random normal variables with N=100 won't have |r| >= 0.5
    assert len(corrs) == 0


def test_time_trends():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    sales = [float(100 + i * 5 + np.random.normal(0, 1)) for i in range(60)]  # Upward trend

    df = pd.DataFrame({"sale_date": dates, "revenue": sales})
    col_types = {"sale_date": "datetime", "revenue": "numeric"}

    trends = find_time_trends(df, col_types)
    assert len(trends) == 1
    t = trends[0]
    assert t["direction"] == "upward"
    assert t["slope"] > 0
    assert t["metric_col"] == "revenue"


def test_category_breakdowns():
    df = pd.DataFrame({
        "dept": ["Engineering", "Engineering", "Marketing", "Marketing", "Sales", "Sales"],
        "budget": [5000.0, 6000.0, 2000.0, 1500.0, 3000.0, 4000.0],
    })
    col_types = {"dept": "categorical", "budget": "numeric"}

    breakdowns = find_category_breakdowns(df, col_types)
    assert len(breakdowns) == 1
    b = breakdowns[0]
    assert b["top_category"] == "Engineering"
    assert b["bottom_category"] == "Marketing"
    assert b["total_sum"] == 21500.0


def test_empty_edge_cases():
    # Only text column
    df = pd.DataFrame({"comments": ["great", "bad", "okay"]})
    col_types = {"comments": "text"}

    results = run_pattern_hunter(df, col_types)
    assert results["correlations"] == []
    assert results["trends"] == []
    assert results["breakdowns"] == []
