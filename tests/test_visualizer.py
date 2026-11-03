"""
tests/test_visualizer.py
Tests for datalens.visualizer Plotly chart generation.
"""
import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datalens.column_types import classify_all
from datalens.data_quality import run_data_detective
from datalens.visualizer import (
    plot_missing_overview,
    plot_distribution,
    plot_correlation,
    plot_time_trend,
    plot_category_breakdown,
)


def test_plot_missing_overview():
    quality_report = {
        "missing_by_column": {
            "col_a": {"count": 10, "pct": 5.0},
            "col_b": {"count": 0, "pct": 0.0},
        }
    }
    fig = plot_missing_overview(quality_report)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_plot_distribution():
    df = pd.DataFrame({"num": range(50), "cat": ["X", "Y"] * 25})
    fig_num = plot_distribution(df, "num", "numeric")
    fig_cat = plot_distribution(df, "cat", "categorical")
    assert isinstance(fig_num, go.Figure)
    assert isinstance(fig_cat, go.Figure)


def test_plot_correlation_with_sampling():
    # 6000 rows (exceeds 5000 max scatter sample size)
    x = np.linspace(0, 100, 6000)
    y = x * 2 + np.random.normal(0, 1, 6000)
    df = pd.DataFrame({"x": x, "y": y})

    fig = plot_correlation(df, "x", "y", sample_size=5000)
    assert isinstance(fig, go.Figure)
    assert "Sampled" in fig.layout.title.text


def test_plot_time_trend_and_category_breakdown():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "amount": range(30),
        "region": ["East", "West", "North"] * 10,
    })
    fig_trend = plot_time_trend(df, "date", "amount")
    fig_cat = plot_category_breakdown(df, "region", "amount")
    assert isinstance(fig_trend, go.Figure)
    assert isinstance(fig_cat, go.Figure)
