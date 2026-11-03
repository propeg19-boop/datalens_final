"""
tests/test_ai_narrator.py
Tests for datalens.ai_narrator deterministic caption generation.
"""
import pytest
from datalens.ai_narrator import generate_caption


def test_correlation_caption():
    finding = {
        "col1": "unit_price",
        "col2": "total_amount",
        "r": 0.85,
        "direction": "positive",
        "strength": "strong",
    }
    caption = generate_caption(finding, "correlation", use_ai=False)
    assert "strong positive correlation" in caption
    assert "unit_price" in caption
    assert "causation" in caption.lower()


def test_trend_caption():
    finding = {
        "metric_col": "revenue",
        "period_type": "daily",
        "direction": "upward",
        "span_start": "2024-01-01",
        "span_end": "2024-06-30",
        "pct_change_per_step": 3.4,
    }
    caption = generate_caption(finding, "trend", use_ai=False)
    assert "upward trend" in caption
    assert "3.4%" in caption


def test_breakdown_caption():
    finding = {
        "category_col": "store_location",
        "metric_col": "sales",
        "top_category": "Lower Manhattan",
        "top_category_sum": 52000.0,
        "top_category_share_pct": 45.2,
        "bottom_category": "Astoria",
        "bottom_category_sum": 12000.0,
    }
    caption = generate_caption(finding, "breakdown", use_ai=False)
    assert "Lower Manhattan" in caption
    assert "45.2%" in caption
