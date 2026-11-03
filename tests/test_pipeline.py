"""
tests/test_pipeline.py
End-to-end integration tests for DataLensPipeline.
"""
from pathlib import Path
import pytest
import pandas as pd

from datalens.pipeline import DataLensPipeline


def test_pipeline_end_to_end_coffee_shop():
    sample_file = Path(__file__).resolve().parent.parent / "data" / "sample" / "coffeeshopsales_sample.csv"
    assert sample_file.exists(), "Sample dataset missing"

    pipeline = DataLensPipeline()
    success, err = pipeline.load_dataset(sample_file)
    assert success is True
    assert err is None
    assert len(pipeline.raw_df) > 0

    # Verify quality audit
    audit = pipeline.run_quality_audit()
    assert "quality_score" in audit
    assert audit["quality_score"] > 90.0

    # Verify cleaning recommendations
    recs = pipeline.recommendations
    assert len(recs["missing_recommendations"]) > 0

    # Apply safe cleaning
    safe_actions = [r for r in recs["missing_recommendations"] if r["is_safe_auto"]]
    pipeline.apply_cleaning_actions(safe_actions)
    assert pipeline.cleaned_quality_report["total_missing_cells"] < audit["total_missing_cells"]

    # Discover patterns
    patterns = pipeline.discover_patterns(use_ai=False)
    assert "correlations" in patterns
    assert "trends" in patterns
    assert "breakdowns" in patterns

    # Train model on unit_price
    model_res = pipeline.train_model("unit_price")
    assert "error" not in model_res
    assert model_res["task"] == "regression"

    # Export report
    md_rep = pipeline.get_summary_report("markdown")
    html_rep = pipeline.get_summary_report("html")
    assert "# DataLens" in md_rep
    assert "<!DOCTYPE html>" in html_rep
