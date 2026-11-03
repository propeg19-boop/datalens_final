"""
datalens/pipeline.py
Central orchestration pipeline for DataLens.
Coordinates ingestion, type detection, quality auditing, cleaning recommendations,
transformation verification, pattern mining, and baseline ML training.
Pure Python/pandas - zero Streamlit imports, perfect for programmatic testing and UI binding.
"""
from typing import Dict, Any, List, Optional, Union, BinaryIO, Tuple
from pathlib import Path
import io
import pandas as pd

from datalens.data_loader import load_data
from datalens.column_types import classify_all
from datalens.data_quality import run_data_detective
from datalens.cleaning_assistant import run_cleaning_assistant
from datalens.verify_cleaning import apply_recommendations, verify_cleaning
from datalens.pattern_hunter import run_pattern_hunter
from datalens.ai_narrator import generate_caption
from datalens.model_trainer import train_baseline_model
from datalens.report import generate_markdown_report, generate_html_report


class DataLensPipeline:
    """
    Session-friendly orchestrator managing the state of an uploaded dataset through its journey.
    """

    def __init__(self):
        self.filename: str = ""
        self.raw_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.column_types: Dict[str, str] = {}
        self.quality_report: Dict[str, Any] = {}
        self.cleaned_quality_report: Dict[str, Any] = {}
        self.recommendations: Dict[str, Any] = {}
        self.verification: Dict[str, Any] = {}
        self.patterns: Dict[str, Any] = {}
        self.model_results: Dict[str, Any] = {}
        self.applied_log: List[Dict[str, Any]] = []

    def load_dataset(
        self,
        file_path_or_buffer: Union[str, Path, io.BytesIO, BinaryIO],
        filename: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Load and initialize dataset."""
        df, err = load_data(file_path_or_buffer, filename=filename)
        if err:
            return False, err

        self.filename = filename or (Path(file_path_or_buffer).name if isinstance(file_path_or_buffer, (str, Path)) else "dataset.csv")
        self.raw_df = df
        self.cleaned_df = df.copy()
        self.column_types = classify_all(df)
        self.run_quality_audit()
        return True, None

    def override_column_type(self, column: str, new_type: str) -> None:
        """Allow user to manually correct a detected column classification."""
        if column in self.column_types:
            self.column_types[column] = new_type
            # Refresh audit with user-corrected types
            self.run_quality_audit()

    def run_quality_audit(self) -> Dict[str, Any]:
        """Execute Data Detective health audit on the current state."""
        active_df = self.cleaned_df if self.cleaned_df is not None else self.raw_df
        if active_df is None:
            return {}

        self.quality_report = run_data_detective(active_df, self.column_types)
        self.recommendations = run_cleaning_assistant(active_df, self.column_types, self.quality_report)
        return self.quality_report

    def apply_cleaning_actions(
        self,
        approved_actions: List[Dict[str, Any]],
        manual_values: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply approved cleaning actions and record verification."""
        if self.raw_df is None:
            return {}

        before_report = self.quality_report.copy()
        cleaned_df, log = apply_recommendations(self.raw_df, approved_actions, manual_values)
        self.cleaned_df = cleaned_df
        self.applied_log = log

        # Re-audit cleaned dataset
        self.column_types = classify_all(cleaned_df)
        after_report = run_data_detective(cleaned_df, self.column_types)
        self.cleaned_quality_report = after_report

        self.verification = verify_cleaning(before_report, after_report, log)
        return self.verification

    def reset_to_raw(self) -> None:
        """Reset dataset to its initial uploaded state."""
        if self.raw_df is not None:
            self.cleaned_df = self.raw_df.copy()
            self.applied_log = []
            self.column_types = classify_all(self.raw_df)
            self.run_quality_audit()
            self.verification = {}

    def discover_patterns(self, use_ai: bool = False) -> Dict[str, Any]:
        """Discover statistical correlations, trends, and breakdowns with captions."""
        active_df = self.cleaned_df if self.cleaned_df is not None else self.raw_df
        if active_df is None:
            return {}

        patterns = run_pattern_hunter(active_df, self.column_types)

        # Generate plain-English captions for findings
        for c in patterns.get("correlations", []):
            c["caption"] = generate_caption(c, "correlation", use_ai=use_ai)

        for t in patterns.get("trends", []):
            t["caption"] = generate_caption(t, "trend", use_ai=use_ai)

        for b in patterns.get("breakdowns", []):
            b["caption"] = generate_caption(b, "breakdown", use_ai=use_ai)

        self.patterns = patterns
        return patterns

    def train_model(self, target_col: str) -> Dict[str, Any]:
        """Train baseline ML models on target column."""
        active_df = self.cleaned_df if self.cleaned_df is not None else self.raw_df
        if active_df is None:
            return {"error": "No dataset loaded."}

        res = train_baseline_model(active_df, self.column_types, target_col)
        self.model_results = res
        return res

    def get_summary_report(self, format: str = "markdown") -> str:
        """Generate comprehensive export summary."""
        active_df = self.cleaned_df if self.cleaned_df is not None else self.raw_df
        summary_payload = {
            "dataset_name": self.filename,
            "rows": len(active_df) if active_df is not None else 0,
            "cols": len(active_df.columns) if active_df is not None else 0,
            "score_before": self.quality_report.get("quality_score", 0.0),
            "score_after": self.cleaned_quality_report.get("quality_score", self.quality_report.get("quality_score", 0.0)),
            "column_types": self.column_types,
            "applied_log": self.applied_log,
            "findings": self.patterns,
            "model_results": self.model_results,
        }

        if format == "html":
            return generate_html_report(summary_payload)
        return generate_markdown_report(summary_payload)


