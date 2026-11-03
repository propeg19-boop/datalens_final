"""
datalens/cleaning_assistant.py
Cleaning Assistant: turns data quality findings into actionable, explainable recommendations.
Implements the PRD Section 5.4 decision matrix with user-in-the-loop control.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from datalens.config import (
    MISSING_THRESHOLD_LOW,
    MISSING_THRESHOLD_HIGH,
)


def _compute_suggested_value(series: pd.Series, action: str) -> Optional[Any]:
    """Calculate suggested replacement value for display in recommendation cards."""
    non_nulls = series.dropna()
    if len(non_nulls) == 0:
        return None

    if action == "impute_median":
        try:
            return float(pd.to_numeric(non_nulls).median())
        except Exception:
            return None
    elif action == "impute_mode":
        modes = non_nulls.mode()
        return modes.iloc[0] if len(modes) > 0 else None
    return None


def run_cleaning_assistant(
    df: pd.DataFrame,
    column_types: Dict[str, str],
    quality_report: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate cleaning recommendations based on the quality report.
    Returns:
    {
        "missing_recommendations": list[dict],
        "duplicate_recommendation": dict | None,
        "outlier_recommendations": list[dict]
    }
    """
    missing_recs: List[Dict[str, Any]] = []

    # 1. Missing Value Recommendations
    for col, m_info in quality_report.get("missing_by_column", {}).items():
        cnt = m_info["count"]
        pct = m_info["pct"]
        c_type = m_info["column_type"]

        if cnt == 0:
            continue

        series = df[col] if col in df.columns else pd.Series()

        if c_type in {"id", "datetime", "text", "time"}:
            action = "manual"
            suggested_val = None
            is_safe = False
            reasoning = (
                f"Column '{col}' is classified as {c_type}. "
                f"Identifiers, dates, and free-form text cannot be statistically inferred; "
                f"a wrong guess would compromise data integrity. Manual review required."
            )
        elif c_type == "numeric":
            if pct <= MISSING_THRESHOLD_LOW:
                action = "impute_median"
                suggested_val = _compute_suggested_value(series, action)
                is_safe = True
                reasoning = (
                    f"Only {cnt} values ({pct:.2f}%) are missing. Median imputation ({suggested_val}) "
                    f"is robust to outliers and preserves the numeric distribution."
                )
            elif pct <= MISSING_THRESHOLD_HIGH:
                action = "impute_knn"
                suggested_val = "Calculated via KNN"
                is_safe = True
                reasoning = (
                    f"{cnt} values ({pct:.2f}%) are missing. KNN imputation estimates missing entries "
                    f"using similar rows across all numeric features, preserving multivariate correlations."
                )
            else:
                action = "manual"
                suggested_val = None
                is_safe = False
                reasoning = (
                    f"High missing rate of {pct:.2f}% ({cnt} rows). Imputing over 15% may artificially distort "
                    f"distributions. Review column relevance or consider dropping it."
                )
        elif c_type == "categorical":
            if pct <= MISSING_THRESHOLD_LOW:
                action = "impute_mode"
                suggested_val = _compute_suggested_value(series, action)
                is_safe = True
                reasoning = (
                    f"Only {cnt} entries ({pct:.2f}%) are missing. Filling with the most frequent category "
                    f"('{suggested_val}') preserves category balance safely."
                )
            else:
                action = "manual"
                suggested_val = "Unknown"
                is_safe = False
                reasoning = (
                    f"Missing share is {pct:.2f}% ({cnt} rows). Consider replacing missing entries with "
                    f"'Unknown' or supplying a domain-specific label."
                )
        else:
            action = "manual"
            suggested_val = None
            is_safe = False
            reasoning = f"Column '{col}' requires manual inspection."

        missing_recs.append({
            "column": col,
            "column_type": c_type,
            "issue": f"{cnt} missing values ({pct:.2f}%)",
            "count": cnt,
            "pct": pct,
            "action": action,
            "suggested_value": suggested_val,
            "reasoning": reasoning,
            "is_safe_auto": is_safe,
        })

    # 2. Duplicate Rows Recommendation
    dup_rec: Optional[Dict[str, Any]] = None
    dup_rows = quality_report.get("duplicate_rows", 0)
    dup_pct = quality_report.get("duplicate_pct", 0.0)

    if dup_rows > 0:
        dup_rec = {
            "issue": f"{dup_rows} exact duplicate rows ({dup_pct:.2f}%)",
            "count": dup_rows,
            "pct": dup_pct,
            "action": "drop_duplicates",
            "reasoning": (
                f"Detected {dup_rows} exact duplicate rows. Removing identical rows prevents "
                f"unintended sample weighting and statistical bias."
            ),
            "is_safe_auto": True,
        }

    # 3. Outlier Recommendations (NEVER auto-removed; review_manual only)
    outlier_recs: List[Dict[str, Any]] = []
    for col, o_info in quality_report.get("outliers_by_column", {}).items():
        cnt = o_info["count"]
        pct = o_info["pct"]
        if cnt == 0:
            continue

        is_skewed = o_info.get("is_skewed", False)
        skewness = o_info.get("skewness", 0.0)
        lower = o_info.get("lower_bound", 0.0)
        upper = o_info.get("upper_bound", 0.0)

        if is_skewed:
            reasoning = (
                f"Column has a skewed distribution (skewness: {skewness:.2f}). {cnt} values ({pct:.2f}%) "
                f"lie outside standard IQR bounds [{lower}, {upper}], but likely represent legitimate "
                f"high-value transactions or premium tiers. Do not delete without domain confirmation."
            )
        else:
            reasoning = (
                f"{cnt} values ({pct:.2f}%) lie outside normal IQR bounds [{lower}, {upper}]. "
                f"Examine whether these are data entry anomalies or genuine extremes before deciding to filter."
            )

        outlier_recs.append({
            "column": col,
            "column_type": "numeric",
            "issue": f"{cnt} IQR outliers ({pct:.2f}%)",
            "count": cnt,
            "pct": pct,
            "action": "review_manual",
            "lower_bound": lower,
            "upper_bound": upper,
            "is_skewed": is_skewed,
            "skewness": skewness,
            "reasoning": reasoning,
            "is_safe_auto": False,
        })

    return {
        "missing_recommendations": missing_recs,
        "duplicate_recommendation": dup_rec,
        "outlier_recommendations": outlier_recs,
    }

