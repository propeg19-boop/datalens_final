"""
datalens/data_quality.py
Data Detective: automated data quality assessment and scoring.
Evaluates completeness (missing), uniqueness (duplicates), and validity (outliers/skewness).
Generates an explainable 0-100 quality score with full breakdown.
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from datalens.config import (
    SCORE_WEIGHT_MISSING,
    SCORE_WEIGHT_DUPLICATE,
    SCORE_WEIGHT_OUTLIER,
    SKEWNESS_THRESHOLD,
    IQR_STANDARD_MULTIPLIER,
    IQR_SKEWED_MULTIPLIER,
)


def run_data_detective(df: pd.DataFrame, column_types: Dict[str, str]) -> Dict[str, Any]:
    """
    Run Data Detective health check on a DataFrame.
    Returns: comprehensive quality report dict.
    """
    n_rows = len(df)
    n_cols = len(df.columns)

    if n_rows == 0 or n_cols == 0:
        return {
            "quality_score": 0.0,
            "score_breakdown": {
                "initial": 100.0,
                "missing_penalty": 0.0,
                "duplicate_penalty": 0.0,
                "outlier_penalty": 0.0,
                "final_score": 0.0,
            },
            "missing_pct_avg": 0.0,
            "missing_by_column": {},
            "total_missing_cells": 0,
            "duplicate_rows": 0,
            "duplicate_pct": 0.0,
            "outliers_by_column": {},
            "total_outlier_count": 0,
            "total_rows": n_rows,
            "total_cols": n_cols,
        }

    # 1. Missing Values Check
    missing_by_column: Dict[str, Dict[str, Any]] = {}
    total_missing = 0
    missing_pcts: List[float] = []

    for col in df.columns:
        cnt = int(df[col].isna().sum())
        pct = (cnt / n_rows) * 100.0
        total_missing += cnt
        missing_pcts.append(pct)
        missing_by_column[col] = {
            "count": cnt,
            "pct": round(pct, 2),
            "column_type": column_types.get(col, "text"),
        }

    avg_missing_pct = float(np.mean(missing_pcts)) if missing_pcts else 0.0

    # 2. Duplicate Rows Check
    dup_rows = int(df.duplicated().sum())
    dup_pct = (dup_rows / n_rows) * 100.0

    # 3. Outlier Analysis (only on numeric columns)
    outliers_by_column: Dict[str, Dict[str, Any]] = {}
    total_outliers = 0
    total_numeric_values = 0

    for col, col_type in column_types.items():
        if col_type == "numeric" and col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            m = len(series)
            if m < 5:
                continue

            total_numeric_values += m
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            # Compute skewness
            skew = float(series.skew()) if m > 2 else 0.0
            is_skewed = abs(skew) >= SKEWNESS_THRESHOLD

            lower_bound = q1 - (IQR_STANDARD_MULTIPLIER * iqr)
            upper_bound = q3 + (IQR_STANDARD_MULTIPLIER * iqr)

            outlier_mask = (series < lower_bound) | (series > upper_bound)
            cnt = int(outlier_mask.sum())
            pct = (cnt / m) * 100.0
            total_outliers += cnt

            note = (
                f"Distribution is skewed (skewness: {skew:.2f}). Values outside IQR are likely "
                f"legitimate extreme values, not errors."
                if is_skewed
                else f"Standard symmetric IQR bounds [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."
            )

            outliers_by_column[col] = {
                "count": cnt,
                "pct": round(pct, 2),
                "q1": round(q1, 3),
                "q3": round(q3, 3),
                "iqr": round(iqr, 3),
                "lower_bound": round(lower_bound, 3),
                "upper_bound": round(upper_bound, 3),
                "skewness": round(skew, 2),
                "is_skewed": is_skewed,
                "note": note,
            }

    outlier_pct_overall = (
        (total_outliers / total_numeric_values * 100.0) if total_numeric_values > 0 else 0.0
    )

    # 4. Explainable Score Calculation
    missing_penalty = avg_missing_pct * SCORE_WEIGHT_MISSING
    duplicate_penalty = dup_pct * SCORE_WEIGHT_DUPLICATE
    outlier_penalty = outlier_pct_overall * SCORE_WEIGHT_OUTLIER

    raw_score = 100.0 - (missing_penalty + duplicate_penalty + outlier_penalty)
    quality_score = max(0.0, min(100.0, raw_score))

    return {
        "quality_score": round(quality_score, 1),
        "score_breakdown": {
            "initial": 100.0,
            "missing_penalty": round(missing_penalty, 2),
            "duplicate_penalty": round(duplicate_penalty, 2),
            "outlier_penalty": round(outlier_penalty, 2),
            "final_score": round(quality_score, 1),
        },
        "missing_pct_avg": round(avg_missing_pct, 2),
        "missing_by_column": missing_by_column,
        "total_missing_cells": total_missing,
        "duplicate_rows": dup_rows,
        "duplicate_pct": round(dup_pct, 2),
        "outliers_by_column": outliers_by_column,
        "total_outlier_count": total_outliers,
        "total_rows": n_rows,
        "total_cols": n_cols,
    }

