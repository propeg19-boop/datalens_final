"""
datalens/pattern_hunter.py
Pattern Hunter: statistical insight discovery engine.
Finds correlations between numeric metrics, temporal trends across time series,
and category breakdowns across dimensions using column types.
Pure Python/pandas - returns clean structured dicts for plotting or narration.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from datalens.config import (
    CORRELATION_MODERATE_THRESHOLD,
    CORRELATION_STRONG_THRESHOLD,
    MAX_CATEGORIES_FOR_BREAKDOWN,
)


def find_correlations(df: pd.DataFrame, column_types: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Compute pairwise Pearson correlations across numeric columns.
    Returns pairs where |r| >= 0.5, sorted by correlation strength.
    Each pair is returned once.
    """
    numeric_cols = [
        col for col, ctype in column_types.items()
        if ctype == "numeric" and col in df.columns
    ]

    if len(numeric_cols) < 2:
        return []

    # Coerce to numeric safely
    num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr_matrix = num_df.corr()

    findings: List[Dict[str, Any]] = []
    seen_pairs = set()

    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i >= j:
                continue
            r = corr_matrix.loc[col1, col2]
            if pd.isna(r):
                continue

            r_val = float(r)
            abs_r = abs(r_val)

            if abs_r >= CORRELATION_MODERATE_THRESHOLD:
                strength = "strong" if abs_r >= CORRELATION_STRONG_THRESHOLD else "moderate"
                direction = "positive" if r_val > 0 else "negative"
                pair_key = tuple(sorted([col1, col2]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                findings.append({
                    "col1": col1,
                    "col2": col2,
                    "r": round(r_val, 3),
                    "abs_r": round(abs_r, 3),
                    "strength": strength,
                    "direction": direction,
                    "sample_count": int(num_df[[col1, col2]].dropna().shape[0]),
                })

    findings.sort(key=lambda x: x["abs_r"], reverse=True)
    return findings


def find_time_trends(df: pd.DataFrame, column_types: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Detect temporal trends in numeric columns over a true date/datetime column.
    Uses daily or weekly aggregation and fits a linear polynomial slope.
    Excludes time-of-day ('time') columns.
    """
    date_cols = [
        col for col, ctype in column_types.items()
        if ctype == "datetime" and col in df.columns
    ]
    numeric_cols = [
        col for col, ctype in column_types.items()
        if ctype == "numeric" and col in df.columns
    ]

    if not date_cols or not numeric_cols:
        return []

    date_col = date_cols[0]
    dt_series = pd.to_datetime(df[date_col], errors="coerce")
    valid_mask = dt_series.notna()

    if valid_mask.sum() < 10:
        return []

    working_df = df.loc[valid_mask].copy()
    working_df["_dt_parsed"] = dt_series.loc[valid_mask]

    min_date = working_df["_dt_parsed"].min()
    max_date = working_df["_dt_parsed"].max()
    span_days = (max_date - min_date).days

    freq = "D"
    period_label = "daily"
    if span_days > 180:
        freq = "W"
        period_label = "weekly"
    if span_days > 730:
        freq = "M"
        period_label = "monthly"

    findings: List[Dict[str, Any]] = []

    for num_col in numeric_cols:
        sub = working_df[["_dt_parsed", num_col]].dropna()
        if len(sub) < 5:
            continue

        try:
            resampled = (
                sub.set_index("_dt_parsed")[num_col]
                .apply(pd.to_numeric, errors="coerce")
                .resample(freq)
                .sum()
                .dropna()
            )

            if len(resampled) < 3:
                continue

            x = np.arange(len(resampled))
            y = resampled.values

            slope, intercept = np.polyfit(x, y, 1)
            mean_y = float(np.mean(y)) if np.mean(y) != 0 else 1.0
            pct_change_per_step = (slope / mean_y) * 100.0

            if abs(pct_change_per_step) < 0.5:
                direction = "flat"
            elif slope > 0:
                direction = "upward"
            else:
                direction = "downward"

            findings.append({
                "date_col": date_col,
                "metric_col": num_col,
                "period_type": period_label,
                "span_start": str(min_date.date()) if hasattr(min_date, "date") else str(min_date),
                "span_end": str(max_date.date()) if hasattr(max_date, "date") else str(max_date),
                "slope": round(float(slope), 3),
                "pct_change_per_step": round(float(pct_change_per_step), 2),
                "direction": direction,
                "total_periods": len(resampled),
                "mean_period_value": round(mean_y, 2),
            })
        except Exception:
            continue

    return findings


def find_category_breakdowns(df: pd.DataFrame, column_types: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Examine distributions of numeric metrics across categorical columns.
    Skips high-cardinality categories (>30 distinct values).
    """
    cat_cols = [
        col for col, ctype in column_types.items()
        if ctype == "categorical" and col in df.columns
    ]
    numeric_cols = [
        col for col, ctype in column_types.items()
        if ctype == "numeric" and col in df.columns
    ]

    findings: List[Dict[str, Any]] = []

    for cat_col in cat_cols:
        series_cat = df[cat_col].dropna()
        n_unique = series_cat.nunique()
        if n_unique < 2 or n_unique > MAX_CATEGORIES_FOR_BREAKDOWN:
            continue

        for num_col in numeric_cols:
            sub = df[[cat_col, num_col]].dropna()
            if len(sub) < 5:
                continue

            try:
                sub[num_col] = pd.to_numeric(sub[num_col], errors="coerce")
                grouped = sub.groupby(cat_col)[num_col].agg(["sum", "mean", "count"]).dropna()
                if len(grouped) < 2:
                    continue

                grouped = grouped.sort_values(by="sum", ascending=False)
                total_sum = float(grouped["sum"].sum())
                top_cat = str(grouped.index[0])
                top_val = float(grouped["sum"].iloc[0])
                top_pct = round((top_val / total_sum * 100.0), 1) if total_sum > 0 else 0.0

                bottom_cat = str(grouped.index[-1])
                bottom_val = float(grouped["sum"].iloc[-1])

                findings.append({
                    "category_col": cat_col,
                    "metric_col": num_col,
                    "num_categories": n_unique,
                    "total_sum": round(total_sum, 2),
                    "top_category": top_cat,
                    "top_category_sum": round(top_val, 2),
                    "top_category_share_pct": top_pct,
                    "bottom_category": bottom_cat,
                    "bottom_category_sum": round(bottom_val, 2),
                })
            except Exception:
                continue

    return findings


def run_pattern_hunter(df: pd.DataFrame, column_types: Dict[str, str]) -> Dict[str, Any]:
    """
    Run complete pattern discovery suite.
    Returns:
    {
        "correlations": list[dict],
        "trends": list[dict],
        "breakdowns": list[dict]
    }
    """
    return {
        "correlations": find_correlations(df, column_types),
        "trends": find_time_trends(df, column_types),
        "breakdowns": find_category_breakdowns(df, column_types),
    }
