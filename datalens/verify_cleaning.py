"""
datalens/verify_cleaning.py
Applies user-approved data transformations to a copy of the dataset.
Includes authentic KNN imputation via scikit-learn, mode/median filling, duplicate dropping,
and produces a defensible before/after verification report.
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from datalens.config import KNN_NEIGHBORS
from datalens.column_types import classify_all
from datalens.data_quality import run_data_detective


def apply_recommendations(
    df: pd.DataFrame,
    approved_actions: List[Dict[str, Any]],
    manual_values: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Apply user-approved cleaning actions onto a COPY of the DataFrame.
    Never modifies original df.
    
    approved_actions: list of action dicts with keys:
      {'column': str, 'action': str, ...} or {'action': 'drop_duplicates', ...}
    manual_values: optional dict {column_name: value} for manual actions
    
    Returns: (cleaned_df, applied_log)
    """
    cleaned_df = df.copy()
    manual_values = manual_values or {}
    applied_log: List[Dict[str, Any]] = []

    # Check for drop_duplicates first
    drop_dup_action = next((a for a in approved_actions if a.get("action") == "drop_duplicates"), None)
    if drop_dup_action:
        initial_len = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        rows_removed = initial_len - len(cleaned_df)
        applied_log.append({
            "target": "dataset",
            "action": "drop_duplicates",
            "value_used": None,
            "rows_affected": rows_removed,
            "status": "applied",
            "note": f"Removed {rows_removed} duplicate rows.",
        })

    # Group actions by type
    knn_cols = []
    
    for item in approved_actions:
        action = item.get("action")
        col = item.get("column")
        if not col or col not in cleaned_df.columns:
            continue

        missing_mask = cleaned_df[col].isna()
        rows_affected = int(missing_mask.sum())

        if rows_affected == 0:
            continue

        if action == "impute_median":
            series_num = pd.to_numeric(cleaned_df[col], errors="coerce")
            val = float(series_num.median()) if series_num.notna().any() else 0.0
            cleaned_df[col] = cleaned_df[col].fillna(val)
            applied_log.append({
                "target": col,
                "action": "impute_median",
                "value_used": val,
                "rows_affected": rows_affected,
                "status": "applied",
                "note": f"Filled {rows_affected} missing values with median ({val}).",
            })

        elif action == "impute_mode":
            mode_vals = cleaned_df[col].dropna().mode()
            val = mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown"
            cleaned_df[col] = cleaned_df[col].fillna(val)
            applied_log.append({
                "target": col,
                "action": "impute_mode",
                "value_used": str(val),
                "rows_affected": rows_affected,
                "status": "applied",
                "note": f"Filled {rows_affected} missing values with mode ('{val}').",
            })

        elif action == "impute_knn":
            knn_cols.append(col)

        elif action in {"manual", "manual_input"}:
            if col in manual_values and manual_values[col] is not None and str(manual_values[col]).strip() != "":
                user_val = manual_values[col]
                cleaned_df[col] = cleaned_df[col].fillna(user_val)
                applied_log.append({
                    "target": col,
                    "action": "manual_input",
                    "value_used": user_val,
                    "rows_affected": rows_affected,
                    "status": "applied",
                    "note": f"Filled {rows_affected} missing values with user-specified value '{user_val}'.",
                })
            else:
                applied_log.append({
                    "target": col,
                    "action": "manual_input",
                    "value_used": None,
                    "rows_affected": 0,
                    "status": "skipped",
                    "note": f"Manual action skipped: no value provided for column '{col}'.",
                })

    # Apply KNN Imputation across numeric features
    if knn_cols:
        numeric_cols = [
            c for c in cleaned_df.columns
            if pd.api.types.is_numeric_dtype(cleaned_df[c]) and not c.endswith("_id")
        ]
        
        # Ensure target knn_cols are in the pool
        pool_cols = list(dict.fromkeys(numeric_cols + knn_cols))
        
        if len(pool_cols) >= 2:
            try:
                # Subset and scale
                sub_df = cleaned_df[pool_cols].copy()
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(sub_df)
                
                n_neighbors = min(KNN_NEIGHBORS, max(1, len(sub_df) - 1))
                imputer = KNNImputer(n_neighbors=n_neighbors)
                imputed_scaled = imputer.fit_transform(scaled_data)
                
                # Inverse transform
                imputed_data = scaler.inverse_transform(imputed_scaled)
                imputed_df = pd.DataFrame(imputed_data, columns=pool_cols, index=cleaned_df.index)
                
                for col in knn_cols:
                    rows_aff = int(cleaned_df[col].isna().sum())
                    cleaned_df[col] = imputed_df[col]
                    applied_log.append({
                        "target": col,
                        "action": "impute_knn",
                        "value_used": f"KNN ({n_neighbors} neighbors)",
                        "rows_affected": rows_aff,
                        "status": "applied",
                        "note": f"Imputed {rows_aff} missing values via KNN using scaled features.",
                    })
            except Exception as e:
                # Fallback to median with transparent logging
                for col in knn_cols:
                    rows_aff = int(cleaned_df[col].isna().sum())
                    med = float(cleaned_df[col].median())
                    cleaned_df[col] = cleaned_df[col].fillna(med)
                    applied_log.append({
                        "target": col,
                        "action": "impute_knn_fallback_median",
                        "value_used": med,
                        "rows_affected": rows_aff,
                        "status": "fallback",
                        "note": f"KNN failed ({str(e)}). Transparently fell back to median ({med}).",
                    })
        else:
            # Not enough numeric features for KNN
            for col in knn_cols:
                rows_aff = int(cleaned_df[col].isna().sum())
                med = float(cleaned_df[col].median())
                cleaned_df[col] = cleaned_df[col].fillna(med)
                applied_log.append({
                    "target": col,
                    "action": "impute_knn_fallback_median",
                    "value_used": med,
                    "rows_affected": rows_aff,
                    "status": "fallback",
                    "note": "Fewer than 2 numeric features available for KNN; safely used median.",
                })

    return cleaned_df, applied_log


def verify_cleaning(
    before_report: Dict[str, Any],
    after_report: Dict[str, Any],
    applied_log: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Produce a rigorous before/after comparison showing score delta and resolved defects.
    """
    applied_log = applied_log or []
    score_before = before_report.get("quality_score", 0.0)
    score_after = after_report.get("quality_score", 0.0)
    improvement = round(score_after - score_before, 1)

    missing_before = before_report.get("total_missing_cells", 0)
    missing_after = after_report.get("total_missing_cells", 0)
    missing_resolved = missing_before - missing_after

    dup_before = before_report.get("duplicate_rows", 0)
    dup_after = after_report.get("duplicate_rows", 0)
    dup_resolved = dup_before - dup_after

    return {
        "score_before": score_before,
        "score_after": score_after,
        "score_improvement": improvement,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "missing_resolved": missing_resolved,
        "duplicates_before": dup_before,
        "duplicates_after": dup_after,
        "duplicates_resolved": dup_resolved,
        "applied_actions_count": len([l for l in applied_log if l.get("status") in {"applied", "fallback"}]),
        "log": applied_log,
    }
