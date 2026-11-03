"""
datalens/model_trainer.py
Baseline Machine Learning trainer for DataLens.
Supports automated regression and classification with explicit Dummy Baselines.
Follows PRD Section 5.9 specifications:
- Data leakage detection
- Preprocessing (imputation, encoding, scaling)
- 80/20 Train/Test split
- Rigorous metric comparison against naive baselines
- Feature importance extraction and honest plain-English verdicts
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

MAX_TRAIN_SAMPLES = 10000


def detect_target_task(target_type: str, series: pd.Series) -> Tuple[Optional[str], Optional[str]]:
    """
    Validate target column and infer learning task.
    Returns: (task, error_message) where task is 'regression' or 'classification'.
    """
    if target_type == "numeric":
        return "regression", None
    elif target_type == "categorical":
        n_unique = series.dropna().nunique()
        if n_unique < 2:
            return None, "Target column must have at least 2 distinct classes."
        if n_unique > 20:
            return None, f"Target column has {n_unique} distinct classes. Classification is capped at 20 classes."
        return "classification", None
    else:
        return None, f"Target column type '{target_type}' is not suitable for prediction. Pick a numeric or categorical column."


def train_baseline_model(
    df: pd.DataFrame,
    column_types: Dict[str, str],
    target_col: str,
) -> Dict[str, Any]:
    """
    Train baseline ML models on the dataset and compare against naive dummy baselines.
    Returns: comprehensive results dict containing metrics, feature importances, and verdict.
    """
    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found in dataset."}

    target_type = column_types.get(target_col, "numeric")
    task, err = detect_target_task(target_type, df[target_col])
    if err:
        return {"error": err}

    # Drop target null rows
    valid_df = df.dropna(subset=[target_col]).copy()
    if len(valid_df) < 20:
        return {"error": f"Insufficient complete rows for training ({len(valid_df)} rows found, minimum 20 required)."}

    # Subsample if dataset is massive for responsive training
    if len(valid_df) > MAX_TRAIN_SAMPLES:
        valid_df = valid_df.sample(n=MAX_TRAIN_SAMPLES, random_state=42)

    # Separate target
    y_raw = valid_df[target_col]

    # Select features: drop IDs, dates, time, and target itself
    candidate_features = [
        col for col, ctype in column_types.items()
        if col in valid_df.columns
        and col != target_col
        and ctype in {"numeric", "categorical"}
    ]

    if not candidate_features:
        return {"error": "No valid predictive features found after excluding IDs, dates, and target."}

    X_raw = valid_df[candidate_features].copy()

    # Check for potential data leakage
    leakage_warnings = []
    for col in candidate_features:
        if column_types.get(col) == "numeric" and task == "regression":
            try:
                corr = abs(float(valid_df[[col, target_col]].apply(pd.to_numeric).corr().iloc[0, 1]))
                if corr >= 0.98:
                    leakage_warnings.append(
                        f"Feature '{col}' has near-perfect correlation (r={corr:.3f}) with '{target_col}'. "
                        f"It may be a direct duplicate or derivative of the target (potential data leakage)."
                    )
            except Exception:
                pass

    # Preprocess features: One-hot encode categoricals, impute missing values
    num_cols = [c for c in candidate_features if column_types.get(c) == "numeric"]
    cat_cols = [c for c in candidate_features if column_types.get(c) == "categorical"]

    # Impute numeric with median
    for c in num_cols:
        X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")
        med = X_raw[c].median()
        X_raw[c] = X_raw[c].fillna(med if pd.notna(med) else 0.0)

    # Impute categorical with mode
    for c in cat_cols:
        mode_val = X_raw[c].mode()
        X_raw[c] = X_raw[c].fillna(mode_val.iloc[0] if len(mode_val) > 0 else "Unknown").astype(str)

    # Dummy encode categoricals
    if cat_cols:
        X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
    else:
        X_encoded = X_raw.copy()

    feature_names = list(X_encoded.columns)

    # Train / Test split (80/20, fixed seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_raw, test_size=0.20, random_state=42
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model training and metric computation
    if task == "regression":
        y_train = pd.to_numeric(y_train, errors="coerce").fillna(0.0)
        y_test = pd.to_numeric(y_test, errors="coerce").fillna(0.0)

        # 1. Dummy Baseline
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_test)

        dummy_r2 = float(r2_score(y_test, y_pred_dummy))
        dummy_mae = float(mean_absolute_error(y_test, y_pred_dummy))
        dummy_rmse = float(root_mean_squared_error(y_test, y_pred_dummy))

        # 2. Linear Regression
        linear = LinearRegression()
        linear.fit(X_train_scaled, y_train)
        y_pred_linear = linear.predict(X_test_scaled)

        linear_r2 = float(r2_score(y_test, y_pred_linear))
        linear_mae = float(mean_absolute_error(y_test, y_pred_linear))
        linear_rmse = float(root_mean_squared_error(y_test, y_pred_linear))

        # 3. Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        rf_r2 = float(r2_score(y_test, y_pred_rf))
        rf_mae = float(mean_absolute_error(y_test, y_pred_rf))
        rf_rmse = float(root_mean_squared_error(y_test, y_pred_rf))

        # Select best model
        best_r2 = max(linear_r2, rf_r2)
        beats_baseline = (rf_mae < dummy_mae) and (best_r2 > 0.10)

        if best_r2 > 0.60:
            verdict_badge = "Strong Model"
            verdict_text = (
                f"The model explains {best_r2 * 100:.1f}% of the variance in '{target_col}', "
                f"reducing MAE by {((dummy_mae - rf_mae) / dummy_mae) * 100:.1f}% over the baseline."
            )
        elif beats_baseline:
            verdict_badge = "Moderate Predictive Power"
            verdict_text = (
                f"The model outperforms naive guessing (MAE: {rf_mae:.2f} vs Dummy: {dummy_mae:.2f}), "
                f"capturing meaningful directional signals."
            )
        else:
            verdict_badge = "Weak Signal / Insufficient Fit"
            verdict_text = (
                f"The model performs close to the naive mean baseline (R²: {best_r2:.2f}). "
                f"The current features may not contain sufficient signal to predict '{target_col}' accurately."
            )

        metrics_table = {
            "models": ["Dummy Baseline (Mean)", "Linear Regression", "Random Forest Regressor"],
            "R2": [round(dummy_r2, 3), round(linear_r2, 3), round(rf_r2, 3)],
            "MAE": [round(dummy_mae, 2), round(linear_mae, 2), round(rf_mae, 2)],
            "RMSE": [round(dummy_rmse, 2), round(linear_rmse, 2), round(rf_rmse, 2)],
        }

        # Feature importances
        raw_importances = rf.feature_importances_

    else:
        # Classification
        y_train = y_train.astype(str)
        y_test = y_test.astype(str)

        # 1. Dummy Baseline
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_test)

        dummy_acc = float(accuracy_score(y_test, y_pred_dummy))
        dummy_f1 = float(f1_score(y_test, y_pred_dummy, average="weighted", zero_division=0))

        # 2. Logistic Regression
        clf_lr = LogisticRegression(max_iter=500, random_state=42)
        clf_lr.fit(X_train_scaled, y_train)
        y_pred_lr = clf_lr.predict(X_test_scaled)

        lr_acc = float(accuracy_score(y_test, y_pred_lr))
        lr_f1 = float(f1_score(y_test, y_pred_lr, average="weighted", zero_division=0))

        # 3. Random Forest Classifier
        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        rf_clf.fit(X_train, y_train)
        y_pred_rf = rf_clf.predict(X_test)

        rf_acc = float(accuracy_score(y_test, y_pred_rf))
        rf_f1 = float(f1_score(y_test, y_pred_rf, average="weighted", zero_division=0))

        best_acc = max(lr_acc, rf_acc)
        beats_baseline = best_acc > (dummy_acc + 0.05)

        if beats_baseline:
            verdict_badge = "Effective Classifier"
            verdict_text = (
                f"The model achieves {best_acc * 100:.1f}% accuracy vs {dummy_acc * 100:.1f}% for the "
                f"majority-class baseline, demonstrating genuine predictive signal."
            )
        else:
            verdict_badge = "Close to Baseline"
            verdict_text = (
                f"The model's accuracy ({best_acc * 100:.1f}%) is close to guessing the majority class "
                f"({dummy_acc * 100:.1f}%). Additional features or more balanced data are recommended."
            )

        metrics_table = {
            "models": ["Dummy Baseline (Majority)", "Logistic Regression", "Random Forest Classifier"],
            "Accuracy": [round(dummy_acc, 3), round(lr_acc, 3), round(rf_acc, 3)],
            "F1_Score": [round(dummy_f1, 3), round(lr_f1, 3), round(rf_f1, 3)],
        }

        raw_importances = rf_clf.feature_importances_

    # Format feature importances
    feat_df = pd.DataFrame({
        "feature": feature_names,
        "importance": raw_importances,
    }).sort_values(by="importance", ascending=False).head(10)

    importances_list = [
        {"feature": row["feature"], "importance": round(float(row["importance"]), 4)}
        for _, row in feat_df.iterrows()
    ]

    return {
        "task": task,
        "target_col": target_col,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features_used": len(feature_names),
        "metrics_table": metrics_table,
        "feature_importances": importances_list,
        "verdict": {
            "badge": verdict_badge,
            "text": verdict_text,
            "beats_baseline": beats_baseline,
        },
        "leakage_warnings": leakage_warnings,
    }
