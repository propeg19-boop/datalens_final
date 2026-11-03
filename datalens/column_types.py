"""
datalens/column_types.py
Data-driven column classification engine for DataLens.
Classifies columns into: 'id', 'datetime', 'numeric', 'categorical', 'text'
(and 'time' subtype where appropriate).
"""
import re
from typing import Dict, Any, Literal
import numpy as np
import pandas as pd

from datalens.config import (
    ID_UNIQUENESS_RATIO,
    ID_MIN_ROWS,
    NUMERIC_CATEGORICAL_MAX_UNIQUE,
    DATE_SAMPLE_SIZE,
    DATE_PARSE_SUCCESS_RATE,
    CATEGORICAL_RATIO,
    CATEGORICAL_MIN_THRESHOLD,
)

ColumnType = Literal["id", "datetime", "numeric", "categorical", "text", "time"]

TIME_PATTERN = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?(\s?[APap][Mm])?$")


def _is_time_string_series(non_nulls: pd.Series) -> bool:
    """Check if strings represent time-of-day without dates (e.g. 10:15:00)."""
    if len(non_nulls) == 0:
        return False
    sample = non_nulls.head(20).astype(str).str.strip()
    matches = sample.apply(lambda s: bool(TIME_PATTERN.match(s)))
    return matches.mean() >= 0.85


def _is_boolean_like(series: pd.Series) -> bool:
    """Detect if column represents boolean flags (True/False, Yes/No, 0/1)."""
    non_nulls = series.dropna()
    if len(non_nulls) == 0:
        return False
    unique_vals = set(non_nulls.unique())
    if len(unique_vals) > 2:
        return False

    str_vals = {str(v).strip().lower() for v in unique_vals}
    boolean_sets = [
        {"true", "false"},
        {"t", "f"},
        {"yes", "no"},
        {"y", "n"},
        {"1", "0"},
        {1, 0},
        {True, False},
    ]
    return any(str_vals.issubset(b) or unique_vals.issubset(b) for b in boolean_sets)


def _parses_as_dates(sample: pd.Series) -> bool:
    """Check if a sample of values reliably parses as valid datetimes."""
    if len(sample) == 0:
        return False
    # Avoid parsing pure numbers or floats as dates
    if pd.api.types.is_numeric_dtype(sample):
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        return bool(parsed.notna().mean() >= DATE_PARSE_SUCCESS_RATE)
    except Exception:
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            return bool(parsed.notna().mean() >= DATE_PARSE_SUCCESS_RATE)
        except Exception:
            return False


def classify_column(series: pd.Series, col_name: str = "") -> str:
    """
    Classify a single pandas Series into:
    'id', 'datetime', 'numeric', 'categorical', 'text', or 'time'
    
    Decision order (priority list, first match wins):
    1. Already a datetime dtype -> datetime
    2. Check boolean -> categorical
    3. Time-of-day check -> time
    4. Object/string date sniffing (random sample, >= 90% success) -> datetime
    5. Repeating foreign key / code (col_name has 'id' and nunique is not near n) -> categorical
    6. Identifiers (nunique >= 0.95 * n and n > 20):
       - Strings / tokens (e.g. TX-10001, UUID) -> id
       - Integer IDs (col_name has 'id', 'num', 'key' or sequential) -> id
    7. Numeric dtype:
       - nunique <= 5 -> categorical (small repeating label)
       - otherwise -> numeric
    8. Low cardinality string / dimension (nunique <= max(threshold, 5% of n)) -> categorical
    9. High cardinality text -> text
    """
    n = len(series)
    non_nulls = series.dropna()
    n_non_null = len(non_nulls)
    col_name_lower = str(col_name).strip().lower()

    if n_non_null == 0:
        return "text"

    # 1. Already a datetime dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # 2. Boolean check
    if pd.api.types.is_bool_dtype(series) or _is_boolean_like(non_nulls):
        return "categorical"

    # 3. Check pure time-of-day strings (e.g. 10:15:00)
    if _is_time_string_series(non_nulls):
        return "time"

    # 4. Date sniffing on object/string types
    sample_size = min(DATE_SAMPLE_SIZE, n_non_null)
    sample = non_nulls.sample(n=sample_size, random_state=42) if sample_size >= 5 else non_nulls
    if _parses_as_dates(sample):
        return "datetime"

    nunique = non_nulls.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(series)
    has_id_keyword = bool(re.search(r"(?:^|_)(?:id|key|code|num|no|uuid)(?:$|_)", col_name_lower))

    # 5. Identifier check
    is_high_uniqueness = (n > ID_MIN_ROWS) and (nunique >= (ID_UNIQUENESS_RATIO * n_non_null))

    if is_high_uniqueness:
        if not is_numeric:
            # String tokens like 'TX-10001', 'ORD-291'
            return "id"
        else:
            # Numeric column with unique values
            # If it has 'id', 'key', 'code', 'no' in name, or integer sequence -> id
            if has_id_keyword:
                return "id"
            # If continuous float or measurements, it is numeric!

    # 6. Numeric column handling
    if is_numeric:
        if has_id_keyword or (nunique <= NUMERIC_CATEGORICAL_MAX_UNIQUE and n > 20):
            return "categorical"
        return "numeric"

    # 7. Low / Moderate cardinality check -> categorical
    max_cat_threshold = max(CATEGORICAL_MIN_THRESHOLD, int(CATEGORICAL_RATIO * n_non_null))
    if nunique <= max_cat_threshold:
        return "categorical"

    # 8. Unstructured free text
    return "text"


def classify_all(df: pd.DataFrame) -> Dict[str, str]:
    """
    Classify all columns in a DataFrame.
    Returns: dict mapping column_name -> type
    """
    return {col: classify_column(df[col], col) for col in df.columns}
