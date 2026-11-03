"""
tests/conftest.py
Shared test fixtures for DataLens test suite.
Provides small synthetic DataFrames and temporary file utilities.
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Add repo root to python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def clean_numeric_df():
    """Tiny clean dataset with correlated numeric columns."""
    np.random.seed(42)
    x = np.linspace(10, 100, 30)
    y = 2.5 * x + np.random.normal(0, 5, 30)
    return pd.DataFrame({
        "id_col": [f"ID_{i}" for i in range(30)],
        "metric_x": x,
        "metric_y": y,
        "category": ["A", "B", "C"] * 10,
    })


@pytest.fixture
def messy_df():
    """Dataset with missing values, exact duplicates, and extreme outliers."""
    return pd.DataFrame({
        "customer_id": [f"C_{i}" for i in range(25)] + ["C_0"],  # 1 duplicate row
        "age": [25, np.nan, 30, 45, 120] + [35] * 21,           # 1 missing, 1 outlier (120)
        "city": ["New York", "Chicago", None, "New York", "Boston"] + ["New York"] * 21,  # 1 missing
        "score": [85.0, 90.0, 78.0, np.nan, 95.0] + [88.0] * 21,
    })


@pytest.fixture
def coffee_shop_fixture():
    """Sample coffee shop sales DataFrame."""
    from data.sample.generate_reference_data import generate_coffee_shop_sample
    return generate_coffee_shop_sample(n_rows=150)
