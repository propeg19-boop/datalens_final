"""
tests/test_column_types.py
Tests for datalens.column_types matching PRD Section 5.2 acceptance criteria and regression guards.
"""
import pytest
import pandas as pd
import numpy as np

from datalens.column_types import classify_column, classify_all


def test_coffee_shop_acceptance_table(coffee_shop_fixture):
    """
    PRD Section 5.2 Acceptance Test:
    Verify exact classification results for coffee shop schema.
    """
    types = classify_all(coffee_shop_fixture)

    assert types["transaction_id"] == "id", "transaction_id must be classified as 'id'"
    assert types["transaction_date"] == "datetime", "transaction_date must be 'datetime'"
    assert types["transaction_time"] in {"datetime", "time"}, "transaction_time must be 'datetime' or 'time'"
    assert types["transaction_qty"] == "numeric", "transaction_qty must remain 'numeric'"
    assert types["store_id"] == "categorical", "store_id must be 'categorical'"
    assert types["store_location"] == "categorical", "store_location must be 'categorical'"
    assert types["product_id"] == "categorical", "product_id must be 'categorical', never numeric"
    assert types["unit_price"] == "numeric", "unit_price must be 'numeric'"
    assert types["product_category"] == "categorical", "product_category must be 'categorical'"
    assert types["product_type"] == "categorical", "product_type must be 'categorical'"
    assert types["product_detail"] == "categorical", "product_detail must be 'categorical'"


def test_boolean_classification():
    s_bool = pd.Series([True, False, True, False] * 10)
    assert classify_column(s_bool, "is_active") == "categorical"

    s_str_bool = pd.Series(["Yes", "No", "Yes", "No"] * 10)
    assert classify_column(s_str_bool, "subscribed") == "categorical"

    s_zero_one = pd.Series([0, 1, 1, 0, 0, 1] * 10)
    assert classify_column(s_zero_one, "flag") == "categorical"


def test_date_sniffing_and_sampling():
    dates = pd.date_range("2024-01-01", periods=100).astype(str)
    s_date = pd.Series(dates)
    assert classify_column(s_date, "created_at") == "datetime"

    # Only 50% dates (below 90% threshold)
    mixed = ["2024-01-01", "random string", "not a date", "2024-05-02"] * 25
    s_mixed = pd.Series(mixed)
    assert classify_column(s_mixed, "random_col") != "datetime"


def test_id_vs_numeric_rules():
    # Numeric column with 6 distinct values (e.g. quantity 1 to 6) in large table -> numeric
    qty = pd.Series([1, 2, 3, 4, 5, 6] * 20)
    assert classify_column(qty, "quantity") == "numeric"

    # Numeric column with id in name -> categorical
    cust_id = pd.Series([101, 102, 103, 101, 102, 103] * 10)
    assert classify_column(cust_id, "customer_id") == "categorical"

    # True unique identifier string -> id
    unique_ids = pd.Series([f"TX_{i}" for i in range(50)])
    assert classify_column(unique_ids, "tx_code") == "id"
