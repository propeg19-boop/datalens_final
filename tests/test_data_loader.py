"""
tests/test_data_loader.py
Unit tests for datalens.data_loader matching PRD Section 5.1 acceptance criteria.
"""
import io
import pytest
import pandas as pd
from datalens.data_loader import load_data, get_excel_sheet_names


def test_load_valid_csv(tmp_path):
    df_orig = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    file_path = tmp_path / "test.csv"
    df_orig.to_csv(file_path, index=False)

    df_loaded, err = load_data(file_path)
    assert err is None
    assert df_loaded is not None
    assert df_loaded.shape == (3, 2)
    assert list(df_loaded.columns) == ["a", "b"]


def test_load_valid_excel(tmp_path):
    df_orig = pd.DataFrame({"col1": [10, 20], "col2": ["foo", "bar"]})
    file_path = tmp_path / "test.xlsx"
    df_orig.to_excel(file_path, index=False)

    df_loaded, err = load_data(file_path)
    assert err is None
    assert df_loaded is not None
    assert df_loaded.shape == (2, 2)


def test_unsupported_extension(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("just some text", encoding="utf-8")

    df_loaded, err = load_data(file_path)
    assert df_loaded is None
    assert "Unsupported file format '.txt'" in err


def test_missing_file():
    df_loaded, err = load_data("non_existent_file_12345.csv")
    assert df_loaded is None
    assert "File not found" in err


def test_empty_file(tmp_path):
    file_path = tmp_path / "empty.csv"
    file_path.write_text("", encoding="utf-8")

    df_loaded, err = load_data(file_path)
    assert df_loaded is None
    assert "empty dataset" in err or "no data" in err.lower()


def test_uploaded_buffer_bytes_io():
    csv_bytes = b"id,val\n1,10\n2,20\n3,30\n"
    buf = io.BytesIO(csv_bytes)

    df_loaded, err = load_data(buf, filename="upload.csv")
    assert err is None
    assert df_loaded is not None
    assert len(df_loaded) == 3


def test_excel_sheet_inspection(tmp_path):
    file_path = tmp_path / "multisheet.xlsx"
    with pd.ExcelWriter(file_path) as writer:
        pd.DataFrame({"x": [1]}).to_excel(writer, sheet_name="SheetOne", index=False)
        pd.DataFrame({"y": [2]}).to_excel(writer, sheet_name="SheetTwo", index=False)

    sheets, err = get_excel_sheet_names(file_path)
    assert err is None
    assert "SheetOne" in sheets
    assert "SheetTwo" in sheets


def test_latin1_encoding_fallback(tmp_path):
    file_path = tmp_path / "latin1.csv"
    # Write characters valid in latin-1 but invalid in pure strict ascii/utf-8 without declaration
    with open(file_path, "wb") as f:
        f.write("city,temperature\nCaf\xe9,25\n".encode("latin-1"))

    df_loaded, err = load_data(file_path)
    assert err is None
    assert df_loaded is not None
    assert len(df_loaded) == 1

