"""
datalens/data_loader.py
Pure Python/pandas data loading module for DataLens.
Accepts file paths and uploaded file buffers (CSV, Excel).
Validates format, size, and structure with graceful error handling.
"""
from pathlib import Path
from typing import Tuple, Optional, Union, BinaryIO, List
import io
import os
import pandas as pd

from datalens.config import (
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    SUPPORTED_EXTENSIONS,
    DEFAULT_CSV_ENCODINGS,
)


def _get_extension_and_size(
    file_path_or_buffer: Union[str, Path, io.BytesIO, io.StringIO, BinaryIO],
    filename: Optional[str] = None,
) -> Tuple[Optional[str], int, Optional[str]]:
    """
    Extract file extension and size in bytes.
    Returns: (ext, size_in_bytes, error_message)
    """
    resolved_name = filename

    if isinstance(file_path_or_buffer, (str, Path)):
        p = Path(file_path_or_buffer)
        if not p.exists():
            return None, 0, f"File not found: '{p}'"
        resolved_name = resolved_name or p.name
        try:
            size = p.stat().st_size
        except Exception as e:
            return None, 0, f"Cannot inspect file size: {e}"
    else:
        # Buffer-like object
        if hasattr(file_path_or_buffer, "size") and getattr(file_path_or_buffer, "size") is not None:
            size = getattr(file_path_or_buffer, "size")
        elif hasattr(file_path_or_buffer, "getbuffer"):
            size = file_path_or_buffer.getbuffer().nbytes
        elif hasattr(file_path_or_buffer, "seek") and hasattr(file_path_or_buffer, "tell"):
            curr = file_path_or_buffer.tell()
            file_path_or_buffer.seek(0, os.SEEK_END)
            size = file_path_or_buffer.tell()
            file_path_or_buffer.seek(curr)
        else:
            size = 0

        if not resolved_name and hasattr(file_path_or_buffer, "name"):
            resolved_name = getattr(file_path_or_buffer, "name")

    if not resolved_name:
        return None, size, "Cannot determine file format: no filename provided for uploaded buffer."

    ext = Path(resolved_name).suffix.lower()
    return ext, size, None


def get_excel_sheet_names(
    file_path_or_buffer: Union[str, Path, io.BytesIO, BinaryIO],
    filename: Optional[str] = None,
) -> Tuple[List[str], Optional[str]]:
    """
    Retrieve sheet names from an Excel file without loading the whole dataset.
    Returns: (list_of_sheet_names, error_message)
    """
    try:
        if hasattr(file_path_or_buffer, "seek"):
            file_path_or_buffer.seek(0)
        excel_file = pd.ExcelFile(file_path_or_buffer)
        return excel_file.sheet_names, None
    except Exception as e:
        return [], f"Failed to inspect Excel sheets: {str(e)}"


def load_data(
    file_path_or_buffer: Union[str, Path, io.BytesIO, io.StringIO, BinaryIO],
    filename: Optional[str] = None,
    sheet_name: Union[str, int] = 0,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Read supported file (.csv, .xlsx, .xls) into a pandas DataFrame.
    Returns: (df, error_message) where exactly one is None.
    """
    ext, size, err = _get_extension_and_size(file_path_or_buffer, filename)
    if err:
        return None, err

    if ext not in SUPPORTED_EXTENSIONS:
        return None, (
            f"Unsupported file format '{ext}'. "
            f"DataLens supports: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if size == 0:
        return None, "File contains no data (empty file)."

    if size > MAX_UPLOAD_SIZE_BYTES:
        return None, (
            f"File exceeds maximum upload size of {MAX_UPLOAD_SIZE_MB}MB "
            f"({size / (1024 * 1024):.1f}MB detected)."
        )

    # Ensure buffer is at start
    if hasattr(file_path_or_buffer, "seek"):
        file_path_or_buffer.seek(0)

    df: Optional[pd.DataFrame] = None

    try:
        if ext == ".csv":
            read_error = None
            for encoding in DEFAULT_CSV_ENCODINGS:
                try:
                    if hasattr(file_path_or_buffer, "seek"):
                        file_path_or_buffer.seek(0)
                    df = pd.read_csv(file_path_or_buffer, encoding=encoding)
                    read_error = None
                    break
                except pd.errors.EmptyDataError:
                    return None, "File contains no data rows (empty dataset)."
                except UnicodeDecodeError:
                    continue
                except Exception as ex:
                    read_error = ex
                    break

            if df is None:
                if read_error is not None:
                    return None, f"Failed to read CSV: {str(read_error)}"
                return None, "Failed to decode CSV with supported encodings (UTF-8, Latin-1, CP1252)."

        elif ext in {".xlsx", ".xls"}:
            if hasattr(file_path_or_buffer, "seek"):
                file_path_or_buffer.seek(0)
            df = pd.read_excel(file_path_or_buffer, sheet_name=sheet_name)

    except pd.errors.EmptyDataError:
        return None, "File contains no data rows (empty dataset)."
    except Exception as e:
        return None, f"Error parsing {ext.upper()} file: {str(e)}"

    if df is None:
        return None, "File could not be parsed."

    if len(df) == 0:
        return None, "File contains no data rows (empty dataset)."

    return df, None
