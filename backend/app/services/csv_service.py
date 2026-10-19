"""CSV parsing and validation for the batch-analysis endpoint."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import FEATURE_COLUMNS, MAX_CSV_ROWS, MAX_CSV_UPLOAD_BYTES


class CsvValidationError(ValueError):
    """Raised when an uploaded CSV is unusable (wrong type, size, headers...)."""


def parse_transactions_csv(raw: bytes, filename: str) -> tuple[
    list[dict[str, float]], list[dict[str, Any]], list[str], list[str]
]:
    """Parse an uploaded CSV into validated transaction rows.

    Returns ``(rows, invalid_rows, ignored_columns, warnings)``.

    * ``rows``       — dicts of the 30 numeric feature columns (validated).
    * ``invalid_rows`` — ``{"row": <1-based data row>, "reason": ...}`` entries
      for rows that were skipped (non-numeric / missing / out of range).
    * ``ignored_columns`` — extra columns present in the file (e.g. ``Class``)
      that are NOT used as prediction features.
    """
    if not filename.lower().endswith(".csv"):
        raise CsvValidationError(
            f"Unsupported file type '{filename}'. Please upload a .csv file."
        )
    if len(raw) == 0:
        raise CsvValidationError("The uploaded file is empty.")
    if len(raw) > MAX_CSV_UPLOAD_BYTES:
        raise CsvValidationError(
            f"File is too large ({len(raw) / 1e6:.1f} MB). "
            f"The limit is {MAX_CSV_UPLOAD_BYTES / 1e6:.0f} MB "
            f"(max {MAX_CSV_ROWS} rows)."
        )

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except pd.errors.EmptyDataError as exc:
        raise CsvValidationError("The CSV contains no parsable columns.") from exc
    except pd.errors.ParserError as exc:
        raise CsvValidationError(f"Malformed CSV: {exc}") from exc

    if df.empty:
        raise CsvValidationError("The CSV contains a header but no data rows.")

    # Normalise headers: strip whitespace, drop a BOM if present.
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise CsvValidationError(
            f"Missing required columns: {missing}. "
            f"Your file has: {list(df.columns)}. "
            "A 'Class' label column is NOT required for prediction — "
            "any extra columns are simply ignored."
        )

    if len(df) > MAX_CSV_ROWS:
        raise CsvValidationError(
            f"Too many rows ({len(df)}). The limit is {MAX_CSV_ROWS} transactions per upload."
        )

    ignored = [c for c in df.columns if c not in FEATURE_COLUMNS]
    warnings: list[str] = []
    if "Class" in ignored:
        warnings.append(
            "A 'Class' column was found and ignored — labels are never used "
            "as prediction features."
        )

    rows: list[dict[str, float]] = []
    invalid: list[dict[str, Any]] = []
    for position, (_, series) in enumerate(df.iterrows(), start=1):
        row: dict[str, float] = {}
        problem: str | None = None
        for col in FEATURE_COLUMNS:
            value = series[col]
            if pd.isna(value):
                problem = f"column '{col}' is missing/empty"
                break
            try:
                num = float(value)
            except (TypeError, ValueError):
                problem = f"column '{col}' is not numeric (value={value!r})"
                break
            if not np.isfinite(num):
                problem = f"column '{col}' is not a finite number"
                break
            row[col] = num
        if problem is None:
            if row["Amount"] < 0:
                problem = "column 'Amount' is negative"
            elif row["Time"] < 0:
                problem = "column 'Time' is negative"
        if problem is None:
            rows.append(row)
        else:
            invalid.append({"row": position, "reason": problem})

    if not rows:
        raise CsvValidationError(
            "None of the rows could be validated. First error: "
            + (invalid[0]["reason"] if invalid else "unknown")
        )

    return rows, invalid, ignored, warnings
