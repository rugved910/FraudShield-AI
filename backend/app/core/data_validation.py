"""Dataset loading and validation helpers.

This module is deliberately strict: it must fail loudly with actionable error
messages rather than silently substituting synthetic data for the real Kaggle
dataset.  A separate, clearly-labelled demo generator is provided for
classroom demos when the real dataset cannot be downloaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    AMOUNT_COLUMN,
    DATASET_SOURCE_URL,
    RAW_DATASET_PATH,
    TARGET_COLUMN,
    TIME_COLUMN,
    V_FEATURES,
)

REQUIRED_COLUMNS = [TIME_COLUMN] + V_FEATURES + [AMOUNT_COLUMN] + [TARGET_COLUMN]


class DatasetError(Exception):
    """Raised when the dataset is missing, corrupt, or fails validation."""


def _dataset_help(path: Path) -> str:
    return (
        f"Expected dataset at: {path}\n\n"
        "How to obtain it:\n"
        f"  1. Download the public Kaggle dataset: {DATASET_SOURCE_URL}\n"
        "  2. Unzip it and place 'creditcard.csv' into data/raw/ of this project.\n"
        "     OR download with the Kaggle CLI:\n"
        "         kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip\n"
        "  3. Re-run the training script.\n\n"
        "If you cannot download the dataset, you can train a clearly-labelled\n"
        "synthetic DEMO model instead:  python scripts/train.py --demo\n"
        "(Demo models are marked as synthetic everywhere and must never be\n"
        " presented as real fraud-detection performance.)"
    )


def load_raw_csv(path: Path | None = None) -> pd.DataFrame:
    """Read the credit-card CSV from disk with clear corruption errors."""
    path = Path(path) if path is not None else RAW_DATASET_PATH
    if not path.exists():
        raise DatasetError(f"Dataset file not found.\n{_dataset_help(path)}")
    if path.stat().st_size == 0:
        raise DatasetError(f"Dataset file is empty (0 bytes).\n{_dataset_help(path)}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DatasetError(f"Dataset file contains no parsable columns: {exc}") from exc
    except pd.errors.ParserError as exc:
        raise DatasetError(
            f"Dataset file is corrupted — pandas could not parse it: {exc}"
        ) from exc
    return df


def validate_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Validate the schema and content of the credit-card dataset.

    Returns a small report dictionary.  Raises :class:`DatasetError` with a
    useful message when validation fails.
    """
    if df.empty:
        raise DatasetError("Dataset contains no rows.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(
            f"Dataset is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Every required column must be numeric.
    non_numeric = [
        c
        for c in REQUIRED_COLUMNS
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise DatasetError(
            f"Non-numeric values in required columns: {non_numeric}. "
            "The dataset appears to be corrupted or of an unexpected format."
        )

    # No NaN / infinite values allowed.
    problem_values = df[REQUIRED_COLUMNS]
    n_nan = int(problem_values.isna().sum().sum())
    n_inf = int(np.isinf(problem_values.to_numpy(dtype=float)).sum())
    if n_nan or n_inf:
        raise DatasetError(
            f"Dataset contains {n_nan} missing and {n_inf} infinite values. "
            "Clean the CSV or re-download it from Kaggle."
        )

    # Class must be a strict binary target.
    classes = set(df[TARGET_COLUMN].unique())
    if not classes.issubset({0, 1}):
        raise DatasetError(
            f"Target column '{TARGET_COLUMN}' must contain only 0 and 1; found {sorted(classes)}."
        )
    if 1 not in classes:
        raise DatasetError(
            "Dataset contains no fraud cases (Class == 1); it cannot be used "
            "to train a fraud detector."
        )

    if (df[AMOUNT_COLUMN] < 0).any():
        raise DatasetError(f"Column '{AMOUNT_COLUMN}' contains negative amounts.")
    if (df[TIME_COLUMN] < 0).any():
        raise DatasetError(f"Column '{TIME_COLUMN}' contains negative time values.")

    n_duplicates = int(df.duplicated().sum())
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "fraud_cases": int((df[TARGET_COLUMN] == 1).sum()),
        "legitimate_cases": int((df[TARGET_COLUMN] == 0).sum()),
        "fraud_percentage": float((df[TARGET_COLUMN] == 1).mean() * 100),
        "duplicate_rows": n_duplicates,
        "missing_values": 0,
        "amount_max": float(df[AMOUNT_COLUMN].max()),
        "time_span_seconds": float(df[TIME_COLUMN].max() - df[TIME_COLUMN].min()),
    }


def load_dataset(
    path: Path | None = None, *, drop_duplicates: bool = True
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load **and** validate the real Kaggle dataset.

    Returns ``(dataframe, validation_report)``.  The report includes the number
    of duplicate rows that were removed (dropping duplicates before splitting
    prevents the same transaction appearing in both train and test sets).
    """
    df = load_raw_csv(path)
    report = validate_dataframe(df)
    if drop_duplicates and report["duplicate_rows"] > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    report["rows_after_dedup"] = int(len(df))
    report["data_source"] = f"kaggle:{DATASET_SOURCE_URL}"
    return df, report


def generate_demo_dataframe(
    n_rows: int = 6000, fraud_rate: float = 0.02, seed: int = 42
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate a clearly-labelled **synthetic** dataset for demo mode.

    This data is NOT real financial data.  It mimics the schema of the Kaggle
    dataset so that the whole pipeline can run offline, and every artifact
    derived from it is stamped with ``data_source="demo-synthetic"``.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n_rows * fraud_rate))

    legitimate = pd.DataFrame(
        rng.normal(0.0, 1.0, size=(n_rows - n_fraud, 28)),
        columns=V_FEATURES,
    )
    fraudulent = pd.DataFrame(
        rng.normal(0.0, 2.0, size=(n_fraud, 28)),
        columns=V_FEATURES,
    )
    # Inject a learnable pattern so the demo model has signal to find.
    fraudulent["V1"] += 3.0
    fraudulent["V17"] -= 3.0

    demo = pd.concat([legitimate, fraudulent], ignore_index=True)
    demo[TIME_COLUMN] = rng.uniform(0, 172800, size=len(demo))
    demo[AMOUNT_COLUMN] = np.where(
        demo.index.isin(fraudulent.index[:n_fraud]),
        rng.gamma(4.0, 40.0, size=len(demo)),
        rng.gamma(2.0, 30.0, size=len(demo)),
    ).round(2)
    demo[TARGET_COLUMN] = [0] * (n_rows - n_fraud) + [1] * n_fraud

    demo = demo[[TIME_COLUMN] + V_FEATURES + [AMOUNT_COLUMN, TARGET_COLUMN]]
    demo = demo.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    report = validate_dataframe(demo)
    report["rows_after_dedup"] = int(len(demo))
    report["data_source"] = "demo-synthetic"
    report["warning"] = (
        "SYNTHETIC DEMO DATA — not real financial transactions. "
        "Metrics from this dataset do not reflect real-world fraud detection."
    )
    return demo, report
