"""Tests for dataset loading and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from backend.app.core.data_validation import (
    DatasetError,
    generate_demo_dataframe,
    load_dataset,
    validate_dataframe,
)


def test_mini_dataset_has_correct_schema(mini_dataset):
    assert list(mini_dataset.columns)[:1] == ["Time"]
    assert "Amount" in mini_dataset.columns
    assert "Class" in mini_dataset.columns
    assert set(mini_dataset["Class"].unique()) <= {0, 1}


def test_validation_report_counts(mini_dataset, valid_report):
    assert valid_report["rows"] == len(mini_dataset)
    assert valid_report["fraud_cases"] + valid_report["legitimate_cases"] == len(mini_dataset)
    assert valid_report["missing_values"] == 0


def test_validate_rejects_missing_columns(mini_dataset):
    df = mini_dataset.drop(columns=["V5"])
    with pytest.raises(DatasetError, match="missing required columns"):
        validate_dataframe(df)


def test_validate_rejects_nan(mini_dataset):
    df = mini_dataset.copy()
    df.loc[0, "V1"] = float("nan")
    with pytest.raises(DatasetError, match="missing"):
        validate_dataframe(df)


def test_validate_rejects_bad_target(mini_dataset):
    df = mini_dataset.copy()
    df.loc[0, "Class"] = 2
    with pytest.raises(DatasetError, match="only 0 and 1"):
        validate_dataframe(df)


def test_validate_rejects_negative_amount(mini_dataset):
    df = mini_dataset.copy()
    df.loc[0, "Amount"] = -5.0
    with pytest.raises(DatasetError, match="negative"):
        validate_dataframe(df)


def test_validate_rejects_non_numeric(mini_dataset):
    df = mini_dataset.copy()
    df["V3"] = df["V3"].astype(str)
    with pytest.raises(DatasetError, match="Non-numeric"):
        validate_dataframe(df)


def test_missing_file_error_message(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "nope.csv")


def test_empty_file_error(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(DatasetError, match="empty"):
        load_dataset(p)


def test_corrupt_file_error(tmp_path):
    p = tmp_path / "bad.csv"
    # unclosed quote -> parser error
    p.write_text('Time,V1,"bad,data\n1,2,3')
    with pytest.raises(DatasetError, match="corrupted"):
        load_dataset(p)


def test_duplicate_removal():
    df, _ = generate_demo_dataframe(n_rows=300, seed=7)
    dup = pd.concat([df, df.head(10)], ignore_index=True)
    report = validate_dataframe(dup)
    assert report["duplicate_rows"] == 10


def test_demo_dataframe_is_labelled_synthetic():
    df, report = generate_demo_dataframe(n_rows=200, seed=1)
    assert report["data_source"] == "demo-synthetic"
    assert "SYNTHETIC" in report["warning"].upper()
