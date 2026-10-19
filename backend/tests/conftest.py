"""Shared pytest fixtures for FraudShield AI.

A tiny deterministic synthetic dataset is used to train a miniature model for
the tests, so the suite never depends on downloading the 150 MB Kaggle file
and never re-trains the production model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.config import FEATURE_COLUMNS, TARGET_COLUMN, V_FEATURES  # noqa: E402
from backend.app.core.data_validation import (  # noqa: E402
    generate_demo_dataframe,
    validate_dataframe,
)
from backend.app.main import create_app  # noqa: E402

SEED = 123


# ---------------------------------------------------------------------------
# Deterministic mini dataset
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def mini_dataset() -> pd.DataFrame:
    df, _ = generate_demo_dataframe(n_rows=1200, fraud_rate=0.05, seed=SEED)
    return df


@pytest.fixture(scope="session")
def valid_report(mini_dataset) -> dict:
    return validate_dataframe(mini_dataset)


# ---------------------------------------------------------------------------
# Mini trained artifacts (trained once per session)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory) -> Path:
    """Train a small logistic-regression pipeline and write artifacts."""
    root = tmp_path_factory.mktemp("artifacts")
    df, _ = generate_demo_dataframe(n_rows=1200, fraud_rate=0.05, seed=SEED)
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    pre = ColumnTransformer([
        ("scaler", StandardScaler(), ["Time", "Amount"]),
        ("pass", "passthrough", V_FEATURES),
    ])
    pipe = Pipeline([
        ("preprocess", pre),
        ("model", LogisticRegression(class_weight="balanced", max_iter=500,
                                     random_state=SEED)),
    ])
    pipe.fit(X_train, y_train)
    threshold = 0.5
    joblib.dump(pipe, root / "model_pipeline.joblib")

    (root / "model_metadata.json").write_text(json.dumps({
        "model_version": "test-1.0.0",
        "model_name": "logistic_regression",
        "data_source": "pytest-fixture",
        "is_demo": True,
        "feature_columns": FEATURE_COLUMNS,
        "decision_threshold": threshold,
        "threshold_tuning": {"method": "fixed 0.5 for tests"},
    }))
    (root / "model_comparison.json").write_text(json.dumps({
        "selection_metric": "validation_pr_auc",
        "selected_model": "logistic_regression",
        "decision_threshold": threshold,
        "models": {
            "logistic_regression": {
                "validation": {
                    "threshold": 0.5, "precision": 0.9, "recall": 0.8,
                    "f1": 0.85, "roc_auc": 0.95, "pr_auc": 0.9,
                    "true_negatives": 10, "false_positives": 1,
                    "false_negatives": 2, "true_positives": 8,
                }
            }
        },
        "test_metrics_selected_model": {
            "threshold": threshold, "precision": 0.88, "recall": 0.81,
            "f1": 0.84, "roc_auc": 0.94, "pr_auc": 0.87,
            "true_negatives": 250, "false_positives": 5,
            "false_negatives": 6, "true_positives": 26,
        },
    }))
    (root / "feature_importance.json").write_text(json.dumps({
        "model": "logistic_regression",
        "importance_type": "standardized_log_odds_coefficients",
        "features": [{"feature": "V1", "importance": 0.5},
                     {"feature": "V2", "importance": 0.2}],
    }))
    (root / "dataset_stats.json").write_text(json.dumps({
        "data_source": "pytest-fixture",
        "total_transactions": int(len(df)),
        "fraud_cases": int((df[TARGET_COLUMN] == 1).sum()),
        "legitimate_cases": int((df[TARGET_COLUMN] == 0).sum()),
        "fraud_percentage": float(df[TARGET_COLUMN].mean() * 100),
    }))
    sample = {c: float(X_test.iloc[0][c]) for c in FEATURE_COLUMNS}
    (root / "sample_transactions.json").write_text(json.dumps({
        "note": "test fixture rows",
        "data_source": "pytest-fixture",
        "samples": [{"true_label": int(y_test.iloc[0]), "features": sample}],
    }))
    return root


@pytest.fixture(scope="session")
def client(artifacts_dir) -> TestClient:
    app = create_app(artifacts_dir=artifacts_dir)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def sample_features() -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    feats = {c: float(rng.normal(0, 1)) for c in V_FEATURES}
    feats["Time"] = 50000.0
    feats["Amount"] = 120.0
    return feats


@pytest.fixture()
def fraud_like_features() -> dict[str, float]:
    """Features resembling the injected demo fraud pattern (V1 high, V17 low)."""
    rng = np.random.default_rng(SEED)
    feats = {c: float(rng.normal(0, 1)) for c in V_FEATURES}
    feats["V1"] = 3.5
    feats["V17"] = -3.5
    feats["Time"] = 90000.0
    feats["Amount"] = 400.0
    return feats


def sample_features_dict() -> dict[str, float]:
    """Plain helper (non-fixture) for use inside test bodies."""
    rng = np.random.default_rng(SEED)
    feats = {c: float(rng.normal(0, 1)) for c in V_FEATURES}
    feats["Time"] = 50000.0
    feats["Amount"] = 120.0
    return feats

