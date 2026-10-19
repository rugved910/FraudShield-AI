"""Central configuration for FraudShield AI.

All paths are resolved relative to the repository root so that the project
runs on any machine without hardcoded absolute paths.  Environment variable
overrides are supported for testing and deployment flexibility.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------
# backend/app/core/config.py -> parents[0]=core, [1]=app, [2]=backend, [3]=root
REPO_ROOT = Path(__file__).resolve().parents[3]

ARTIFACTS_DIR = Path(os.getenv("FRAUDSHIELD_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
DATA_DIR = Path(os.getenv("FRAUDSHIELD_DATA_DIR", REPO_ROOT / "data" / "raw"))
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

RAW_DATASET_PATH = DATA_DIR / "creditcard.csv"

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"
TARGET_COLUMN = "Class"
V_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = [TIME_COLUMN] + V_FEATURES + [AMOUNT_COLUMN]

DATASET_SOURCE_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"

# ---------------------------------------------------------------------------
# API limits
# ---------------------------------------------------------------------------
BATCH_SIZE_LIMIT = 1000
MAX_CSV_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_CSV_ROWS = 1000

# Only the local Vite dev server is allowed by default (CORS).
DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# ---------------------------------------------------------------------------
# Risk categorisation (configurable)
# ---------------------------------------------------------------------------
# A transaction is bucketed by comparing its fraud probability *p* against the
# tuned decision threshold *t*:
#   low     : p <  t * RISK_LOW_MAX_RATIO
#   medium  : t * RISK_LOW_MAX_RATIO <= p < t      (borderline — needs review)
#   high    : p >= t                                (flagged as potentially fraudulent)
RISK_LOW_MAX_RATIO = 0.5

APP_NAME = "FraudShield AI"
APP_DESCRIPTION = (
    "Academic fraud-detection demo API. Predictions are model outputs, "
    "not confirmed fraud, and must never be used to auto-block payments."
)

# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------
# When the real Kaggle dataset is unavailable the training script can be run
# with --demo to produce a clearly-labelled synthetic model.  Every artifact it
# writes carries data_source="demo-synthetic" so the UI can display a warning.
DEMO_DATA_SOURCE_LABEL = "demo-synthetic"


def cors_origins() -> list[str]:
    """Return CORS origins, honouring the FRAUDSHIELD_CORS_ORIGINS env var."""
    raw = os.getenv("FRAUDSHIELD_CORS_ORIGINS")
    if not raw:
        return list(DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]
