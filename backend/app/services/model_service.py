"""Model loading, inference and explanation services for FraudShield AI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..core.config import (
    ARTIFACTS_DIR,
    FEATURE_COLUMNS,
    RISK_LOW_MAX_RATIO,
)


class ModelNotLoadedError(RuntimeError):
    """Raised when an endpoint needs the model but artifacts are missing."""


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


@dataclass
class ModelBundle:
    """Everything the API needs at inference time (no raw dataset required)."""

    pipeline: Any
    metadata: dict
    comparison: dict
    evaluation: dict
    feature_importance: dict
    dataset_stats: dict | None
    samples: dict | None
    loaded_at: str

    @property
    def threshold(self) -> float:
        return float(self.metadata["decision_threshold"])

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    @property
    def is_demo(self) -> bool:
        return bool(self.metadata.get("is_demo", False))

    @property
    def feature_columns(self) -> list[str]:
        return list(self.metadata.get("feature_columns", FEATURE_COLUMNS))


class ModelService:
    """Loads the trained artifacts once and serves predictions from memory."""

    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
        self.bundle: ModelBundle | None = None
        self.load_error: str | None = None

    # ------------------------------------------------------------------ load
    @property
    def is_loaded(self) -> bool:
        return self.bundle is not None

    def load(self) -> None:
        """Load artifacts; record (not raise) errors so /health can report them."""
        pipeline_path = self.artifacts_dir / "model_pipeline.joblib"
        metadata = _read_json(self.artifacts_dir / "model_metadata.json")
        if not pipeline_path.exists():
            self.load_error = (
                f"Trained model not found at {pipeline_path}. "
                "Run 'python scripts/train.py' first."
            )
            self.bundle = None
            return
        if metadata is None:
            self.load_error = (
                f"Model metadata not found at {self.artifacts_dir / 'model_metadata.json'}. "
                "Re-run training to regenerate artifacts."
            )
            self.bundle = None
            return
        try:
            pipeline = joblib.load(pipeline_path)
        except Exception as exc:  # corrupt/incompatible artifact
            self.load_error = f"Failed to load model artifact: {exc}"
            self.bundle = None
            return
        self.bundle = ModelBundle(
            pipeline=pipeline,
            metadata=metadata,
            comparison=_read_json(self.artifacts_dir / "model_comparison.json") or {},
            evaluation=_read_json(self.artifacts_dir / "evaluation_metrics.json") or {},
            feature_importance=_read_json(self.artifacts_dir / "feature_importance.json") or {},
            dataset_stats=_read_json(self.artifacts_dir / "dataset_stats.json"),
            samples=_read_json(self.artifacts_dir / "sample_transactions.json"),
            loaded_at=datetime.now(timezone.utc).isoformat(),
        )
        self.load_error = None

    def require(self) -> ModelBundle:
        if self.bundle is None:
            raise ModelNotLoadedError(
                self.load_error or "Model is not loaded."
            )
        return self.bundle

    # ------------------------------------------------------------- inference
    def predict_one(self, features: dict[str, float]) -> dict[str, Any]:
        result = self.predict_batch([features])[0]
        return result

    def predict_batch(self, rows: list[dict[str, float]]) -> list[dict[str, Any]]:
        bundle = self.require()
        df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        probs = bundle.pipeline.predict_proba(df)[:, 1]
        threshold = bundle.threshold
        results = []
        for features, prob in zip(rows, probs):
            prob = float(prob)
            label = "Potentially Fraudulent" if prob >= threshold else "Legitimate"
            results.append({
                "transaction_id": f"tx-{abs(hash((tuple(features[c] for c in FEATURE_COLUMNS), prob))) % 10**10:010d}",
                "predicted_class": int(prob >= threshold),
                "predicted_label": label,
                "fraud_probability": round(prob, 6),
                "risk_category": risk_category(prob, threshold),
                "threshold": threshold,
                "model_version": bundle.model_version,
                "probability_note": (
                    "Raw model score (not a calibrated probability). "
                    "It reflects similarity to fraud patterns seen in the "
                    "training data — it is NOT the true probability of fraud "
                    "and NOT a confirmed fraud label."
                ),
            })
        return results

    # ----------------------------------------------------------- explanation
    def explain(self, features: dict[str, float]) -> list[dict[str, Any]]:
        """Top local feature contributions for one transaction.

        * Linear models: contribution = standardized coefficient * scaled value.
        * Tree ensembles: global importance weighted by how far the (scaled)
          feature value sits from its training mean, with direction.

        These are statistical associations, never causal claims.
        """
        bundle = self.require()
        df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        scaled = bundle.pipeline.named_steps["preprocess"].transform(df)[0]
        model = bundle.pipeline.named_steps["model"]
        cols = bundle.pipeline.named_steps["preprocess"].get_feature_names_out()

        if hasattr(model, "coef_"):
            weights = model.coef_[0]
            basis = scaled  # contribution relative to zero
            kind = "linear"
        else:
            importances = model.feature_importances_
            # deviation from zero in scaled space (preprocessing centres Time/Amount)
            basis = scaled
            weights = importances * np.abs(basis)
            kind = "tree"

        contributions = weights * basis
        order = np.argsort(np.abs(contributions))[::-1][:3]
        direction_map = {1: "increases fraud risk", -1: "decreases fraud risk"}
        out = []
        for i in order:
            value = float(features[str(cols[i])]) if str(cols[i]) in features else float(features[FEATURE_COLUMNS[i]])
            out.append({
                "feature": str(cols[i]),
                "value": value,
                "contribution": round(float(contributions[i]), 5),
                "direction": direction_map[int(np.sign(contributions[i]) or 1)],
                "interpretation": (
                    "Standardised linear contribution to the fraud score."
                    if kind == "linear"
                    else "Anonymised PCA feature far from its typical training value, "
                         "weighted by how much the forest relies on it."
                ),
            })
        out.append({
            "feature": "_note",
            "value": 0.0,
            "contribution": 0.0,
            "direction": "n/a",
            "interpretation": (
                "V1..V28 are anonymised PCA components — they cannot be mapped to "
                "merchant, location, or cardholder attributes. These contributions "
                "show statistical association only, never causation."
            ),
        })
        return out


def risk_category(probability: float, threshold: float) -> str:
    """Bucket a fraud probability into Low / Medium / High (configurable)."""
    if probability >= threshold:
        return "high"
    if probability >= threshold * RISK_LOW_MAX_RATIO:
        return "medium"
    return "low"
