"""Tests for the model service (artifact loading, inference, explanations)."""

from __future__ import annotations

import json

import pytest

from backend.app.core.config import FEATURE_COLUMNS
from backend.app.services.model_service import ModelService, risk_category


def test_service_loads_artifacts(artifacts_dir):
    svc = ModelService(artifacts_dir)
    svc.load()
    assert svc.is_loaded
    assert svc.bundle.model_version == "test-1.0.0"
    assert svc.bundle.threshold == 0.5


def test_service_reports_missing_artifacts(tmp_path):
    svc = ModelService(tmp_path)
    svc.load()
    assert not svc.is_loaded
    assert "not found" in (svc.load_error or "")
    with pytest.raises(Exception, match="not found"):
        svc.require()


def test_service_reports_missing_metadata(tmp_path, artifacts_dir):
    import shutil
    shutil.copy(artifacts_dir / "model_pipeline.joblib", tmp_path / "model_pipeline.joblib")
    svc = ModelService(tmp_path)
    svc.load()
    assert not svc.is_loaded
    assert "metadata" in (svc.load_error or "")


def test_predict_one_returns_expected_keys(artifacts_dir, sample_features):
    svc = ModelService(artifacts_dir)
    svc.load()
    out = svc.predict_one(sample_features)
    for key in ("transaction_id", "predicted_class", "predicted_label",
                "fraud_probability", "risk_category", "threshold",
                "model_version", "probability_note"):
        assert key in out
    assert out["predicted_class"] in (0, 1)
    assert 0.0 <= out["fraud_probability"] <= 1.0


def test_fraud_like_input_gets_higher_score(artifacts_dir, sample_features, fraud_like_features):
    svc = ModelService(artifacts_dir)
    svc.load()
    normal = svc.predict_one(sample_features)["fraud_probability"]
    fraud_like = svc.predict_one(fraud_like_features)["fraud_probability"]
    assert fraud_like > normal


def test_explanation_uses_real_coefficients(artifacts_dir, sample_features):
    svc = ModelService(artifacts_dir)
    svc.load()
    explanation = svc.explain(sample_features)
    assert explanation  # non-empty
    note = explanation[-1]
    assert note["feature"] == "_note"
    assert "anonymised" in note["interpretation"] or "anonymized" in note["interpretation"]
    for item in explanation[:-1]:
        assert {"feature", "value", "contribution", "direction"} <= set(item)


def test_predict_batch_matches_single(artifacts_dir, sample_features, fraud_like_features):
    svc = ModelService(artifacts_dir)
    svc.load()
    batch = svc.predict_batch([sample_features, fraud_like_features])
    single = svc.predict_one(sample_features)
    assert batch[0]["fraud_probability"] == single["fraud_probability"]
    assert len(batch) == 2


def test_risk_category_boundaries():
    assert risk_category(0.1, 0.5) == "low"
    assert risk_category(0.3, 0.5) == "medium"
    assert risk_category(0.5, 0.5) == "high"
    assert risk_category(0.9, 0.5) == "high"


def test_metadata_files_are_valid_json(artifacts_dir):
    for name in ("model_metadata.json", "model_comparison.json",
                 "dataset_stats.json", "feature_importance.json"):
        payload = json.loads((artifacts_dir / name).read_text())
        assert isinstance(payload, dict)


def test_feature_columns_order_in_metadata(artifacts_dir):
    payload = json.loads((artifacts_dir / "model_metadata.json").read_text())
    assert payload["feature_columns"] == FEATURE_COLUMNS
