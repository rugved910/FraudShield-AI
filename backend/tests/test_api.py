"""End-to-end API tests using FastAPI's TestClient."""

from __future__ import annotations

import io

from backend.app.core.config import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# General / health
# ---------------------------------------------------------------------------
def test_root_info(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "FraudShield AI"
    assert "/docs" in body["documentation"]
    assert any("/api/predict" in e for e in body["endpoints"])


def test_health_model_ready(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_ready"] is True
    assert body["status"] == "ok"
    assert body["model_version"] == "test-1.0.0"


def test_health_without_artifacts(tmp_path):
    from fastapi.testclient import TestClient
    from backend.app.main import create_app

    app = create_app(artifacts_dir=tmp_path)
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["model_ready"] is False
        assert body["status"] == "degraded"


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------
def test_model_info(client):
    r = client.get("/api/model-info")
    assert r.status_code == 200
    body = r.json()
    for key in ("model_version", "model_name", "decision_threshold",
                "feature_schema", "data_source"):
        assert key in body
    assert body["feature_schema"]["features"] == FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------
def test_predict_valid(client, sample_features):
    r = client.post("/api/predict", json=sample_features)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in (0, 1)
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_category"] in {"low", "medium", "high"}
    assert body["disclaimer"]
    assert len(body["explanation"]) > 1


def test_predict_missing_field_rejected(client, sample_features):
    del sample_features["V7"]
    r = client.post("/api/predict", json=sample_features)
    assert r.status_code == 422


def test_predict_unexpected_field_rejected(client, sample_features):
    sample_features["CardNumber"] = 1234
    r = client.post("/api/predict", json=sample_features)
    assert r.status_code == 422


def test_predict_negative_amount_rejected(client, sample_features):
    sample_features["Amount"] = -1
    r = client.post("/api/predict", json=sample_features)
    assert r.status_code == 422


def test_predict_out_of_range_v_feature_rejected(client, sample_features):
    sample_features["V1"] = 99999
    r = client.post("/api/predict", json=sample_features)
    assert r.status_code == 422


def test_predict_non_numeric_rejected(client, sample_features):
    sample_features["V2"] = "abc"
    r = client.post("/api/predict", json=sample_features)
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------
def test_batch_prediction(client, sample_features, fraud_like_features):
    payload = {"transactions": [sample_features, fraud_like_features]}
    r = client.post("/api/predict/batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["total"] == 2
    assert len(body["results"]) == 2
    assert sum(body["summary"]["risk_distribution"].values()) == 2


def test_batch_rejects_empty(client):
    r = client.post("/api/predict/batch", json={"transactions": []})
    assert r.status_code == 422


def test_batch_rejects_too_large(client, sample_features):
    payload = {"transactions": [sample_features] * 1001}
    r = client.post("/api/predict/batch", json=payload)
    assert r.status_code == 422


def test_batch_rejects_bad_row(client, sample_features, fraud_like_features):
    bad = dict(fraud_like_features)
    bad["Amount"] = -3
    r = client.post("/api/predict/batch",
                    json={"transactions": [sample_features, bad]})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# CSV upload
# ---------------------------------------------------------------------------
def _csv_bytes(header: str, rows: list[str]) -> bytes:
    return io.BytesIO(("\n".join([header] + rows) + "\n").encode()).getvalue()


def _header() -> str:
    return ",".join(FEATURE_COLUMNS)


def _row_from(features: dict[str, float]) -> str:
    return ",".join(str(features[c]) for c in FEATURE_COLUMNS)


def test_csv_valid_upload(client, sample_features, fraud_like_features):
    content = _csv_bytes(_header(), [_row_from(sample_features),
                                     _row_from(fraud_like_features)])
    r = client.post("/api/predict/csv",
                    files={"file": ("tx.csv", content, "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["total"] == 2
    assert len(body["preview"]) == 2
    assert body["invalid_rows"] == []


def test_csv_ignores_class_column(client, sample_features):
    header = _header() + ",Class"
    row = _row_from(sample_features) + ",1"
    r = client.post("/api/predict/csv",
                    files={"file": ("tx.csv", _csv_bytes(header, [row]), "text/csv")})
    assert r.status_code == 200
    assert "Class" in r.json()["ignored_columns"]


def test_csv_invalid_rows_reported_not_fatal(client, sample_features):
    header = _header()
    bad_row = _row_from(sample_features).split(",")
    bad_row[5] = "not-a-number"  # corrupt V4
    r = client.post("/api/predict/csv",
                    files={"file": ("tx.csv",
                                    _csv_bytes(header, [_row_from(sample_features),
                                                        ",".join(bad_row)]),
                                    "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["total"] == 1
    assert len(body["invalid_rows"]) == 1
    assert "not numeric" in body["invalid_rows"][0]["reason"]


def test_csv_missing_columns_rejected(client):
    r = client.post("/api/predict/csv",
                    files={"file": ("tx.csv", b"Time,Amount\n1,2\n", "text/csv")})
    assert r.status_code == 400
    assert "Missing required columns" in r.json()["detail"]


def test_csv_wrong_extension_rejected(client):
    r = client.post("/api/predict/csv",
                    files={"file": ("tx.txt", b"a,b\n1,2\n", "text/plain")})
    assert r.status_code == 400


def test_csv_all_invalid_rejected(client, sample_features):
    bad_row = ",".join(["x"] * len(FEATURE_COLUMNS))
    r = client.post("/api/predict/csv",
                    files={"file": ("tx.csv",
                                    _csv_bytes(_header(), [bad_row]), "text/csv")})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Analytics endpoints
# ---------------------------------------------------------------------------
def test_analytics_overview(client):
    r = client.get("/api/analytics/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["dataset"]["total_transactions"] > 0
    assert body["test_metrics"]["precision"] is not None
    assert body["decision_threshold"] > 0


def test_analytics_model_comparison(client):
    r = client.get("/api/analytics/model-comparison")
    assert r.status_code == 200
    body = r.json()
    assert body["selected_model"] == "logistic_regression"
    assert "validation" in body["models"]["logistic_regression"]


def test_analytics_confusion_matrix(client):
    r = client.get("/api/analytics/confusion-matrix")
    assert r.status_code == 200
    body = r.json()
    tn, fp = body["matrix"][0]
    fn, tp = body["matrix"][1]
    assert tn + fp + fn + tp == body["true_negatives"] + body["false_positives"] \
        + body["false_negatives"] + body["true_positives"]


def test_analytics_feature_importance(client):
    r = client.get("/api/analytics/feature-importance")
    assert r.status_code == 200
    body = r.json()
    assert body["importance_type"]
    assert body["features"][0]["feature"]


def test_analytics_curves_validation(client):
    r = client.get("/api/analytics/curves?split=validation")
    # fixture artifacts do not include curves -> 404, not a crash
    assert r.status_code in (200, 404)


def test_analytics_curves_bad_split(client):
    r = client.get("/api/analytics/curves?split=banana")
    assert r.status_code == 400


def test_samples_endpoint(client):
    r = client.get("/api/samples")
    assert r.status_code == 200
    body = r.json()
    assert len(body["samples"]) >= 1
    assert set(body["samples"][0]["features"].keys()) == set(FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Missing artifacts -> prediction endpoints must fail with 503, not crash
# ---------------------------------------------------------------------------
def test_predict_without_model_returns_503(tmp_path):
    from fastapi.testclient import TestClient
    from backend.app.main import create_app
    from backend.tests.conftest import sample_features_dict
    app = create_app(artifacts_dir=tmp_path)
    with TestClient(app) as c:
        r = c.post("/api/predict", json=sample_features_dict())
        assert r.status_code == 503
        assert r.json()["error"] == "model_not_loaded"
