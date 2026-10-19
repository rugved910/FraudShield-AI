"""Tests for preprocessing and leakage prevention."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from backend.app.core.config import FEATURE_COLUMNS, V_FEATURES
from scripts.train import (
    build_preprocessor,
    classification_metrics,
    tune_threshold,
)

SEED = 123


def test_scaler_is_fitted_on_train_only(mini_dataset):
    """The scaler's statistics must match the training fold, not the full data."""
    X = mini_dataset[FEATURE_COLUMNS]
    X_train, X_test = train_test_split(X, test_size=0.25, random_state=SEED)
    pre = build_preprocessor()
    pre.fit(X_train)

    scaler = pre.named_transformers_["time_amount_scaler"]
    assert scaler.mean_[0] == pytest.approx(X_train["Time"].mean())
    assert scaler.mean_[1] == pytest.approx(X_train["Amount"].mean())
    # Deliberately NOT the full-dataset mean -> proves no leakage.
    assert scaler.mean_[1] != pytest.approx(X["Amount"].mean())


def test_v_features_are_not_rescaled(mini_dataset):
    """V1..V28 are already PCA-scaled and must pass through untouched."""
    pre = build_preprocessor()
    pre.fit(mini_dataset[FEATURE_COLUMNS])
    transformed = pre.transform(mini_dataset[FEATURE_COLUMNS])
    # columns 2..29 are the passthrough V features
    np.testing.assert_allclose(
        transformed[:, 2:30],
        mini_dataset[V_FEATURES].to_numpy(),
        rtol=1e-7,
    )


def test_transform_preserves_feature_order(mini_dataset):
    pre = build_preprocessor()
    pre.fit(mini_dataset[FEATURE_COLUMNS])
    names = list(pre.get_feature_names_out())
    assert names[:2] == ["Time", "Amount"]
    assert names[2:] == V_FEATURES


def test_classification_metrics_counts():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.6, 0.9, 0.3, 0.2, 0.8])
    m = classification_metrics(y_true, y_prob, threshold=0.5)
    # predictions at 0.5: 0,1,1,0,0,1 -> TP=2, FP=1, FN=1, TN=2
    assert (m["true_positives"], m["false_positives"],
            m["false_negatives"], m["true_negatives"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)


def test_tune_threshold_returns_valid_threshold(mini_dataset):
    rng = np.random.default_rng(SEED)
    y_true = (rng.random(500) < 0.1).astype(int)
    y_prob = np.clip(y_true * rng.uniform(0.4, 1.0, 500) +
                      (1 - y_true) * rng.uniform(0.0, 0.6, 500), 0, 1)
    thr, sweep = tune_threshold(y_true, y_prob)
    assert 0.0 < thr < 1.0
    assert len(sweep) > 10
    assert {"threshold", "precision", "recall", "f1"} <= set(sweep[0])


def test_split_manifest_preserves_fraud_ratio(mini_dataset):
    """Stratified splitting keeps fraud rate consistent across splits."""
    y = mini_dataset["Class"]
    X = mini_dataset[FEATURE_COLUMNS]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    assert abs(X_train.shape[0] / len(mini_dataset) - 0.8) < 0.01
    # fraud counts scale with split sizes
    assert abs((y_train.mean() / y.mean()) - 1) < 0.01
