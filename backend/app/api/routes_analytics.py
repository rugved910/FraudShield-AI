"""Analytics endpoints backed by saved training artifacts."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


def _bundle(request: Request):
    service = request.app.state.model_service
    return service.require()  # raises ModelNotLoadedError -> 503 handler


@router.get("/overview")
def overview(request: Request) -> dict[str, Any]:
    """Dataset statistics and headline evaluation numbers."""
    bundle = _bundle(request)
    if not bundle.dataset_stats:
        raise HTTPException(404, "dataset_stats.json not found — re-run training.")
    comparison = bundle.comparison
    selected = comparison.get("selected_model")
    test_metrics = comparison.get("test_metrics_selected_model", {})
    return {
        "dataset": bundle.dataset_stats,
        "selected_model": selected,
        "model_version": bundle.model_version,
        "decision_threshold": bundle.threshold,
        "is_demo": bundle.is_demo,
        "test_metrics": {
            "precision": test_metrics.get("precision"),
            "recall": test_metrics.get("recall"),
            "f1": test_metrics.get("f1"),
            "pr_auc": test_metrics.get("pr_auc"),
            "roc_auc": test_metrics.get("roc_auc"),
        },
        "interpretation": (
            "'Fraud detection rate' in this app means recall on the held-out "
            "test set — the share of actual fraud cases the model flagged. "
            "It is measured, not estimated, and only meaningful for the "
            "modelled dataset."
        ),
    }


@router.get("/model-comparison")
def model_comparison(request: Request) -> dict[str, Any]:
    """Validation metrics of every trained model + test metrics of the winner."""
    bundle = _bundle(request)
    if not bundle.comparison:
        raise HTTPException(404, "model_comparison.json not found — re-run training.")
    return bundle.comparison


@router.get("/confusion-matrix")
def confusion_matrix(request: Request) -> dict[str, Any]:
    """Confusion matrix of the selected model on the held-out test set."""
    bundle = _bundle(request)
    test_metrics = bundle.comparison.get("test_metrics_selected_model")
    if not test_metrics:
        raise HTTPException(404, "Test-set metrics not found — re-run training.")
    return {
        "model": bundle.comparison.get("selected_model"),
        "split": "test",
        "threshold": bundle.threshold,
        "true_negatives": test_metrics["true_negatives"],
        "false_positives": test_metrics["false_positives"],
        "false_negatives": test_metrics["false_negatives"],
        "true_positives": test_metrics["true_positives"],
        "matrix": [
            [test_metrics["true_negatives"], test_metrics["false_positives"]],
            [test_metrics["false_negatives"], test_metrics["true_positives"]],
        ],
        "labels": ["Legitimate", "Fraud"],
        "note": (
            "False negatives are actual fraud transactions the model missed; "
            "false positives are legitimate transactions it flagged for review."
        ),
    }


@router.get("/feature-importance")
def feature_importance(request: Request) -> dict[str, Any]:
    """Feature importance (trees) or standardised coefficients (linear model)."""
    bundle = _bundle(request)
    if not bundle.feature_importance:
        raise HTTPException(404, "feature_importance.json not found — re-run training.")
    return bundle.feature_importance


@router.get("/curves")
def curves(request: Request, split: str = "validation") -> dict[str, Any]:
    """ROC / precision-recall curves for the validation or test split."""
    bundle = _bundle(request)
    evaluation = bundle.evaluation
    if split not in {"validation", "test"}:
        raise HTTPException(400, "split must be 'validation' or 'test'.")
    if split == "validation":
        data = evaluation.get("validation_curves")
    else:
        data = evaluation.get("test_curves")
    if not data:
        raise HTTPException(404, f"No {split} curves found — re-run training.")
    return {"split": split, "curves": data, "selected_model": evaluation.get("selected_model")}
