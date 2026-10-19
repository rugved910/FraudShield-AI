"""Prediction endpoints: single, batch and CSV upload."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from ..schemas.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchSummary,
    CsvPredictionResponse,
    InvalidRow,
    PredictionResult,
    SinglePredictionResponse,
    TransactionFeatures,
)
from ..services.csv_service import CsvValidationError, parse_transactions_csv
from ..services.model_service import ModelService

router = APIRouter()


def _service(request: Request) -> ModelService:
    return request.app.state.model_service


def _summarise(results: list[dict]) -> BatchSummary:
    total = len(results)
    fraud = sum(1 for r in results if r["predicted_class"] == 1)
    risk = {"low": 0, "medium": 0, "high": 0}
    for r in results:
        risk[r["risk_category"]] += 1
    return BatchSummary(
        total=total,
        predicted_fraud=fraud,
        predicted_legitimate=total - fraud,
        fraud_rate_percentage=round(fraud / total * 100, 2) if total else 0.0,
        risk_distribution=risk,
    )


@router.post("/predict", response_model=SinglePredictionResponse, tags=["predictions"])
def predict_single(transaction: TransactionFeatures, request: Request):
    """Score one transaction and return class, probability, risk and explanation."""
    service = _service(request)
    service.require()  # fail fast with 503 if the model is not loaded
    features = transaction.as_feature_dict()
    result = service.predict_one(features)
    explanation = service.explain(features)
    return SinglePredictionResponse(**result, explanation=explanation)


@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["predictions"])
def predict_batch(payload: BatchPredictionRequest, request: Request):
    """Score up to 1,000 transactions supplied as JSON."""
    service = _service(request)
    bundle = service.require()
    rows = [t.as_feature_dict() for t in payload.transactions]
    results = service.predict_batch(rows)
    return BatchPredictionResponse(
        results=[PredictionResult(**r) for r in results],
        summary=_summarise(results),
        model_version=bundle.model_version,
        threshold=bundle.threshold,
    )


@router.post("/predict/csv", response_model=CsvPredictionResponse, tags=["predictions"])
async def predict_csv(request: Request,
                      file: Annotated[UploadFile, File(description="CSV with the 30 feature columns")]):
    """Upload a CSV of transactions; get per-row predictions plus a summary.

    The ``Class`` column is *not* required and is never used as a feature.
    Invalid rows are reported individually instead of failing the whole file.
    """
    service = _service(request)
    bundle = service.require()
    raw = await file.read()
    try:
        rows, invalid, ignored, _warnings = parse_transactions_csv(
            raw, file.filename or "upload.csv"
        )
    except CsvValidationError as exc:
        return JSONResponse(status_code=400, content={"error": "invalid_csv", "detail": str(exc)})

    results = service.predict_batch(rows)
    return CsvPredictionResponse(
        preview=rows[:10],
        results=[PredictionResult(**r) for r in results],
        summary=_summarise(results),
        invalid_rows=[InvalidRow(**ir) for ir in invalid],
        ignored_columns=ignored,
        model_version=bundle.model_version,
        threshold=bundle.threshold,
        filename=file.filename or "upload.csv",
    )


@router.get("/samples", tags=["predictions"])
def get_samples(request: Request):
    """Return real held-out sample transactions for the demo loader."""
    service = _service(request)
    bundle = service.require()
    if not bundle.samples:
        return {"note": "No sample transactions were saved during training.", "samples": []}
    return bundle.samples
