"""FastAPI application factory for FraudShield AI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core.config import APP_DESCRIPTION, APP_NAME, cors_origins
from .schemas.schemas import ApiInfo
from .services.model_service import ModelNotLoadedError, ModelService
from .api.routes_analytics import router as analytics_router
from .api.routes_predictions import router as predictions_router
API_VERSION = "1.0.0"


def create_app(artifacts_dir: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load once at startup so predictions never touch the raw dataset.
        app.state.model_service.load()
        yield

    app = FastAPI(
        title=APP_NAME,
        description=APP_DESCRIPTION,
        version=API_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.model_service = ModelService(artifacts_dir)

    @app.exception_handler(ModelNotLoadedError)
    async def _model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
        return JSONResponse(
            status_code=503,
            content={
                "error": "model_not_loaded",
                "detail": str(exc),
                "hint": "Train the model first:  python scripts/train.py",
            },
        )

    @app.get("/", response_model=ApiInfo, tags=["general"])
    def root() -> ApiInfo:
        return ApiInfo(
            name=APP_NAME,
            version=API_VERSION,
            description=(
                "Academic credit-card fraud detection API. Trained with "
                "scikit-learn / imbalanced-learn on the public Kaggle "
                "'mlg-ulb/creditcardfraud' dataset."
            ),
            documentation="/docs",
            endpoints=[
                "GET /health",
                "GET /api/model-info",
                "POST /api/predict",
                "POST /api/predict/batch",
                "POST /api/predict/csv",
                "GET /api/samples",
                "GET /api/analytics/overview",
                "GET /api/analytics/model-comparison",
                "GET /api/analytics/confusion-matrix",
                "GET /api/analytics/feature-importance",
                "GET /api/analytics/curves",
            ],
            disclaimer=(
                "Predictions are model outputs for education only — not "
                "confirmed fraud and never a basis for auto-blocking payments."
            ),
        )

    @app.get("/health", tags=["general"])
    def health() -> dict:
        service: ModelService = app.state.model_service
        if service.is_loaded:
            bundle = service.bundle
            return {
                "status": "ok",
                "model_ready": True,
                "model_version": bundle.model_version,
                "model_name": bundle.metadata.get("model_name"),
                "data_source": bundle.metadata.get("data_source"),
                "is_demo": bundle.is_demo,
                "loaded_at": bundle.loaded_at,
                "detail": None,
            }
        return {
            "status": "degraded",
            "model_ready": False,
            "model_version": None,
            "model_name": None,
            "data_source": None,
            "is_demo": False,
            "loaded_at": None,
            "detail": service.load_error or "Model not loaded.",
        }

    @app.get("/api/model-info", tags=["general"])
    def model_info() -> dict:
        """Model version, training date, threshold, feature schema and metadata."""
        service: ModelService = app.state.model_service
        bundle = service.require()
        meta = bundle.metadata
        return {
            "model_version": meta.get("model_version"),
            "model_name": meta.get("model_name"),
            "trained_at_utc": meta.get("trained_at_utc"),
            "data_source": meta.get("data_source"),
            "is_demo": bundle.is_demo,
            "decision_threshold": bundle.threshold,
            "threshold_tuning": meta.get("threshold_tuning"),
            "risk_categories": meta.get("risk_categories"),
            "preprocessing": meta.get("preprocessing"),
            "split_sizes": meta.get("split_sizes"),
            "random_seed": meta.get("random_seed"),
            "selection_metric": bundle.comparison.get("selection_metric"),
            "selection_reason": bundle.comparison.get("selection_reason"),
            "feature_schema": {
                "features": meta.get("feature_columns"),
                "notes": [
                    "V1..V28 are anonymised PCA components computed by the "
                    "dataset authors; they cannot be mapped to real-world attributes.",
                    "Time = seconds since the first transaction of the 48-hour window.",
                    "Amount = transaction amount in EUR.",
                ],
            },
            "probability_note": (
                "fraud_probability is the raw decision score of the model. "
                "Probability calibration has NOT been verified, so it must not "
                "be read as the true probability that a transaction is fraudulent."
            ),
        }

    app.include_router(predictions_router, prefix="/api", tags=["predictions"])
    app.include_router(analytics_router, prefix="/api/analytics", tags=["analytics"])
    return app


app = create_app()
