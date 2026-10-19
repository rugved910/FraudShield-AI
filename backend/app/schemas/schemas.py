"""Pydantic request/response schemas for the FraudShield AI API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.config import FEATURE_COLUMNS

# The V features are anonymised PCA components; in the published dataset they
# sit roughly within [-30, 30].  A generous bound catches corrupted inputs
# (e.g. a shifted CSV column) without rejecting legitimate outliers.
_V_BOUND = 500.0


def _v_field(name: str) -> float:
    return Field(..., description=f"Anonymised PCA component {name}",
                 ge=-_V_BOUND, le=_V_BOUND)


class TransactionFeatures(BaseModel):
    """One transaction with the 30 model input features."""

    # Reject unexpected fields loudly — a typo must not be silently ignored.
    model_config = ConfigDict(extra="forbid")

    Time: float = Field(
        ...,
        ge=0.0,
        le=200_000.0,
        description="Seconds elapsed since the first transaction in the dataset "
                    "(0 .. ~172800 for a 48-hour window).",
    )
    Amount: float = Field(
        ..., ge=0.0, le=1_000_000.0,
        description="Transaction amount in EUR. Must be non-negative.",
    )

    V1: float = _v_field("V1")
    V2: float = _v_field("V2")
    V3: float = _v_field("V3")
    V4: float = _v_field("V4")
    V5: float = _v_field("V5")
    V6: float = _v_field("V6")
    V7: float = _v_field("V7")
    V8: float = _v_field("V8")
    V9: float = _v_field("V9")
    V10: float = _v_field("V10")
    V11: float = _v_field("V11")
    V12: float = _v_field("V12")
    V13: float = _v_field("V13")
    V14: float = _v_field("V14")
    V15: float = _v_field("V15")
    V16: float = _v_field("V16")
    V17: float = _v_field("V17")
    V18: float = _v_field("V18")
    V19: float = _v_field("V19")
    V20: float = _v_field("V20")
    V21: float = _v_field("V21")
    V22: float = _v_field("V22")
    V23: float = _v_field("V23")
    V24: float = _v_field("V24")
    V25: float = _v_field("V25")
    V26: float = _v_field("V26")
    V27: float = _v_field("V27")
    V28: float = _v_field("V28")

    @field_validator(*FEATURE_COLUMNS)
    @classmethod
    def _finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("feature values must be finite numbers")
        return v

    def as_feature_dict(self) -> dict[str, float]:
        return {c: float(getattr(self, c)) for c in FEATURE_COLUMNS}


class ExplanationItem(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: str
    interpretation: str


class PredictionResult(BaseModel):
    transaction_id: str
    predicted_class: int = Field(..., description="1 = flagged as potentially fraudulent, 0 = legitimate")
    predicted_label: str
    fraud_probability: float = Field(..., description="Raw model score; not a calibrated probability")
    risk_category: Literal["low", "medium", "high"]
    threshold: float
    model_version: str
    probability_note: str


class SinglePredictionResponse(PredictionResult):
    explanation: list[ExplanationItem]
    disclaimer: str = (
        "This is a machine-learning prediction for academic demonstration. "
        "It is not a confirmed fraud finding and must never be used to "
        "automatically block a payment."
    )


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[TransactionFeatures] = Field(
        ..., min_length=1, description="1..1000 transactions"
    )

    @field_validator("transactions")
    @classmethod
    def _limit(cls, v: list) -> list:
        from ..core.config import BATCH_SIZE_LIMIT

        if len(v) > BATCH_SIZE_LIMIT:
            raise ValueError(f"batch size exceeds the limit of {BATCH_SIZE_LIMIT}")
        return v


class BatchSummary(BaseModel):
    total: int
    predicted_fraud: int
    predicted_legitimate: int
    fraud_rate_percentage: float
    risk_distribution: dict[str, int]


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResult]
    summary: BatchSummary
    model_version: str
    threshold: float


class InvalidRow(BaseModel):
    row: int
    reason: str


class CsvPredictionResponse(BaseModel):
    """Result of a CSV batch upload."""
    preview: list[dict[str, float]]
    results: list[PredictionResult]
    summary: BatchSummary
    invalid_rows: list[InvalidRow]
    ignored_columns: list[str]
    model_version: str
    threshold: float
    filename: str


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_version: str | None
    model_name: str | None
    data_source: str | None
    is_demo: bool
    loaded_at: str | None
    detail: str | None


class ApiInfo(BaseModel):
    name: str
    version: str
    description: str
    documentation: str
    endpoints: list[str]
    disclaimer: str
