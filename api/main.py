"""FastAPI service exposing the trained return-risk model.

Endpoints
---------
``GET  /health``   liveness plus whether a model is loaded
``GET  /model``    provenance: which model, trained when, on what, at what threshold
``POST /predict``  score one or more transactions
``POST /explain``  per-feature contributions for a single transaction

Design notes
------------
The service loads the :class:`~returns_clv.artifacts.ModelBundle` produced by
training, so the preprocessing applied at serving time is byte-identical to the
preprocessing applied during training - there is no second implementation of the
feature logic to drift out of sync.

Requests are validated by Pydantic against the documented category levels and
value ranges, so a malformed or out-of-domain transaction is rejected with a 422
rather than silently scored. Those levels are declared statically in the schema
(Pydantic builds its validators at import time, before any model is loaded), so
startup cross-checks them against the levels the loaded bundle was trained on
and warns on any drift. Every response carries both decisions:
the global-threshold flag and the value-aware expected-value flag, along with the
pounds at stake, because the right action depends on the order's value and not
on probability alone.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from returns_clv import __version__
from returns_clv.artifacts import ModelBundle, load_bundle
from returns_clv.config import load_config
from returns_clv.logging_utils import configure_logging, get_logger

logger = get_logger("returns_clv.api")

MAX_BATCH_SIZE = 1000

# Module-level state, populated on startup and read by the endpoints.
_state: dict[str, Any] = {"bundle": None, "config": None, "error": None}


def _model_path() -> Path:
    """Resolve the bundle location, overridable for containers and tests."""
    override = os.environ.get("RETURNS_CLV_MODEL_PATH")
    if override:
        return Path(override)
    return load_config().paths.models_path


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Load the model once at startup rather than per request."""
    configure_logging()
    _state["config"] = load_config()
    try:
        _state["bundle"] = load_bundle(_model_path())
        logger.info(
            "loaded %s trained %s (threshold %.2f)",
            _state["bundle"].model_name,
            _state["bundle"].trained_at,
            _state["bundle"].threshold,
        )
        _warn_on_schema_drift(_state["bundle"])
    except FileNotFoundError as exc:
        # Start anyway so /health can report the problem instead of crash-looping.
        _state["error"] = str(exc)
        logger.error("model unavailable: %s", exc)
    yield
    _state.clear()


def _warn_on_schema_drift(bundle: ModelBundle) -> None:
    """Compare the request schema's category levels against the model's.

    The schema cannot be built from the bundle, so this is the next best thing:
    if retraining introduces a new product category or payment method, the
    mismatch is logged at startup rather than surfacing as unexplained 422s.
    """
    from typing import get_args

    declared = {
        field: set(get_args(Transaction.model_fields[field].annotation))
        for field in bundle.categorical_levels
        if field in Transaction.model_fields
    }
    for field, values in declared.items():
        trained = set(bundle.categorical_levels[field])
        if trained - values:
            logger.warning(
                "%s: model knows %s but the request schema rejects them",
                field,
                sorted(trained - values),
            )
        if values - trained:
            logger.warning(
                "%s: request schema accepts %s, which the model never saw in training",
                field,
                sorted(values - trained),
            )


app = FastAPI(
    title="E-commerce Return Risk API",
    description=(
        "Scores a transaction for the probability that it will be returned and "
        "recommends whether intervening is worth the cost."
    ),
    version=__version__,
    lifespan=lifespan,
)


def _bundle() -> ModelBundle:
    bundle = _state.get("bundle")
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_state.get("error") or "model not loaded; run `returns-clv train` first",
        )
    return bundle


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Transaction(BaseModel):
    """One transaction to score. Field bounds mirror the training data."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "product_category": "Fashion",
                "payment_method": "Cash_on_Delivery",
                "device_type": "Mobile",
                "customer_segment": "First_Time",
                "order_value_gbp": 129.99,
                "click_depth": 2,
                "time_on_page_seconds": 35,
                "product_page_visits": 3,
                "customer_tenure_days": 12,
                "order_frequency_12m": 1,
            }
        },
    )

    product_category: Literal["Fashion", "Electronics", "Home_Garden"]
    payment_method: Literal["Credit_Card", "PayPal", "Cash_on_Delivery"]
    device_type: Literal["Mobile", "Desktop", "Tablet"]
    customer_segment: Literal["First_Time", "Repeat", "Wholesale"]
    order_value_gbp: float = Field(gt=0, le=100_000, description="Order value in GBP")
    click_depth: int = Field(ge=1, le=100, description="Pages viewed before purchase")
    time_on_page_seconds: int = Field(ge=0, le=86_400, description="Dwell time on the product page")
    product_page_visits: int = Field(ge=1, le=1000, description="Views of this product")
    customer_tenure_days: int = Field(ge=0, le=20_000, description="Days since first purchase")
    order_frequency_12m: int = Field(ge=0, le=10_000, description="Orders in the last 12 months")


class Prediction(BaseModel):
    """Scored result for one transaction."""

    return_probability: float = Field(description="Probability the order is returned")
    risk_band: str = Field(description="Low_Risk, Medium_Risk or High_Risk")
    flagged_by_threshold: bool = Field(description="Above the deployed global threshold")
    flagged_by_expected_value: bool = Field(
        description="Intervening has positive expected value for this order"
    )
    expected_return_cost_gbp: float = Field(description="Probability-weighted cost of the return")
    expected_intervention_saving_gbp: float = Field(
        description="Expected saving from intervening, net of the intervention cost"
    )
    recommended_action: str


class PredictionResponse(BaseModel):
    model_name: str
    threshold: float
    predictions: list[Prediction]


class FeatureContribution(BaseModel):
    feature: str
    contribution: float
    direction: str


class ExplanationResponse(BaseModel):
    return_probability: float
    top_contributions: list[FeatureContribution]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    api_version: str
    detail: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    """Liveness probe. Reports 'degraded' when no model is loaded."""
    loaded = _state.get("bundle") is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        api_version=__version__,
        detail=None if loaded else _state.get("error"),
    )


@app.get("/model", tags=["operations"])
def model_card() -> dict[str, Any]:
    """Provenance and headline test metrics for the loaded model."""
    return _bundle().describe()


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(
    transactions: Annotated[list[Transaction], Body(min_length=1, max_length=MAX_BATCH_SIZE)],
) -> PredictionResponse:
    """Score a batch of transactions.

    Returns both decisions - the global threshold and the expected-value rule -
    so the caller can apply whichever matches their operating policy, together
    with the pounds behind the recommendation.
    """
    bundle = _bundle()
    config = _state["config"]
    frame = pd.DataFrame([t.model_dump() for t in transactions])

    try:
        probabilities = bundle.predict_proba(frame)
    except (KeyError, ValueError) as exc:  # pragma: no cover - schema guards this
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    economics = bundle.economics
    values = frame["order_value_gbp"].to_numpy(dtype=float)
    cost_per_return = (
        values * economics["gross_margin_rate"]
        + economics["return_logistics_cost_gbp"]
        + values * economics["return_restocking_loss_rate"]
    )
    expected_cost = probabilities * cost_per_return
    expected_saving = (
        probabilities * economics["intervention_effectiveness"] * cost_per_return
        - economics["intervention_cost_gbp"]
    )
    bands = bundle.risk_band(probabilities, config.clv.risk_bands.low, config.clv.risk_bands.medium)

    predictions = [
        Prediction(
            return_probability=round(float(p), 4),
            risk_band=str(band),
            flagged_by_threshold=bool(p >= bundle.threshold),
            flagged_by_expected_value=bool(saving > 0),
            expected_return_cost_gbp=round(float(cost), 2),
            expected_intervention_saving_gbp=round(float(saving), 2),
            recommended_action=(
                "intervene: expected saving exceeds the intervention cost"
                if saving > 0
                else "no action: intervening would cost more than it saves"
            ),
        )
        for p, band, cost, saving in zip(
            probabilities, bands, expected_cost, expected_saving, strict=True
        )
    ]
    return PredictionResponse(
        model_name=bundle.model_name, threshold=bundle.threshold, predictions=predictions
    )


@app.post("/explain", response_model=ExplanationResponse, tags=["scoring"])
def explain(transaction: Transaction) -> ExplanationResponse:
    """Per-feature SHAP contributions behind a single prediction."""
    from returns_clv.explain import explain_single

    bundle = _bundle()
    record = transaction.model_dump()
    probability = float(bundle.predict_proba(pd.DataFrame([record]))[0])
    try:
        contributions = explain_single(bundle, record)
    except Exception as exc:  # pragma: no cover - explainer backend issues
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"could not explain this prediction: {exc}",
        ) from exc
    return ExplanationResponse(
        return_probability=round(probability, 4),
        top_contributions=[FeatureContribution(**c) for c in contributions],
    )
