"""FastAPI service contract."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client(base_config, trained_summary, monkeypatch_module):
    """A client whose service loads the bundle trained by the fixtures."""
    monkeypatch_module.setenv("RETURNS_CLV_MODEL_PATH", str(base_config.paths.models_path))
    import api.main as api_main

    with TestClient(api_main.app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    patcher = MonkeyPatch()
    yield patcher
    patcher.undo()


def test_health_reports_a_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True


def test_model_endpoint_exposes_provenance(client):
    payload = client.get("/model").json()
    assert payload["model_name"]
    assert payload["trained_at"]
    assert 0 < payload["threshold"] < 1
    assert len(payload["feature_columns"]) == 10


def test_predict_scores_a_transaction(client, sample_transaction):
    response = client.post("/predict", json=[sample_transaction])
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 1
    prediction = body["predictions"][0]
    assert 0 <= prediction["return_probability"] <= 1
    assert prediction["risk_band"] in {"Low_Risk", "Medium_Risk", "High_Risk"}
    assert prediction["recommended_action"]


def test_predict_handles_a_batch(client, sample_transaction):
    batch = [sample_transaction, {**sample_transaction, "customer_segment": "Wholesale"}]
    body = client.post("/predict", json=batch).json()
    assert len(body["predictions"]) == 2


def test_high_risk_profile_scores_above_low_risk_profile(client, sample_transaction):
    risky = client.post("/predict", json=[sample_transaction]).json()["predictions"][0]
    safe_transaction = {
        **sample_transaction,
        "payment_method": "Credit_Card",
        "customer_segment": "Wholesale",
        "product_category": "Electronics",
        "click_depth": 9,
        "time_on_page_seconds": 280,
    }
    safe = client.post("/predict", json=[safe_transaction]).json()["predictions"][0]
    assert risky["return_probability"] > safe["return_probability"]


def test_expected_value_flag_tracks_order_value(client, sample_transaction):
    """The same risk profile on a larger order is more worth intervening on."""
    cheap = client.post("/predict", json=[{**sample_transaction, "order_value_gbp": 12.0}]).json()
    pricey = client.post("/predict", json=[{**sample_transaction, "order_value_gbp": 950.0}]).json()
    assert (
        pricey["predictions"][0]["expected_intervention_saving_gbp"]
        > cheap["predictions"][0]["expected_intervention_saving_gbp"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"product_category": "Groceries"},
        {"order_value_gbp": -5},
        {"click_depth": 0},
        {"payment_method": "Bitcoin"},
        {"time_on_page_seconds": -1},
    ],
)
def test_invalid_input_is_rejected_with_422(client, sample_transaction, mutation):
    response = client.post("/predict", json=[{**sample_transaction, **mutation}])
    assert response.status_code == 422


def test_unknown_field_is_rejected(client, sample_transaction):
    response = client.post("/predict", json=[{**sample_transaction, "sneaky": 1}])
    assert response.status_code == 422


def test_empty_batch_is_rejected(client):
    assert client.post("/predict", json=[]).status_code == 422


def test_explain_returns_ranked_contributions(client, sample_transaction):
    response = client.post("/explain", json=sample_transaction)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["return_probability"] <= 1
    assert body["top_contributions"]
    magnitudes = [abs(c["contribution"]) for c in body["top_contributions"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert schema["info"]["title"] == "E-commerce Return Risk API"
