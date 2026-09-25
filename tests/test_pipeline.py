"""End-to-end pipeline behaviour: training, artefacts, CLV and explainability.

These tests train a real (small) model, so they are the slowest in the suite and
the ones that would catch a break in the wiring between stages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from returns_clv.artifacts import ModelBundle, load_bundle, save_bundle
from returns_clv.clv import assign_risk_bands, compute_clv, customer_table, segment_summary
from returns_clv.models import (
    DISPLAY_NAMES,
    SUPPORTED_MODELS,
    build_pipeline,
    class_ratio,
    search_space,
)

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SUPPORTED_MODELS)
def test_every_model_fits_and_predicts_probabilities(base_config, dataset, name):
    from returns_clv.features import select_model_inputs

    X = select_model_inputs(dataset, base_config)
    y = dataset["returned"].to_numpy()
    pipeline = build_pipeline(name, base_config, class_ratio=class_ratio(y))
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_resampling_lives_inside_the_estimator(base_config):
    """The whole point: SMOTE is a pipeline step, so CV resamples per fold."""
    pipeline = build_pipeline("logistic_regression", base_config)
    assert "resample" in dict(pipeline.steps)
    step_names = [name for name, _ in pipeline.steps]
    assert step_names.index("resample") < step_names.index("classifier")


def test_resampling_can_be_disabled_in_favour_of_class_weights(base_config, dataset):
    config = type(base_config)(
        **{**base_config.__dict__, "training": type(base_config.training)(resampler="none")}
    )
    pipeline = build_pipeline("logistic_regression", config)
    assert "resample" not in dict(pipeline.steps)
    assert pipeline.named_steps["classifier"].class_weight == "balanced"


def test_unknown_model_is_rejected(base_config):
    with pytest.raises(ValueError, match="unknown model"):
        build_pipeline("neural_vibes", base_config)
    with pytest.raises(ValueError, match="unknown model"):
        search_space("neural_vibes", base_config)


def test_search_spaces_target_the_classifier_step(base_config):
    for name in ("logistic_regression", "random_forest", "xgboost"):
        assert all(key.startswith("classifier__") for key in search_space(name, base_config))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_training_writes_every_declared_artefact(base_config, trained_summary):
    assert (base_config.paths.models_path / "return_risk_model.joblib").exists()
    reports = base_config.paths.reports_path
    for name in ("model_comparison.csv", "policy_comparison.csv", "training_summary.json"):
        assert (reports / name).exists(), name
    assert list(base_config.paths.figures_path.glob("*.png"))


def test_threshold_is_never_fitted_on_the_test_set(trained_summary):
    assert "validation" in trained_summary["threshold"]["fitted_on"]
    assert "test" not in trained_summary["threshold"]["fitted_on"].replace("threshold", "")


def test_selected_model_beats_the_majority_baseline(trained_summary):
    candidates = {c["model"]: c for c in trained_summary["candidates"]}
    baseline = candidates[DISPLAY_NAMES["dummy"]]["test_at_0.5"]
    assert trained_summary["deployed_test_metrics"]["roc_auc"] > baseline["roc_auc"]


def test_reported_performance_stays_under_the_achievable_ceiling(trained_summary):
    """A model beating the true generating probability would mean a leak."""
    model_auc = trained_summary["deployed_test_metrics"]["roc_auc"]
    ceiling = trained_summary["achievable_ceiling"]["roc_auc"]
    assert model_auc <= ceiling + 0.02, "model exceeds the information-theoretic ceiling"


def test_split_sizes_are_recorded(trained_summary):
    split = trained_summary["split"]
    assert split["train"]["n_rows"] > split["test"]["n_rows"]
    assert split["strategy"] == "temporal"


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------


def test_bundle_round_trips(base_config, bundle, tmp_path):
    path = save_bundle(bundle, tmp_path)
    reloaded = load_bundle(path)
    assert reloaded.model_name == bundle.model_name
    assert reloaded.threshold == bundle.threshold
    assert reloaded.feature_columns == bundle.feature_columns


def test_load_bundle_gives_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="returns-clv train"):
        load_bundle(tmp_path)


def test_bundle_rejects_an_impossible_threshold(bundle):
    with pytest.raises(ValueError, match="threshold must be in"):
        ModelBundle(
            pipeline=bundle.pipeline,
            threshold=1.4,
            model_name="x",
            feature_columns=bundle.feature_columns,
            categorical_levels={},
        )


def test_bundle_scores_raw_records(bundle, sample_transaction):
    proba = bundle.predict_proba([sample_transaction])
    assert proba.shape == (1,)
    assert 0 <= proba[0] <= 1


def test_bundle_rejects_incomplete_records(bundle, sample_transaction):
    del sample_transaction["click_depth"]
    with pytest.raises(KeyError, match="missing required feature columns"):
        bundle.predict_proba([sample_transaction])


def test_bundle_ignores_column_order(bundle, sample_transaction):
    reversed_record = dict(reversed(list(sample_transaction.items())))
    assert bundle.predict_proba([sample_transaction])[0] == pytest.approx(
        bundle.predict_proba([reversed_record])[0]
    )


def test_value_aware_flag_needs_order_value(bundle, sample_transaction):
    del sample_transaction["order_value_gbp"]
    with pytest.raises(KeyError, match="order_value_gbp"):
        bundle.flag_value_aware([sample_transaction])


def test_bundle_describes_itself_serialisably(bundle):
    description = bundle.describe()
    assert description["model_name"] == bundle.model_name
    assert set(description["categorical_levels"]) == {
        "product_category",
        "payment_method",
        "device_type",
        "customer_segment",
    }


# ---------------------------------------------------------------------------
# CLV
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def customers(base_config, dataset, bundle):
    scores = pd.Series(bundle.predict_proba(dataset), index=dataset.index)
    table = customer_table(dataset, scores, base_config)
    table = compute_clv(table, base_config.economics, base_config.clv)
    return assign_risk_bands(table, base_config.clv)


def test_customer_table_has_one_row_per_customer(dataset, customers):
    assert len(customers) == dataset["customer_id"].nunique()
    assert customers["customer_id"].is_unique


def test_customer_frequency_comes_from_observed_history(customers):
    """Not from the dataset column that validation flags as unreliable."""
    assert (customers["annual_orders"] > 0).all()
    assert not customers["annual_orders"].equals(customers["stated_order_frequency_12m"])


def test_clv_falls_as_return_risk_rises(base_config, customers):
    """Same spend, higher predicted return rate, lower lifetime value."""
    low = customers.copy()
    low["predicted_return_prob"] = 0.05
    high = customers.copy()
    high["predicted_return_prob"] = 0.65
    low_clv = compute_clv(low, base_config.economics, base_config.clv)["clv_gbp"]
    high_clv = compute_clv(high, base_config.economics, base_config.clv)["clv_gbp"]
    assert (low_clv > high_clv).all()


def test_zero_discount_and_full_retention_give_simple_multiples(base_config, customers):
    from dataclasses import replace

    clv_config = replace(
        base_config.clv, horizon_years=2, discount_rate=0.0, annual_retention_rate=1.0
    )
    valued = compute_clv(customers, base_config.economics, clv_config)
    assert valued["clv_gbp"].to_numpy() == pytest.approx(
        2 * valued["annual_contribution_gbp"].to_numpy()
    )


def test_intervention_reduces_expected_return_cost(base_config, customers):
    treated = np.ones(len(customers), dtype=bool)
    with_policy = compute_clv(
        customers, base_config.economics, base_config.clv, intervention=treated, suffix="_p"
    )
    assert (
        with_policy["expected_return_cost_gbp_p"] <= with_policy["expected_return_cost_gbp"] + 1e-9
    ).all()
    assert (with_policy["intervention_spend_gbp_p"] > 0).all()


def test_risk_bands_partition_the_customer_base(base_config, customers):
    assert set(customers["risk_segment"].unique()) <= {"Low_Risk", "Medium_Risk", "High_Risk"}
    summary = segment_summary(customers)
    assert summary["customers"].sum() == len(customers)
    assert summary["share_of_customers"].sum() == pytest.approx(1.0)


def test_risk_band_thresholds_are_respected(base_config, customers):
    bands = base_config.clv.risk_bands
    low = customers[customers["risk_segment"] == "Low_Risk"]
    assert (low["predicted_return_prob"] < bands.low).all()


def test_out_of_sample_scores_cover_every_transaction(base_config, dataset, bundle):
    from returns_clv.clv import out_of_sample_scores

    scores = out_of_sample_scores(dataset, bundle, base_config)
    assert len(scores) == len(dataset)
    assert not scores.isna().any()
    assert ((scores >= 0) & (scores <= 1)).all()


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------


def test_single_prediction_explanation_ranks_contributions(bundle, sample_transaction):
    from returns_clv.explain import explain_single

    contributions = explain_single(bundle, sample_transaction, top_n=4)
    assert len(contributions) == 4
    magnitudes = [abs(c["contribution"]) for c in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert all(
        c["direction"] in {"increases return risk", "reduces return risk"} for c in contributions
    )
