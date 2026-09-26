"""Metrics, thresholds and the money model."""

from __future__ import annotations

import numpy as np
import pytest

from returns_clv.evaluate import (
    attainment,
    bayes_ceiling,
    choose_threshold,
    classification_metrics,
    heuristic_baseline_flags,
    net_benefit,
    policy_comparison,
    return_cost,
    summarise_models,
    threshold_sweep,
    value_aware_flags,
)


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    n = 2000
    truth = rng.beta(2, 5, n)
    y = (rng.random(n) < truth).astype(int)
    proba = np.clip(truth + rng.normal(0, 0.06, n), 0.001, 0.999)
    values = rng.lognormal(4.5, 0.8, n).clip(10, 1000)
    return y, proba, values, truth


def test_perfect_predictions_score_perfectly():
    y = np.array([0, 0, 1, 1])
    metrics = classification_metrics(y, np.array([0.01, 0.02, 0.98, 0.99]), 0.5)
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["fp"] == 0 and metrics["fn"] == 0


def test_confusion_counts_reconcile(scored):
    y, proba, _, _ = scored
    metrics = classification_metrics(y, proba, 0.4)
    assert metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == len(y)
    assert metrics["recall"] == pytest.approx(metrics["tp"] / (metrics["tp"] + metrics["fn"]))


def test_threshold_changes_decisions_not_ranking(scored):
    y, proba, _, _ = scored
    low = classification_metrics(y, proba, 0.2)
    high = classification_metrics(y, proba, 0.8)
    assert low["recall"] > high["recall"]
    assert low["roc_auc"] == pytest.approx(high["roc_auc"]), "AUC must not depend on threshold"


def test_return_cost_components(base_config):
    economics = base_config.economics
    costs = return_cost(np.array([100.0]), economics)
    assert costs[0] == pytest.approx(
        100 * economics.gross_margin_rate
        + economics.return_logistics_cost_gbp
        + 100 * economics.return_restocking_loss_rate
    )


def test_flagging_nothing_saves_nothing(scored, base_config):
    y, _, values, _ = scored
    result = net_benefit(y, np.zeros(len(y), dtype=bool), values, base_config.economics)
    assert result["net_saving_gbp"] == 0.0
    assert result["n_flagged"] == 0


def test_flagging_everything_pays_for_every_intervention(scored, base_config):
    y, _, values, _ = scored
    result = net_benefit(y, np.ones(len(y), dtype=bool), values, base_config.economics)
    assert result["intervention_spend_gbp"] == pytest.approx(
        len(y) * base_config.economics.intervention_cost_gbp
    )
    assert result["gross_saving_gbp"] > 0


def test_value_aware_rule_is_more_willing_on_expensive_orders(base_config):
    proba = np.array([0.4, 0.4])
    values = np.array([15.0, 900.0])
    flags = value_aware_flags(proba, values, base_config.economics)
    assert not flags[0], "a cheap order should not justify the intervention"
    assert flags[1], "the same risk on an expensive order should"


def test_chosen_threshold_maximises_its_criterion(scored, base_config):
    y, proba, values, _ = scored
    choice, sweep = choose_threshold(y, proba, values, base_config.economics, criterion="f1")
    assert choice.criterion == "f1"
    assert choice.score == pytest.approx(sweep["f1"].max())
    assert 0 < choice.threshold < 1


def test_choose_threshold_records_where_it_was_fitted(scored, base_config):
    y, proba, values, _ = scored
    choice, _ = choose_threshold(y, proba, values, base_config.economics, fitted_on="validation")
    assert choice.fitted_on == "validation"


def test_unknown_criterion_is_rejected(scored, base_config):
    y, proba, values, _ = scored
    with pytest.raises(ValueError, match="unknown criterion"):
        choose_threshold(y, proba, values, base_config.economics, criterion="vibes")


def test_threshold_sweep_is_monotone_in_flag_count(scored, base_config):
    y, proba, values, _ = scored
    sweep = threshold_sweep(y, proba, values, base_config.economics)
    assert sweep["n_flagged"].is_monotonic_decreasing


def test_policy_comparison_includes_the_no_model_baselines(scored, base_config):
    y, proba, values, _ = scored
    table = policy_comparison(y, proba, values, base_config.economics, threshold=0.5)
    assert "Do nothing" in table.index
    assert "Intervene on every order" in table.index
    assert table.loc["Do nothing", "net_saving_gbp"] == 0.0


def test_bayes_ceiling_beats_any_noisy_model(scored):
    y, proba, _, truth = scored
    ceiling = bayes_ceiling(y, truth)
    model = classification_metrics(y, proba)
    assert ceiling["roc_auc"] >= model["roc_auc"]


def test_attainment_is_a_fraction_of_available_signal(scored):
    y, proba, _, truth = scored
    ceiling = bayes_ceiling(y, truth)
    scores = attainment(classification_metrics(y, proba), ceiling)
    assert 0 < scores["roc_auc_attainment"] <= 1.05


def test_heuristic_baseline_flags_the_documented_groups(dataset):
    flags = heuristic_baseline_flags(dataset)
    expected = (
        (dataset["payment_method"] == "Cash_on_Delivery")
        | (dataset["customer_segment"] == "First_Time")
    ).to_numpy()
    assert (flags == expected).all()


def test_summarise_models_orders_headline_metrics_first():
    table = summarise_models(
        {
            "A": {"roc_auc": 0.7, "f1": 0.5, "accuracy": 0.6},
            "B": {"roc_auc": 0.8, "f1": 0.4, "accuracy": 0.7},
        }
    )
    assert list(table.columns)[0] == "roc_auc"
    assert table.loc["B", "roc_auc"] == 0.8
