"""Dataset generation and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from returns_clv.data.generate import (
    CLICK_DEPTH_RANGE,
    ORDER_VALUE_RANGE,
    OUTPUT_COLUMNS,
    generate_dataset,
    true_return_probability,
)
from returns_clv.data.validate import validate_dataset


def test_generated_schema_and_size(base_config, dataset):
    assert list(dataset.columns) == OUTPUT_COLUMNS
    assert len(dataset) == 4000
    assert dataset["transaction_id"].is_unique
    assert not dataset.isna().any().any()


def test_generation_is_reproducible(base_config):
    first = generate_dataset(base_config, n_transactions=500, seed=7)
    second = generate_dataset(base_config, n_transactions=500, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_give_different_data(base_config):
    first = generate_dataset(base_config, n_transactions=500, seed=1)
    second = generate_dataset(base_config, n_transactions=500, seed=2)
    assert not first["returned"].equals(second["returned"])


def test_feature_bounds_respected(dataset):
    assert dataset["click_depth"].between(*CLICK_DEPTH_RANGE).all()
    assert dataset["order_value_gbp"].between(*ORDER_VALUE_RANGE).all()
    assert dataset["customer_tenure_days"].ge(0).all()


def test_customer_attributes_are_internally_consistent(dataset):
    """The defect the original dataset carries must not reappear here."""
    grouped = dataset.groupby("customer_id")
    assert (grouped["customer_segment"].nunique() == 1).all()


def test_tenure_increases_with_time_within_a_customer(dataset):
    frame = dataset.copy()
    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"])
    frame = frame.sort_values(["customer_id", "transaction_date"])
    diffs = frame.groupby("customer_id")["customer_tenure_days"].diff().dropna()
    assert (diffs >= 0).all(), "tenure must never go backwards in time"


def test_order_frequency_is_computed_from_the_past_only(dataset):
    """Every customer's earliest order reports a trailing count of 1.

    Same-day orders are ordered arbitrarily among themselves, so the assertion
    is on each customer's minimum rather than on whichever row sorts first.
    """
    minimum = dataset.groupby("customer_id")["order_frequency_12m"].min()
    assert (minimum == 1).all()
    # The count can never exceed the customer's total order count.
    totals = dataset.groupby("customer_id").size()
    assert (dataset.groupby("customer_id")["order_frequency_12m"].max() <= totals).all()


def test_documented_effects_are_present(base_config, dataset):
    rates = dataset.groupby("customer_segment")["returned"].mean()
    assert rates["First_Time"] > rates["Repeat"] > rates["Wholesale"]
    by_payment = dataset.groupby("payment_method")["returned"].mean()
    assert by_payment["Cash_on_Delivery"] > by_payment["Credit_Card"]


def test_true_probability_matches_realised_rate(base_config, dataset):
    expected = true_return_probability(dataset, base_config)
    assert expected.between(0, base_config.data_generation.max_return_probability).all()
    # Law of large numbers: the realised rate tracks the generating probability.
    assert abs(expected.mean() - dataset["returned"].mean()) < 0.03


def test_true_probability_needs_its_inputs(base_config, dataset):
    with pytest.raises(KeyError, match="missing columns"):
        true_return_probability(dataset.drop(columns=["click_depth"]), base_config)


def test_validation_passes_on_generated_data(base_config, dataset):
    report = validate_dataset(dataset, base_config)
    assert report.passed, [c.detail for c in report.failures]
    assert not report.warnings, [c.detail for c in report.warnings]


def test_validation_catches_missing_values(base_config, dataset):
    broken = dataset.copy()
    broken.loc[broken.index[:5], "order_value_gbp"] = np.nan
    report = validate_dataset(broken, base_config)
    assert not report.passed
    assert "missing_values" in {c.name for c in report.failures}


def test_validation_catches_duplicate_ids(base_config, dataset):
    broken = pd.concat([dataset, dataset.head(3)], ignore_index=True)
    report = validate_dataset(broken, base_config)
    assert "duplicate_ids" in {c.name for c in report.failures}


def test_validation_catches_customer_attribute_conflicts(base_config, dataset):
    """This is exactly the defect present in the original coursework dataset."""
    broken = dataset.copy()
    target_customer = broken["customer_id"].iloc[0]
    mask = broken["customer_id"] == target_customer
    broken.loc[mask.idxmax(), "customer_segment"] = "Wholesale"
    broken.loc[broken.index[mask][-1], "customer_segment"] = "First_Time"
    report = validate_dataset(broken, base_config)
    assert "customer_invariants" in {c.name for c in report.failures}


def test_validation_catches_out_of_range_features(base_config, dataset):
    broken = dataset.copy()
    broken.loc[broken.index[0], "click_depth"] = 99
    report = validate_dataset(broken, base_config)
    assert "feature_bounds" in {c.name for c in report.failures}


def test_validation_report_serialises(base_config, dataset):
    report = validate_dataset(dataset, base_config)
    payload = report.to_dict()
    assert payload["passed"] is True
    assert payload["n_checks"] == len(report.checks)
    assert "| Check | Status | Detail |" in report.to_markdown()
    assert not report.to_frame().empty


@pytest.mark.parametrize("n", [50, 399, 400, 401, 5000])
def test_generation_hits_the_requested_size_for_any_target(base_config, n):
    """Including targets smaller than the configured customer count."""
    df = generate_dataset(base_config, n_transactions=n, seed=3)
    assert len(df) == n
    assert df["customer_id"].nunique() <= base_config.data_generation.n_customers
