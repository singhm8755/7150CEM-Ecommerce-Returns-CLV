"""Train / validation / test splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from returns_clv.config import load_config
from returns_clv.splits import cv_indices, make_split


@pytest.fixture(params=["temporal", "random", "grouped"])
def strategy(request) -> str:
    return request.param


def _config(base_config, strategy):
    return load_config(
        root=base_config.paths.raw_csv_path.parents[1], overrides={"split": {"strategy": strategy}}
    )


def test_splits_are_disjoint_and_complete(base_config, dataset, strategy):
    config = _config(base_config, strategy)
    split = make_split(dataset, config)
    sizes = len(split.train) + len(split.validation) + len(split.test)
    assert sizes == len(dataset)
    assert not set(split.train.index) & set(split.test.index)


def test_split_proportions_are_approximately_respected(base_config, dataset, strategy):
    config = _config(base_config, strategy)
    split = make_split(dataset, config)
    assert 0.10 <= len(split.test) / len(dataset) <= 0.22


def test_temporal_split_puts_the_future_in_test(base_config, dataset):
    config = _config(base_config, "temporal")
    split = make_split(dataset, config)
    train_max = pd.to_datetime(split.train["transaction_date"]).max()
    test_min = pd.to_datetime(split.test["transaction_date"]).min()
    assert train_max < test_min, "a temporal split must not train on the future"


def test_grouped_split_keeps_customers_on_one_side(base_config, dataset):
    config = _config(base_config, "grouped")
    split = make_split(dataset, config)
    assert not set(split.train["customer_id"]) & set(split.test["customer_id"])


def test_describe_reports_sizes_and_rates(base_config, dataset, strategy):
    config = _config(base_config, strategy)
    described = make_split(dataset, config).describe("returned")
    assert described["strategy"] == strategy
    assert 0 < described["train"]["positive_rate"] < 1
    assert described["train"]["share"] > described["test"]["share"]


def test_temporal_cv_folds_never_train_on_later_data(base_config, dataset):
    config = _config(base_config, "temporal")
    split = make_split(dataset, config)
    dates = pd.to_datetime(split.train["transaction_date"]).to_numpy()
    for train_idx, test_idx in cv_indices(split.train, config, 3):
        assert dates[train_idx].max() <= dates[test_idx].min()


def test_random_cv_folds_are_stratified(base_config, dataset):
    config = _config(base_config, "random")
    split = make_split(dataset, config)
    overall = split.train["returned"].mean()
    for _, test_idx in cv_indices(split.train, config, 3):
        fold_rate = split.train["returned"].to_numpy()[test_idx].mean()
        assert abs(fold_rate - overall) < 0.05
