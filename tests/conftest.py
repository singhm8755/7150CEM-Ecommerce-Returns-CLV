"""Shared fixtures.

Tests run against a small generated dataset rather than the committed CSV, so
the suite is fast, deterministic and independent of whether the pipeline has
been run. Session-scoped fixtures do the expensive work once.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from returns_clv.config import Config, load_config
from returns_clv.data.generate import generate_dataset

SMALL_DATASET_ROWS = 4000


@pytest.fixture(scope="session")
def base_config(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """Config with a fast search budget and artefact paths inside a temp dir."""
    root = tmp_path_factory.mktemp("project")
    return load_config(
        root=root,
        overrides={
            "data_generation": {"n_customers": 400, "n_transactions": SMALL_DATASET_ROWS},
            "training": {
                "models": ["logistic_regression"],
                "cv_folds": 2,
                "search_iterations": 2,
            },
            "explainability": {"shap_sample_size": 200},
        },
    )


@pytest.fixture(scope="session")
def dataset(base_config: Config) -> pd.DataFrame:
    """A small, internally consistent synthetic dataset."""
    return generate_dataset(base_config, n_transactions=SMALL_DATASET_ROWS)


@pytest.fixture(scope="session")
def dataset_on_disk(base_config: Config, dataset: pd.DataFrame) -> Path:
    """The dataset written where the config expects to find it."""
    path = base_config.paths.raw_csv_path
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def trained_summary(base_config: Config, dataset: pd.DataFrame, dataset_on_disk: Path) -> dict:
    """Run training once for every test that needs a fitted model on disk."""
    from returns_clv.train import run_training

    return run_training(base_config, df=dataset)


@pytest.fixture(scope="session")
def bundle(base_config: Config, trained_summary: dict):
    from returns_clv.artifacts import load_bundle

    return load_bundle(base_config.paths.models_path)


@pytest.fixture
def sample_transaction() -> dict:
    return {
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
