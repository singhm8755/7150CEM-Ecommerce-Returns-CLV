"""Feature engineering and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from returns_clv.features import (
    ENGINEERED_FEATURES,
    BehaviouralFeatureEngineer,
    build_feature_pipeline,
    feature_names,
    select_model_inputs,
)


def test_select_model_inputs_returns_declared_columns_in_order(base_config, dataset):
    X = select_model_inputs(dataset, base_config)
    assert list(X.columns) == base_config.features.all_features


def test_select_model_inputs_rejects_missing_columns(base_config, dataset):
    with pytest.raises(KeyError, match="missing required feature columns"):
        select_model_inputs(dataset.drop(columns=["click_depth"]), base_config)


def test_engineer_adds_every_declared_feature(base_config, dataset):
    engineer = BehaviouralFeatureEngineer()
    out = engineer.fit_transform(select_model_inputs(dataset, base_config))
    assert set(ENGINEERED_FEATURES) <= set(out.columns)
    assert np.isfinite(out[ENGINEERED_FEATURES].to_numpy()).all()


def test_engineer_is_stateless_so_it_cannot_leak(base_config, dataset):
    """Transforming a subset must give the same values as transforming the whole."""
    X = select_model_inputs(dataset, base_config)
    engineer = BehaviouralFeatureEngineer().fit(X)
    whole = engineer.transform(X).loc[X.index[:50], ENGINEERED_FEATURES]
    subset = engineer.transform(X.iloc[:50])[ENGINEERED_FEATURES]
    pd.testing.assert_frame_equal(whole, subset)


def test_engineer_can_be_disabled(base_config, dataset):
    X = select_model_inputs(dataset, base_config)
    out = BehaviouralFeatureEngineer(enabled=False).fit_transform(X)
    pd.testing.assert_frame_equal(out, X)


def test_engineer_validates_input_columns(base_config, dataset):
    X = select_model_inputs(dataset, base_config).drop(columns=["click_depth"])
    with pytest.raises(ValueError, match="missing columns"):
        BehaviouralFeatureEngineer().fit(X)


def test_pipeline_one_hot_encodes_rather_than_label_encodes(base_config, dataset):
    pipeline = build_feature_pipeline(base_config)
    pipeline.fit(select_model_inputs(dataset, base_config))
    names = feature_names(pipeline)
    # Three levels per categorical column, each becoming its own indicator.
    assert "payment_method_Cash_on_Delivery" in names
    assert "payment_method" not in names


def test_pipeline_output_width_matches_names(base_config, dataset):
    X = select_model_inputs(dataset, base_config)
    pipeline = build_feature_pipeline(base_config)
    matrix = pipeline.fit_transform(X)
    assert matrix.shape == (len(X), len(feature_names(pipeline)))


def test_unseen_category_does_not_crash_serving(base_config, dataset):
    """`handle_unknown='ignore'` keeps a deployed model alive on new values."""
    X = select_model_inputs(dataset, base_config)
    pipeline = build_feature_pipeline(base_config).fit(X)
    novel = X.head(1).copy()
    novel.loc[novel.index[0], "product_category"] = "Groceries"
    assert pipeline.transform(novel).shape[1] == len(feature_names(pipeline))


def test_scaler_is_fitted_not_applied_globally(base_config, dataset):
    """Standardisation must come from the fitted data, not from the transform input."""
    X = select_model_inputs(dataset, base_config)
    pipeline = build_feature_pipeline(base_config).fit(X.iloc[:200])
    transformed = pipeline.transform(X.iloc[200:400])
    names = feature_names(pipeline)
    column = transformed[:, names.index("order_value_gbp")]
    # If the scaler refitted on the new data the mean would be ~0 by construction.
    assert abs(column.mean()) > 1e-9
