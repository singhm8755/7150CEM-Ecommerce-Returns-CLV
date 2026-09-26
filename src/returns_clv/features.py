"""Feature engineering and preprocessing.

Two design decisions matter here.

**One-hot, not label encoding.** The coursework notebooks used
``LabelEncoder`` on nominal columns, which tells a model that
``Credit_Card < PayPal < Cash_on_Delivery``. Tree ensembles can partially route
around that, but linear models cannot. Every categorical column is one-hot
encoded instead.

**Every transformation lives inside the estimator.** Preprocessing is a
scikit-learn ``Pipeline`` step, never a step applied to the whole frame before
splitting. That is what keeps scaler statistics from leaking across the
train/test boundary and what makes the saved artefact deployable: the API hands
raw JSON to ``predict_proba`` and the pipeline does the rest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from returns_clv.config import Config
from returns_clv.logging_utils import get_logger

logger = get_logger(__name__)

# Thresholds used by the derived flags. They mirror the browsing-behaviour
# effects documented in the DGP and confirmed by the EDA notebook.
LOW_CLICK_DEPTH_THRESHOLD = 3
SHORT_DWELL_SECONDS = 60

ENGINEERED_NUMERIC = [
    "dwell_seconds_per_click",
    "visits_per_click",
    "log_order_value",
    "value_per_page_visit",
]
ENGINEERED_BINARY = ["is_low_click_depth", "is_short_dwell"]
ENGINEERED_FEATURES = [*ENGINEERED_NUMERIC, *ENGINEERED_BINARY]

_REQUIRED_FOR_ENGINEERING = [
    "click_depth",
    "time_on_page_seconds",
    "product_page_visits",
    "order_value_gbp",
]


class BehaviouralFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add row-wise browsing-intensity features.

    The transformer is stateless: every output depends only on the row it is
    computed from, so it cannot leak information between train and test folds.
    It still implements ``fit`` so it composes inside a scikit-learn pipeline and
    validates its input columns once, up front.

    Args:
        enabled: When False the transformer is a pass-through. Used by the
            ablation experiment that measures what the derived features add.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y=None) -> BehaviouralFeatureEngineer:  # noqa: ARG002, N803
        if self.enabled:
            missing = sorted(set(_REQUIRED_FOR_ENGINEERING) - set(X.columns))
            if missing:
                raise ValueError(f"cannot engineer features, missing columns: {missing}")
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        if not self.enabled:
            return X.copy()

        out = X.copy()
        clicks = out["click_depth"].clip(lower=1)
        visits = out["product_page_visits"].clip(lower=1)

        # Dwell time per click separates careful shoppers from rushed ones better
        # than either raw column on its own.
        out["dwell_seconds_per_click"] = out["time_on_page_seconds"] / clicks
        out["visits_per_click"] = visits / clicks
        # Order value is log-normal by construction; the log makes it linear-friendly.
        out["log_order_value"] = np.log1p(out["order_value_gbp"])
        out["value_per_page_visit"] = out["order_value_gbp"] / visits
        out["is_low_click_depth"] = (out["click_depth"] <= LOW_CLICK_DEPTH_THRESHOLD).astype(int)
        out["is_short_dwell"] = (out["time_on_page_seconds"] < SHORT_DWELL_SECONDS).astype(int)
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        base = list(input_features if input_features is not None else self.feature_names_in_)
        return np.asarray(base if not self.enabled else [*base, *ENGINEERED_FEATURES], dtype=object)


def build_preprocessor(config: Config, *, engineered: bool = True) -> ColumnTransformer:
    """Assemble the column-wise preprocessing step.

    Args:
        config: Supplies the categorical and numeric column lists.
        engineered: Whether the derived columns from
            :class:`BehaviouralFeatureEngineer` are expected downstream.

    Returns:
        A ``ColumnTransformer`` that one-hot encodes categoricals, standardises
        continuous features and passes binary flags through untouched.
    """
    numeric = list(config.features.numeric)
    if engineered:
        numeric += ENGINEERED_NUMERIC

    transformers = [
        (
            "categorical",
            # `handle_unknown="ignore"` keeps the deployed model from crashing on
            # a category value that did not exist at training time.
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64),
            list(config.features.categorical),
        ),
        ("numeric", StandardScaler(), numeric),
    ]
    if engineered:
        transformers.append(("binary", "passthrough", ENGINEERED_BINARY))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def feature_steps(config: Config, *, engineered: bool = True) -> list[tuple[str, object]]:
    """Feature steps as a flat list, ready to splice into a larger pipeline.

    ``imblearn`` rejects nested pipelines as intermediate steps, so estimator
    construction splices these in rather than nesting
    :func:`build_feature_pipeline`.
    """
    return [
        ("engineer", BehaviouralFeatureEngineer(enabled=engineered)),
        ("preprocess", build_preprocessor(config, engineered=engineered)),
    ]


def build_feature_pipeline(config: Config, *, engineered: bool = True) -> Pipeline:
    """Feature engineering followed by preprocessing, as one fittable step."""
    return Pipeline(feature_steps(config, engineered=engineered))


def select_model_inputs(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Return only the raw columns the model consumes, in a stable order.

    Keeping this in one place means the training code, the CLV scorer and the API
    all present columns to the pipeline identically.
    """
    missing = sorted(set(config.features.all_features) - set(df.columns))
    if missing:
        raise KeyError(f"input is missing required feature columns: {missing}")
    return df.loc[:, config.features.all_features].copy()


def feature_names(fitted: Pipeline | ColumnTransformer) -> list[str]:
    """Human-readable output feature names from a fitted pipeline or transformer."""
    return [str(name) for name in fitted.get_feature_names_out()]
