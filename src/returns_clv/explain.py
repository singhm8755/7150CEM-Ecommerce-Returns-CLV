"""Model explainability with SHAP.

A return-risk score that a category manager cannot interrogate will not get
used, and a flagged customer is entitled to a reason. SHAP values decompose each
prediction into per-feature contributions that sum to the prediction itself,
which supports both the global view ("what does this model rely on?") and the
per-transaction view ("why was this order flagged?").

The explainer is applied to the *transformed* feature space, so contributions are
reported against one-hot columns and engineered features by name rather than
against opaque matrix positions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from returns_clv import plots
from returns_clv.artifacts import ModelBundle, load_bundle
from returns_clv.config import Config
from returns_clv.features import select_model_inputs
from returns_clv.logging_utils import banner, get_logger, write_json

logger = get_logger(__name__)


def unwrap_estimator(pipeline: Any) -> Any:
    """Strip calibration wrappers to reach the fitted modelling pipeline."""
    inner = getattr(pipeline, "calibrated_classifiers_", None)
    if inner:
        estimator = inner[0].estimator
        return getattr(estimator, "estimator", estimator)
    inner = getattr(pipeline, "estimator", None)
    if inner is not None:
        return getattr(inner, "estimator", inner)
    return pipeline


def transformed_frame(estimator: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Run ``X`` through every step before the classifier, keeping column names."""
    engineer = estimator.named_steps["engineer"]
    preprocess = estimator.named_steps["preprocess"]
    matrix = preprocess.transform(engineer.transform(X))
    names = [str(n) for n in preprocess.get_feature_names_out()]
    return pd.DataFrame(matrix, columns=names, index=X.index)


def shap_values(
    estimator: Any, frame: pd.DataFrame, background: pd.DataFrame | None = None
) -> np.ndarray:
    """SHAP values for the positive class, as a (n_rows, n_features) array.

    Args:
        estimator: The fitted modelling pipeline.
        frame: Rows to explain, already in transformed feature space.
        background: Reference distribution the explanation is measured against.
            Required for non-tree models when ``frame`` holds a single row: an
            explainer handed only that row compares it to itself and returns
            zeros for everything. Tree explainers read the reference off the
            tree structure and ignore this.
    """
    import shap

    classifier = estimator.named_steps["classifier"]
    model_type = type(classifier).__name__
    reference = background if background is not None else frame

    if model_type in {"RandomForestClassifier", "XGBClassifier", "GradientBoostingClassifier"}:
        explainer = shap.TreeExplainer(classifier)
        values = explainer.shap_values(frame, check_additivity=False)
    elif model_type == "LogisticRegression":
        explainer = shap.LinearExplainer(classifier, reference)
        values = explainer.shap_values(frame)
    else:  # pragma: no cover - fallback for a model without a fast explainer
        sampled = shap.sample(reference, min(100, len(reference)), random_state=0)
        explainer = shap.KernelExplainer(lambda d: classifier.predict_proba(d)[:, 1], sampled)
        values = explainer.shap_values(frame, nsamples=100)

    values = np.asarray(values)
    if values.ndim == 3:
        # (rows, features, classes) for multi-output tree explainers: keep class 1.
        values = values[..., 1]
    return values


def feature_direction(frame: pd.DataFrame, values: np.ndarray) -> pd.Series:
    """Which way each feature pushes, as the correlation of value with contribution.

    Averaging signed SHAP values across a population answers the wrong question
    for a one-hot column. ``customer_segment_First_Time`` is 0 for most rows, and
    those rows contribute negatively relative to the baseline, so the mean comes
    out negative even though being a first-time buyer plainly raises return risk.

    Correlating each feature's *value* with its *contribution* asks the question
    that was meant: does a higher value push the prediction towards a return?
    That works identically for indicators and for continuous features.
    """
    directions = {}
    for position, column in enumerate(frame.columns):
        feature = frame[column].to_numpy(dtype=float)
        contribution = values[:, position]
        if feature.std() < 1e-12 or contribution.std() < 1e-12:
            directions[column] = 0.0
        else:
            directions[column] = float(np.corrcoef(feature, contribution)[0, 1])
    return pd.Series(directions, name="direction")


def _label(direction: float, tolerance: float = 0.05) -> str:
    """Turn a direction score into a word, without over-reading a weak signal."""
    if abs(direction) < tolerance:
        return "mixed"
    return "return" if direction > 0 else "keep"


def run_explain(config: Config, *, df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Compute and report global feature attributions for the deployed model.

    Returns:
        Summary dictionary, also written to ``outputs/reports/explainability.json``.
    """
    banner(logger, "stage 5 of 5: explainability")
    config.paths.ensure_dirs()
    from returns_clv.train import load_dataset  # local import avoids a cycle

    df = load_dataset(config) if df is None else df
    bundle: ModelBundle = load_bundle(config.paths.models_path)
    estimator = unwrap_estimator(bundle.pipeline)

    sample_size = min(config.explainability.shap_sample_size, len(df))
    sample = df.sample(sample_size, random_state=config.seed)
    frame = transformed_frame(estimator, select_model_inputs(sample, config))

    logger.info("computing SHAP values for %s transactions", f"{sample_size:,}")
    values = shap_values(estimator, frame)

    importance = pd.Series(np.abs(values).mean(axis=0), index=frame.columns, name="mean_abs_shap")
    importance = importance.sort_values(ascending=False)
    direction = feature_direction(frame, values)

    figures = config.paths.figures_path
    reports = config.paths.reports_path
    plots.plot_feature_importance(importance, figures / "11_feature_importance.png")

    table = pd.concat([importance, direction], axis=1)
    table["pushes_towards"] = [_label(value) for value in table["direction"]]
    table.round(6).to_csv(reports / "feature_importance.csv")

    top = importance.head(10)
    summary: dict[str, Any] = {
        "model": bundle.model_name,
        "sample_size": int(sample_size),
        "n_features": int(len(importance)),
        "top_features": [
            {
                "feature": name,
                "mean_abs_shap": float(importance[name]),
                "direction": float(direction[name]),
                "pushes_towards": _label(direction[name]),
            }
            for name in top.index
        ],
    }
    write_json(summary, reports / "explainability.json")

    logger.info("top drivers: %s", ", ".join(top.index[:5]))
    return summary


def explain_single(
    bundle: ModelBundle, record: dict[str, Any], *, top_n: int = 5
) -> list[dict[str, Any]]:  # noqa: D401
    """Per-transaction explanation, used by the API's ``/explain`` endpoint.

    Args:
        bundle: Deployed model bundle.
        record: One raw transaction as a mapping of feature name to value.
        top_n: How many contributing features to return.

    Returns:
        The largest contributions, each with its signed SHAP value.
    """
    estimator = unwrap_estimator(bundle.pipeline)
    frame = transformed_frame(estimator, bundle.prepare([record]))

    background = getattr(bundle, "explainer_background", None)
    if background is not None:
        background = pd.DataFrame(background, columns=frame.columns)
    values = shap_values(estimator, frame, background)[0]

    contributions = pd.Series(values, index=frame.columns)
    ranked = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    return [
        {
            "feature": str(name),
            "contribution": float(value),
            "direction": _direction(float(value)),
        }
        for name, value in ranked.head(top_n).items()
    ]


def _direction(value: float, tolerance: float = 1e-9) -> str:
    """Describe a contribution, without calling a rounding artefact a signal."""
    if abs(value) < tolerance:
        return "no material effect"
    return "increases return risk" if value > 0 else "reduces return risk"
