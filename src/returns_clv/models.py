"""Model zoo: estimator pipelines and their hyperparameter search spaces.

Every candidate is returned as a single ``imblearn.pipeline.Pipeline``:

    feature engineering -> preprocessing -> resampling -> classifier

Bundling resampling into the estimator is the central methodological fix in this
project. ``imblearn``'s pipeline applies resamplers on ``fit`` only, never on
``transform``/``predict``, so during cross-validation SMOTE sees the training
folds and nothing else. The coursework notebooks resampled the whole training
set *before* ``GridSearchCV``, which let synthetic minority points derived from a
validation fold's neighbours end up in the fold used to score it - inflating
cross-validated scores while leaving test scores untouched. See
``docs/methodology.md`` for the measured size of that gap.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from returns_clv.config import Config
from returns_clv.features import feature_steps
from returns_clv.logging_utils import get_logger

logger = get_logger(__name__)

MODEL_STEP = "classifier"
RESAMPLE_STEP = "resample"

SUPPORTED_MODELS = ("logistic_regression", "random_forest", "xgboost", "dummy")

# Human-readable names used in reports and figures.
DISPLAY_NAMES = {
    "dummy": "Majority-class baseline",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def build_resampler(config: Config):
    """Instantiate the configured resampler, or None when resampling is disabled."""
    kind = config.training.resampler
    if kind == "none":
        return None
    if kind == "smote":
        return SMOTE(random_state=config.seed, k_neighbors=5)
    if kind == "adasyn":
        return ADASYN(random_state=config.seed, n_neighbors=5)
    raise ValueError(f"unknown resampler {kind!r}")  # pragma: no cover


def _classifier(name: str, config: Config, class_ratio: float):
    """Create the bare classifier.

    When resampling is disabled the classifier carries the class-imbalance
    correction itself, so the two mechanisms are never applied at once.
    """
    reweight = config.training.resampler == "none"

    if name == "dummy":
        return DummyClassifier(strategy="prior", random_state=config.seed)
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            random_state=config.seed,
            class_weight="balanced" if reweight else None,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            random_state=config.seed,
            n_jobs=config.training.n_jobs,
            class_weight="balanced" if reweight else None,
        )
    if name == "xgboost":
        return XGBClassifier(
            random_state=config.seed,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=config.training.n_jobs,
            scale_pos_weight=class_ratio if reweight else 1.0,
        )
    raise ValueError(f"unknown model {name!r}; expected one of {SUPPORTED_MODELS}")


def search_space(name: str, config: Config) -> dict[str, Any]:
    """Hyperparameter distributions for ``RandomizedSearchCV``.

    Keys are prefixed with the pipeline step name so the search tunes the
    classifier inside the full preprocessing pipeline.
    """
    seed = config.seed
    spaces: dict[str, dict[str, Any]] = {
        "dummy": {},
        "logistic_regression": {
            # `penalty` is deliberately left at its default: scikit-learn >= 1.8
            # expresses the L1/L2 mix through `l1_ratio` instead.
            f"{MODEL_STEP}__C": loguniform(1e-3, 1e2),
            f"{MODEL_STEP}__solver": ["lbfgs"],
        },
        "random_forest": {
            f"{MODEL_STEP}__n_estimators": randint(150, 401),
            f"{MODEL_STEP}__max_depth": [8, 12, 16, 24, None],
            f"{MODEL_STEP}__min_samples_leaf": randint(1, 40),
            f"{MODEL_STEP}__max_features": ["sqrt", "log2", 0.5],
        },
        "xgboost": {
            f"{MODEL_STEP}__n_estimators": randint(200, 601),
            f"{MODEL_STEP}__max_depth": randint(3, 9),
            f"{MODEL_STEP}__learning_rate": loguniform(0.01, 0.3),
            f"{MODEL_STEP}__subsample": uniform(0.6, 0.4),
            f"{MODEL_STEP}__colsample_bytree": uniform(0.6, 0.4),
            f"{MODEL_STEP}__min_child_weight": randint(1, 20),
            f"{MODEL_STEP}__reg_lambda": loguniform(1e-2, 1e2),
        },
    }
    if name not in spaces:
        raise ValueError(f"unknown model {name!r}; expected one of {SUPPORTED_MODELS}")
    space = spaces[name]
    # Give every distribution the project seed so searches are reproducible.
    for value in space.values():
        if hasattr(value, "random_state"):
            value.random_state = np.random.default_rng(seed).integers(0, 2**31 - 1)
    return space


def build_pipeline(
    name: str,
    config: Config,
    *,
    class_ratio: float = 1.0,
    engineered: bool = True,
    resample: bool = True,
) -> ImbPipeline:
    """Assemble a complete, fittable estimator for ``name``.

    Args:
        name: One of :data:`SUPPORTED_MODELS`.
        config: Pipeline configuration.
        class_ratio: ``n_negative / n_positive`` in the training data, used for
            ``scale_pos_weight`` when resampling is disabled.
        engineered: Include the derived behavioural features.
        resample: Include the resampling step. The baseline model skips it.

    Returns:
        An ``imblearn`` pipeline accepting raw feature columns as input.
    """
    steps: list[tuple[str, Any]] = list(feature_steps(config, engineered=engineered))

    resampler = build_resampler(config) if (resample and name != "dummy") else None
    if resampler is not None:
        steps.append((RESAMPLE_STEP, resampler))

    steps.append((MODEL_STEP, _classifier(name, config, class_ratio)))
    return ImbPipeline(steps)


def class_ratio(y) -> float:
    """``n_negative / n_positive``, the standard ``scale_pos_weight`` value."""
    positives = int(np.sum(y))
    if positives == 0:  # pragma: no cover
        raise ValueError("target contains no positive class")
    return float((len(y) - positives) / positives)
