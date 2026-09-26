"""Experiments that quantify methodological choices.

Each function answers a question with a number rather than an assertion, and each
corresponds to a decision made elsewhere in the pipeline:

* :func:`leakage_experiment` - how much does resampling before cross-validation
  inflate reported scores?
* :func:`threshold_optimism_experiment` - how much does tuning the decision
  threshold on the test set flatter the result?
* :func:`feature_ablation_experiment` - do the engineered behavioural features
  earn their place?

These are what turn "I followed best practice" into "here is what ignoring it
would have cost."
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from returns_clv import evaluate
from returns_clv.config import Config
from returns_clv.features import build_feature_pipeline, select_model_inputs
from returns_clv.logging_utils import banner, get_logger, write_json
from returns_clv.models import build_pipeline, build_resampler, class_ratio
from returns_clv.splits import cv_indices, make_split

logger = get_logger(__name__)

DEFAULT_MODEL = "random_forest"


def leakage_experiment(
    config: Config, df: pd.DataFrame, *, model: str = DEFAULT_MODEL
) -> dict[str, Any]:
    """Measure the optimism from resampling before cross-validation.

    Two protocols, same model, same data, same folds:

    *Correct*  - SMOTE lives inside the pipeline, so each fold resamples only its
    own training portion.

    *Leaky*    - the whole training block is preprocessed and resampled once, and
    cross-validation then runs over the resampled matrix. Synthetic minority
    points interpolated between a held-out row and its neighbours end up in the
    fold used to score that row, so the model is partly scored on echoes of the
    data it trained on.

    The number that matters is the *optimism gap*: cross-validated score minus
    the score on genuinely unseen test data. A trustworthy protocol has a small
    gap; the leaky one does not.
    """
    target = config.features.target
    split = make_split(df, config)
    X_train = select_model_inputs(split.train, config)
    y_train = split.train[target].to_numpy()
    X_test = select_model_inputs(split.test, config)
    y_test = split.test[target].to_numpy()
    folds = cv_indices(split.train, config, config.training.cv_folds)
    scoring = config.training.scoring

    # --- correct protocol ---------------------------------------------------
    correct = build_pipeline(model, config, class_ratio=class_ratio(y_train))
    correct_cv = float(
        np.mean(cross_val_score(correct, X_train, y_train, cv=folds, scoring=scoring, n_jobs=1))
    )
    correct.fit(X_train, y_train)
    correct_test = evaluate.classification_metrics(y_test, correct.predict_proba(X_test)[:, 1])[
        scoring_key(scoring)
    ]

    # --- leaky protocol -----------------------------------------------------
    features = build_feature_pipeline(config)
    matrix = features.fit_transform(X_train)
    resampler = build_resampler(config)
    resampled_X, resampled_y = resampler.fit_resample(matrix, y_train)
    logger.info(
        "leaky protocol resampled %s training rows up to %s before splitting folds",
        f"{len(y_train):,}",
        f"{len(resampled_y):,}",
    )

    bare = clone(
        build_pipeline(model, config, class_ratio=class_ratio(y_train)).named_steps["classifier"]
    )
    # Folds must be regenerated: the resampled matrix has a different length and
    # no meaningful time order, which is itself part of what makes this wrong.
    leaky_cv = float(
        np.mean(
            cross_val_score(
                bare, resampled_X, resampled_y, cv=config.training.cv_folds, scoring=scoring
            )
        )
    )
    bare.fit(resampled_X, resampled_y)
    leaky_test = evaluate.classification_metrics(
        y_test, bare.predict_proba(features.transform(X_test))[:, 1]
    )[scoring_key(scoring)]

    result = {
        "model": model,
        "metric": scoring,
        "correct": {
            "cv_score": correct_cv,
            "test_score": correct_test,
            "optimism_gap": correct_cv - correct_test,
        },
        "leaky": {
            "cv_score": leaky_cv,
            "test_score": leaky_test,
            "optimism_gap": leaky_cv - leaky_test,
        },
        "inflation_in_reported_cv": leaky_cv - correct_cv,
    }
    logger.info(
        "resample-inside-CV: cv %.4f vs test %.4f (gap %+.4f)",
        correct_cv,
        correct_test,
        result["correct"]["optimism_gap"],
    )
    logger.info(
        "resample-before-CV: cv %.4f vs test %.4f (gap %+.4f)",
        leaky_cv,
        leaky_test,
        result["leaky"]["optimism_gap"],
    )
    return result


def scoring_key(scoring: str) -> str:
    """Map a scikit-learn scoring name onto our metric dictionary key."""
    return {"average_precision": "average_precision", "roc_auc": "roc_auc", "f1": "f1"}.get(
        scoring, "average_precision"
    )


def threshold_optimism_experiment(
    config: Config, df: pd.DataFrame, *, model: str = DEFAULT_MODEL
) -> dict[str, Any]:
    """Compare a threshold tuned on validation against one tuned on test.

    Tuning on test reports the best F1 that any cut-off could have achieved on
    that exact sample - a number nobody could have obtained in advance.
    """
    target = config.features.target
    split = make_split(df, config)
    estimator = build_pipeline(
        model, config, class_ratio=class_ratio(split.train[target].to_numpy())
    )
    estimator.fit(select_model_inputs(split.train, config), split.train[target].to_numpy())

    val_proba = estimator.predict_proba(select_model_inputs(split.validation, config))[:, 1]
    test_proba = estimator.predict_proba(select_model_inputs(split.test, config))[:, 1]
    y_val = split.validation[target].to_numpy()
    y_test = split.test[target].to_numpy()

    honest_choice, _ = evaluate.choose_threshold(
        y_val,
        val_proba,
        split.validation["order_value_gbp"].to_numpy(),
        config.economics,
        criterion="f1",
        fitted_on="validation",
    )
    peeking_choice, _ = evaluate.choose_threshold(
        y_test,
        test_proba,
        split.test["order_value_gbp"].to_numpy(),
        config.economics,
        criterion="f1",
        fitted_on="test (this is the mistake)",
    )

    honest_f1 = evaluate.classification_metrics(y_test, test_proba, honest_choice.threshold)["f1"]
    peeking_f1 = evaluate.classification_metrics(y_test, test_proba, peeking_choice.threshold)["f1"]
    return {
        "model": model,
        "threshold_from_validation": honest_choice.threshold,
        "threshold_from_test": peeking_choice.threshold,
        "reported_f1_honest": honest_f1,
        "reported_f1_tuned_on_test": peeking_f1,
        "overstatement": peeking_f1 - honest_f1,
    }


def feature_ablation_experiment(
    config: Config, df: pd.DataFrame, *, models: list[str] | None = None
) -> dict[str, Any]:
    """Do the engineered behavioural features earn their place?

    Run across every candidate model rather than one, because the answer depends
    on the model class: a tree ensemble can discover a threshold effect such as
    ``click_depth <= 3`` by splitting, so handing it the indicator changes
    little. A linear model cannot, and has to be given the non-linearity
    explicitly. Reporting only the tree result would hide that.
    """
    target = config.features.target
    split = make_split(df, config)
    X_train = select_model_inputs(split.train, config)
    y_train = split.train[target].to_numpy()
    X_test = select_model_inputs(split.test, config)
    y_test = split.test[target].to_numpy()

    results: dict[str, Any] = {}
    for model in models or list(config.training.models):
        scores = {}
        for label, engineered in (("raw_features_only", False), ("with_engineered_features", True)):
            pipeline: ImbPipeline = build_pipeline(
                model, config, class_ratio=class_ratio(y_train), engineered=engineered
            )
            pipeline.fit(X_train, y_train)
            metrics = evaluate.classification_metrics(y_test, pipeline.predict_proba(X_test)[:, 1])
            scores[label] = {
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
            }
        delta = (
            scores["with_engineered_features"]["average_precision"]
            - scores["raw_features_only"]["average_precision"]
        )
        results[model] = {**scores, "average_precision_delta": delta}
        logger.info("%-20s engineered features change average precision by %+.4f", model, delta)
    return results


def run_experiments(config: Config, *, df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Run every methodology experiment and write the results."""
    banner(logger, "methodology experiments")
    config.paths.ensure_dirs()
    from returns_clv.train import load_dataset  # local import avoids a cycle

    df = load_dataset(config) if df is None else df
    results = {
        "resampling_leakage": leakage_experiment(config, df),
        "threshold_optimism": threshold_optimism_experiment(config, df),
        "feature_ablation": feature_ablation_experiment(config, df),
    }
    write_json(results, config.paths.reports_path / "methodology_experiments.json")
    return results
