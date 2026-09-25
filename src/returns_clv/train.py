"""Training stage: hyperparameter search, calibration, threshold choice, evaluation.

The protocol, and why each piece is where it is:

1. **Split into train / validation / test** by the configured strategy, default
   temporal.
2. **Search hyperparameters on the training block only**, with resampling inside
   the estimator pipeline so each cross-validation fold resamples its own
   training portion. Under a temporal protocol the folds are expanding windows,
   so a fold is never scored on transactions that precede its training data.
3. **Select a model on the validation block** using a threshold-free metric
   (average precision), so selection cannot be gamed by a lucky cut-off.
4. **Calibrate on one half of the validation block.** Isotonic regression maps
   raw scores onto honest probabilities, which the CLV stage then multiplies by
   pound values.
5. **Choose the operating threshold on the other half.** Calibration and
   threshold selection use disjoint data, so the threshold is not fitted to the
   calibrator's own residual error.
6. **Score the test block once**, at the end, and report it.

The test block is never used to fit anything - not a scaler, not a calibrator,
not a threshold. That single discipline is the difference between the numbers in
``outputs/reports`` and the optimistic ones the original notebooks reported.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from returns_clv import evaluate, plots
from returns_clv.artifacts import ModelBundle, categorical_levels, dataset_fingerprint, save_bundle
from returns_clv.config import Config
from returns_clv.data.generate import true_return_probability
from returns_clv.data.validate import validate_dataset
from returns_clv.features import select_model_inputs
from returns_clv.logging_utils import banner, get_logger, write_json
from returns_clv.models import DISPLAY_NAMES, build_pipeline, class_ratio, search_space
from returns_clv.splits import DataSplit, cv_indices, make_split

logger = get_logger(__name__)

# Rows kept as the SHAP reference distribution inside the saved bundle.
BACKGROUND_SAMPLE_SIZE = 300


@dataclass
class ModelResult:
    """Everything learned about one candidate model."""

    name: str
    display_name: str
    estimator: Any
    cv_score: float
    cv_params: dict[str, Any]
    validation_metrics: dict[str, float]
    fit_seconds: float
    test_metrics: dict[str, float] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        return {
            "model": self.display_name,
            "cv_score": self.cv_score,
            "fit_seconds": round(self.fit_seconds, 1),
            **{f"val_{k}": v for k, v in self.validation_metrics.items() if isinstance(v, float)},
        }


def load_dataset(config: Config) -> pd.DataFrame:
    """Read the raw transactions and run validation, logging any defects found."""
    path = config.paths.raw_csv_path
    if not path.exists():
        raise FileNotFoundError(
            f"dataset not found at {path}. Run `returns-clv generate` to create one."
        )
    df = pd.read_csv(path)
    logger.info("loaded %s transactions from %s", f"{len(df):,}", path.name)

    report = validate_dataset(df, config)
    report.log()
    if not report.passed:
        raise ValueError(
            f"dataset failed {len(report.failures)} validation check(s): "
            f"{[c.name for c in report.failures]}"
        )
    if report.warnings:
        logger.warning(
            "%d validation warning(s) - the pipeline compensates for these, see "
            "docs/methodology.md",
            len(report.warnings),
        )
    return df


def _fit_candidate(name: str, split: DataSplit, config: Config) -> ModelResult:
    """Hyperparameter-search one model on the training block, score it on validation."""
    target = config.features.target
    X_train = select_model_inputs(split.train, config)
    y_train = split.train[target].to_numpy()
    X_val = select_model_inputs(split.validation, config)
    y_val = split.validation[target].to_numpy()

    pipeline = build_pipeline(name, config, class_ratio=class_ratio(y_train))
    space = search_space(name, config)
    started = time.perf_counter()

    if space:
        folds = cv_indices(split.train, config, config.training.cv_folds)
        search = RandomizedSearchCV(
            pipeline,
            space,
            n_iter=config.training.search_iterations,
            cv=folds,
            scoring=config.training.scoring,
            random_state=config.seed,
            # The estimators themselves are parallel; keeping the search serial
            # avoids oversubscribing cores and duplicating the training data
            # once per worker.
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        estimator = search.best_estimator_
        cv_score = float(search.best_score_)
        cv_params = {k.split("__", 1)[-1]: v for k, v in search.best_params_.items()}
    else:  # the baseline has nothing to tune
        pipeline.fit(X_train, y_train)
        estimator = pipeline
        cv_score = float("nan")
        cv_params = {}

    elapsed = time.perf_counter() - started
    val_proba = estimator.predict_proba(X_val)[:, 1]
    val_metrics = evaluate.classification_metrics(y_val, val_proba, threshold=0.5)

    logger.info(
        "%-24s cv %s=%.4f | validation AP=%.4f AUC=%.4f | %.1fs",
        name,
        config.training.scoring,
        cv_score,
        val_metrics["average_precision"],
        val_metrics["roc_auc"],
        elapsed,
    )
    return ModelResult(
        name=name,
        display_name=DISPLAY_NAMES.get(name, name),
        estimator=estimator,
        cv_score=cv_score,
        cv_params=cv_params,
        validation_metrics=val_metrics,
        fit_seconds=elapsed,
    )


def _calibrate(estimator: Any, X: pd.DataFrame, y: np.ndarray, config: Config) -> Any:
    """Wrap a fitted estimator in a probability calibrator fitted on held-out data."""
    method = config.training.calibration
    if method == "none":
        logger.info("calibration disabled")
        return estimator
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    except ImportError:  # pragma: no cover - scikit-learn < 1.6
        calibrated = CalibratedClassifierCV(estimator, cv="prefit", method=method)
    calibrated.fit(X, y)
    logger.info("calibrated with %s regression on %s held-out rows", method, f"{len(X):,}")
    return calibrated


def run_training(config: Config, *, df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Execute the full training stage and write every artefact.

    Args:
        config: Pipeline configuration.
        df: Optional pre-loaded dataset, used by tests to avoid disk reads.

    Returns:
        A summary dictionary, also written to ``outputs/reports/training_summary.json``.
    """
    banner(logger, "stage 3 of 5: model training")
    config.paths.ensure_dirs()
    target = config.features.target

    df = load_dataset(config) if df is None else df
    split = make_split(df, config)

    # --- 1-2. search hyperparameters, one model at a time --------------------
    results: list[ModelResult] = []
    for name in ["dummy", *config.training.models]:
        results.append(_fit_candidate(name, split, config))

    # --- 3. select on validation, with a threshold-free metric ---------------
    metric = config.training.selection_metric
    tuned = [r for r in results if r.name != "dummy"]
    best = max(tuned, key=lambda r: r.validation_metrics[metric])
    logger.info(
        "selected %s on validation %s=%.4f",
        best.display_name,
        metric,
        best.validation_metrics[metric],
    )

    # --- 4-5. calibrate and choose a threshold on disjoint validation halves --
    calib_part, threshold_part = train_test_split(
        split.validation,
        test_size=0.5,
        random_state=config.seed,
        stratify=split.validation[target],
    )
    calibrated = _calibrate(
        best.estimator,
        select_model_inputs(calib_part, config),
        calib_part[target].to_numpy(),
        config,
    )

    threshold_proba = calibrated.predict_proba(select_model_inputs(threshold_part, config))[:, 1]
    choice, validation_sweep = evaluate.choose_threshold(
        threshold_part[target].to_numpy(),
        threshold_proba,
        threshold_part["order_value_gbp"].to_numpy(),
        config.economics,
        criterion="net_saving",
        fitted_on="validation (threshold half)",
    )

    # --- 6. score the test block, once --------------------------------------
    banner(logger, "test-set evaluation")
    X_test = select_model_inputs(split.test, config)
    y_test = split.test[target].to_numpy()
    order_values = split.test["order_value_gbp"].to_numpy()

    test_probabilities: dict[str, np.ndarray] = {}
    for result in results:
        proba = result.estimator.predict_proba(X_test)[:, 1]
        test_probabilities[result.display_name] = proba
        result.test_metrics = evaluate.classification_metrics(y_test, proba, threshold=0.5)

    calibrated_proba = calibrated.predict_proba(X_test)[:, 1]
    deployed_name = f"{best.display_name} (calibrated)"
    test_probabilities[deployed_name] = calibrated_proba
    deployed_metrics = evaluate.classification_metrics(y_test, calibrated_proba, choice.threshold)

    # How much of the learnable signal was captured.
    ceiling = evaluate.bayes_ceiling(y_test, true_return_probability(split.test, config))
    attainment = evaluate.attainment(deployed_metrics, ceiling)
    logger.info(
        "test ROC-AUC %.4f against a ceiling of %.4f (%.1f%% of available signal captured)",
        deployed_metrics["roc_auc"],
        ceiling["roc_auc"],
        100 * attainment["roc_auc_attainment"],
    )

    # --- business value ----------------------------------------------------
    policies = evaluate.policy_comparison(
        y_test,
        calibrated_proba,
        order_values,
        config.economics,
        threshold=choice.threshold,
        heuristic_flags=evaluate.heuristic_baseline_flags(split.test),
    )
    test_sweep = evaluate.threshold_sweep(y_test, calibrated_proba, order_values, config.economics)
    logger.info(
        "net saving on the test period: £%s (%s)",
        f"{policies['net_saving_gbp'].max():,.0f}",
        policies["net_saving_gbp"].idxmax(),
    )

    # --- artefacts ---------------------------------------------------------
    comparison = _comparison_table(results, deployed_name, deployed_metrics)
    reports = config.paths.reports_path
    comparison.to_csv(reports / "model_comparison.csv")
    policies.round(2).to_csv(reports / "policy_comparison.csv")
    validation_sweep.to_csv(reports / "threshold_sweep_validation.csv", index=False)
    test_sweep.to_csv(reports / "threshold_sweep_test.csv", index=False)

    _write_figures(
        config=config,
        y_test=y_test,
        test_probabilities=test_probabilities,
        deployed_name=deployed_name,
        comparison=comparison,
        ceiling=ceiling,
        test_sweep=test_sweep,
        threshold=choice.threshold,
        policies=policies,
        split=split,
        config_ceiling=ceiling,
    )

    bundle = ModelBundle(
        pipeline=calibrated,
        threshold=choice.threshold,
        model_name=best.display_name,
        feature_columns=config.features.all_features,
        categorical_levels=categorical_levels(df, config),
        threshold_criterion=choice.criterion,
        explainer_background=_explainer_background(best.estimator, split.train, config),
        metrics={"test": deployed_metrics, "ceiling": ceiling, **attainment},
        config_summary=evaluate.evaluation_config_summary(config),
        economics={
            "gross_margin_rate": config.economics.gross_margin_rate,
            "return_logistics_cost_gbp": config.economics.return_logistics_cost_gbp,
            "return_restocking_loss_rate": config.economics.return_restocking_loss_rate,
            "intervention_cost_gbp": config.economics.intervention_cost_gbp,
            "intervention_effectiveness": config.economics.intervention_effectiveness,
        },
        dataset_fingerprint=dataset_fingerprint(df),
    )
    bundle_path = save_bundle(bundle, config.paths.models_path)

    summary = {
        "dataset": {
            "path": str(config.paths.raw_csv_path),
            "n_transactions": int(len(df)),
            "n_customers": int(df["customer_id"].nunique()),
            "return_rate": float(df[target].mean()),
            "fingerprint": bundle.dataset_fingerprint,
        },
        "protocol": evaluate.evaluation_config_summary(config),
        "split": split.describe(target),
        "candidates": [
            {
                "model": r.display_name,
                "cv_score": r.cv_score,
                "best_params": r.cv_params,
                "fit_seconds": round(r.fit_seconds, 1),
                "validation": r.validation_metrics,
                "test_at_0.5": r.test_metrics,
            }
            for r in results
        ],
        "selected_model": best.display_name,
        "threshold": choice.to_dict(),
        "deployed_test_metrics": deployed_metrics,
        "achievable_ceiling": ceiling,
        "signal_attainment": attainment,
        "business_impact": policies.reset_index().to_dict(orient="records"),
        "artefacts": {
            "model_bundle": str(bundle_path),
            "reports": sorted(p.name for p in reports.glob("*.csv")),
            "figures": sorted(p.name for p in config.paths.figures_path.glob("*.png")),
        },
    }
    write_json(summary, reports / "training_summary.json")
    banner(logger, "training complete")
    return summary


def _explainer_background(estimator: Any, train: pd.DataFrame, config: Config) -> np.ndarray | None:
    """A sample of training rows in transformed space, stored for serving-time SHAP.

    Explaining one prediction requires a reference distribution to measure it
    against. At serving time only the single request row is available, so the
    reference has to travel with the model.
    """
    from returns_clv.explain import transformed_frame

    try:
        sample = train.sample(min(BACKGROUND_SAMPLE_SIZE, len(train)), random_state=config.seed)
        return transformed_frame(estimator, select_model_inputs(sample, config)).to_numpy()
    except Exception:  # pragma: no cover - never lose a trained model over this
        logger.exception("could not build the explainer background; /explain will be degraded")
        return None


def _comparison_table(
    results: list[ModelResult], deployed_name: str, deployed_metrics: dict[str, float]
) -> pd.DataFrame:
    """Test-set comparison of every candidate plus the deployed configuration."""
    rows = {r.display_name: r.test_metrics for r in results}
    rows[deployed_name] = deployed_metrics
    return evaluate.summarise_models(rows)


def _write_figures(
    *,
    config: Config,
    y_test: np.ndarray,
    test_probabilities: dict[str, np.ndarray],
    deployed_name: str,
    comparison: pd.DataFrame,
    ceiling: dict[str, float],
    test_sweep: pd.DataFrame,
    threshold: float,
    policies: pd.DataFrame,
    split: DataSplit,
    config_ceiling: dict[str, float],
) -> None:
    """Render every evaluation figure. Failures here must not lose a trained model."""
    figures = config.paths.figures_path
    # The baseline predicts a constant, so it has no informative curve.
    curve_models = {
        name: proba for name, proba in test_probabilities.items() if name != DISPLAY_NAMES["dummy"]
    }

    try:
        roc = {name: roc_curve(y_test, p)[:2] for name, p in curve_models.items()}
        truth = true_return_probability(split.test, config).to_numpy()
        ceiling_fpr, ceiling_tpr, _ = roc_curve(y_test, truth)
        plots.plot_roc_curves(
            roc,
            {n: roc_auc_score(y_test, p) for n, p in curve_models.items()},
            figures / "01_roc_curves.png",
            ceiling=(ceiling_fpr, ceiling_tpr),
            ceiling_auc=ceiling["roc_auc"],
        )

        pr = {}
        ap = {}
        for name, proba in curve_models.items():
            precision, recall, _ = precision_recall_curve(y_test, proba)
            pr[name] = (recall, precision)
            ap[name] = average_precision_score(y_test, proba)
        plots.plot_pr_curves(pr, ap, float(y_test.mean()), figures / "02_precision_recall.png")

        plots.plot_calibration(
            {name: (y_test, proba) for name, proba in curve_models.items()},
            figures / "03_calibration.png",
            brier={name: brier_score_loss(y_test, proba) for name, proba in curve_models.items()},
        )

        matrices = {}
        for name, proba in curve_models.items():
            cut = threshold if name == deployed_name else 0.5
            metrics = evaluate.classification_metrics(y_test, proba, cut)
            matrices[name] = np.array(
                [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
            )
        plots.plot_confusion_matrices(matrices, figures / "04_confusion_matrices.png")

        plots.plot_threshold_sweep(test_sweep, threshold, figures / "05_threshold_sweep.png")
        plots.plot_model_comparison(
            comparison, figures / "06_model_comparison.png", ceiling=config_ceiling
        )
        plots.plot_policy_impact(policies, figures / "07_policy_impact.png")
    except Exception:  # pragma: no cover - a plotting failure must not lose the model
        logger.exception("figure generation failed; model artefacts are still written")


def resolve_report_path(config: Config, name: str) -> Path:
    """Convenience for other stages that read this stage's reports."""
    return config.paths.reports_path / name
