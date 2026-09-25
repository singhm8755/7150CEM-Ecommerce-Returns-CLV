"""Customer lifetime value and policy simulation.

Two corrections to the coursework version drive the numbers here.

**Every transaction is scored out-of-sample.** The original notebook loaded the
trained model and scored the entire dataset, including the rows the model was
fitted on, so the "predicted" return probabilities were partly memorised. Here
the training block is scored by cross-validation (each fold predicted by a model
that never saw it) and the validation and test blocks are scored by the deployed
model, which never saw them either. No customer's CLV rests on an in-sample
prediction.

**Purchase frequency is measured, not taken on trust.** Validation flags
``order_frequency_12m`` in the supplied dataset as inconsistent with the
transaction history it accompanies - first-time buyers carry a stated frequency
of 1 while having roughly ten orders on file. Since CLV multiplies by frequency,
using that column would misstate the answer for most of the customer base. The
frequency used here is each customer's observed order count annualised over
their active period.

The CLV formula is a standard discounted contribution model:

    CLV = sum over t of  annual_contribution x retention^(t-1) / (1 + d)^t

where the annual contribution nets expected return losses off gross margin.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from returns_clv import plots
from returns_clv.artifacts import ModelBundle, load_bundle
from returns_clv.config import ClvConfig, Config, EconomicsConfig
from returns_clv.features import select_model_inputs
from returns_clv.logging_utils import banner, get_logger, write_json
from returns_clv.splits import cv_indices, make_split

logger = get_logger(__name__)

DAYS_PER_YEAR = 365.25
MIN_ACTIVE_YEARS = 0.25  # floor so a customer seen once does not get a huge implied rate


def out_of_sample_scores(df: pd.DataFrame, bundle: ModelBundle, config: Config) -> pd.Series:
    """Predicted return probability for every transaction, none of it in-sample.

    Training-block rows are predicted by cross-validation; validation and test
    rows by the deployed model, which was fitted on the training block only.

    Args:
        df: Full transaction dataset.
        bundle: The deployed model bundle.
        config: Pipeline configuration.

    Returns:
        Probability of return per row, indexed like ``df``.
    """
    split = make_split(df, config)
    scores = pd.Series(np.nan, index=df.index, name="predicted_return_prob")

    # Validation and test rows were never fitted on: the deployed model is
    # already out-of-sample for them.
    holdout = pd.concat([split.validation, split.test])
    scores.loc[holdout.index] = bundle.predict_proba(holdout)

    # Training rows need cross-validated predictions to stay honest.
    train = split.train
    logger.info("cross-validating %s training rows for out-of-fold scores", f"{len(train):,}")
    estimator = _unwrap_calibrated(bundle.pipeline)
    folds = cv_indices(train, config, config.training.cv_folds)
    X_train = select_model_inputs(train, config)
    y_train = train[config.features.target].to_numpy()

    # `cross_val_predict` refuses non-partitioning folds, and expanding-window
    # folds deliberately do not cover the earliest rows, so the loop is written
    # out: each fold is predicted by a model fitted only on what precedes it.
    predicted = np.zeros(len(train), dtype=bool)
    for fold, (fit_idx, predict_idx) in enumerate(folds, start=1):
        model = _fit_calibrated(estimator, X_train, y_train, fit_idx, config)
        scores.loc[train.index[predict_idx]] = model.predict_proba(X_train.iloc[predict_idx])[:, 1]
        predicted[predict_idx] = True
        logger.debug("fold %d: fitted on %d rows, scored %d", fold, len(fit_idx), len(predict_idx))

    if (~predicted).any():
        # The earliest block sits inside every expanding window, so no fold can
        # score it going forwards. Rather than fall back to an in-sample score,
        # fit one more model on everything *after* that block and use it to
        # backcast. The direction of time is reversed, but the prediction is
        # still made by a model that never saw these rows.
        uncovered = train.index[~predicted]
        logger.info(
            "backcasting %s earliest training rows with a model fitted on the later blocks",
            f"{len(uncovered):,}",
        )
        backcaster = _fit_calibrated(estimator, X_train, y_train, np.flatnonzero(predicted), config)
        scores.loc[uncovered] = backcaster.predict_proba(X_train[~predicted])[:, 1]

    if scores.isna().any():  # pragma: no cover
        raise AssertionError(f"{int(scores.isna().sum())} transactions went unscored")
    return scores


def _fit_calibrated(
    estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    fit_index: np.ndarray,
    config: Config,
) -> Any:
    """Fit a fold model and calibrate it on a held-out slice of its own data.

    Without this the out-of-fold scores would be raw SMOTE-trained outputs while
    the validation and test scores come from a calibrated model - two different
    probability scales stitched into one column. Since CLV multiplies by that
    column, the mismatch would show up directly in pounds, so every fold model
    is calibrated the same way the deployed one is.
    """
    method = config.training.calibration
    if method == "none" or len(fit_index) < 50:
        model = clone(estimator)
        model.fit(X.iloc[fit_index], y[fit_index])
        return model

    fit_part, calib_part = train_test_split(
        fit_index,
        test_size=0.2,
        random_state=config.seed,
        stratify=y[fit_index],
    )
    model = clone(estimator)
    model.fit(X.iloc[fit_part], y[fit_part])
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    except ImportError:  # pragma: no cover - scikit-learn < 1.6
        calibrated = CalibratedClassifierCV(model, cv="prefit", method=method)
    calibrated.fit(X.iloc[calib_part], y[calib_part])
    return calibrated


def _unwrap_calibrated(pipeline: Any) -> Any:
    """Return an unfitted-compatible estimator for cross-validation.

    ``CalibratedClassifierCV`` wrapping a frozen estimator cannot be refitted
    from scratch, so cross-validation uses the underlying pipeline. The
    difference is calibration only; ranking is unaffected.
    """
    inner = getattr(pipeline, "estimator", None)
    if inner is None:
        return pipeline
    # sklearn's FrozenEstimator keeps the original under `estimator`.
    return getattr(inner, "estimator", inner)


def customer_table(df: pd.DataFrame, scores: pd.Series, config: Config) -> pd.DataFrame:
    """Aggregate transactions to one row per customer.

    Returns the spend, observed purchase rate and mean predicted return risk
    that the CLV formula consumes.
    """
    frame = df.copy()
    frame["predicted_return_prob"] = scores
    dates = pd.to_datetime(frame[config.features.date_column])
    frame["_date"] = dates

    grouped = frame.groupby("customer_id")
    customers = grouped.agg(
        segment=("customer_segment", "first"),
        n_transactions=(config.features.target, "size"),
        total_revenue_gbp=("order_value_gbp", "sum"),
        avg_order_value_gbp=("order_value_gbp", "mean"),
        actual_return_rate=(config.features.target, "mean"),
        predicted_return_prob=("predicted_return_prob", "mean"),
        first_order=("_date", "min"),
        last_order=("_date", "max"),
        stated_order_frequency_12m=("order_frequency_12m", "max"),
    )

    # Observed purchase rate: orders per year over the period the customer was
    # actually active, floored so a single-order customer is not extrapolated
    # into an implausibly high annual rate.
    window_end = dates.max()
    active_years = ((window_end - customers["first_order"]).dt.days / DAYS_PER_YEAR).clip(
        lower=MIN_ACTIVE_YEARS
    )
    customers["active_years"] = active_years
    customers["annual_orders"] = customers["n_transactions"] / active_years
    return customers.reset_index()


def compute_clv(
    customers: pd.DataFrame,
    economics: EconomicsConfig,
    clv_config: ClvConfig,
    *,
    probability_column: str = "predicted_return_prob",
    intervention: np.ndarray | pd.Series | None = None,
    suffix: str = "",
) -> pd.DataFrame:
    """Add discounted lifetime value columns to ``customers``.

    Args:
        customers: Customer-level table from :func:`customer_table`.
        economics: Margin, return cost and intervention parameters.
        clv_config: Horizon, discount rate and retention.
        probability_column: Which return-probability column to use.
        intervention: Optional boolean mask of customers receiving the
            return-reduction intervention on every order.
        suffix: Appended to the generated column names, so a policy scenario can
            sit alongside the baseline in one frame.

    Returns:
        ``customers`` with ``annual_gross_margin``, ``expected_return_cost``,
        ``annual_contribution`` and ``clv_gbp`` columns (plus ``suffix``).
    """
    out = customers.copy()
    probability = out[probability_column].to_numpy(dtype=float)
    aov = out["avg_order_value_gbp"].to_numpy(dtype=float)
    orders = out["annual_orders"].to_numpy(dtype=float)

    treated = (
        np.zeros(len(out), dtype=bool)
        if intervention is None
        else np.asarray(intervention, dtype=bool)
    )
    effective_probability = np.where(
        treated, probability * (1 - economics.intervention_effectiveness), probability
    )

    gross_margin = orders * aov * economics.gross_margin_rate
    cost_per_return = (
        aov * economics.gross_margin_rate
        + economics.return_logistics_cost_gbp
        + (aov * economics.return_restocking_loss_rate)
    )
    expected_return_cost = orders * effective_probability * cost_per_return
    intervention_spend = np.where(treated, orders * economics.intervention_cost_gbp, 0.0)
    annual_contribution = gross_margin - expected_return_cost - intervention_spend

    # Discounted, retention-weighted sum over the horizon.
    discount = 0.0
    for year in range(1, clv_config.horizon_years + 1):
        discount += (
            clv_config.annual_retention_rate ** (year - 1) / (1 + clv_config.discount_rate) ** year
        )

    out[f"annual_gross_margin_gbp{suffix}"] = gross_margin
    out[f"expected_return_cost_gbp{suffix}"] = expected_return_cost
    out[f"intervention_spend_gbp{suffix}"] = intervention_spend
    out[f"annual_contribution_gbp{suffix}"] = annual_contribution
    out[f"clv_gbp{suffix}"] = annual_contribution * discount
    return out


def assign_risk_bands(customers: pd.DataFrame, clv_config: ClvConfig) -> pd.DataFrame:
    """Label each customer Low / Medium / High risk on predicted probability."""
    out = customers.copy()
    bands = clv_config.risk_bands
    out["risk_segment"] = np.select(
        [out["predicted_return_prob"] < bands.low, out["predicted_return_prob"] < bands.medium],
        ["Low_Risk", "Medium_Risk"],
        default="High_Risk",
    )
    out["risk_segment"] = pd.Categorical(
        out["risk_segment"], categories=["Low_Risk", "Medium_Risk", "High_Risk"], ordered=True
    )
    return out


def segment_summary(customers: pd.DataFrame) -> pd.DataFrame:
    """Per-risk-band roll-up used by the report and the dashboard."""
    summary = customers.groupby("risk_segment", observed=False).agg(
        customers=("customer_id", "count"),
        mean_predicted_return_prob=("predicted_return_prob", "mean"),
        mean_actual_return_rate=("actual_return_rate", "mean"),
        mean_order_value_gbp=("avg_order_value_gbp", "mean"),
        mean_annual_orders=("annual_orders", "mean"),
        mean_clv_gbp=("clv_gbp", "mean"),
        total_clv_gbp=("clv_gbp", "sum"),
        total_revenue_gbp=("total_revenue_gbp", "sum"),
    )
    summary["share_of_customers"] = summary["customers"] / summary["customers"].sum()
    return summary


def run_clv(config: Config, *, df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Execute the CLV stage: score, aggregate, value, segment and simulate.

    Returns:
        Summary dictionary, also written to ``outputs/reports/clv_summary.json``.
    """
    banner(logger, "stage 4 of 5: customer lifetime value")
    config.paths.ensure_dirs()
    from returns_clv.train import load_dataset  # local import avoids a cycle

    df = load_dataset(config) if df is None else df
    bundle = load_bundle(config.paths.models_path)
    logger.info("scoring with %s (threshold %.2f)", bundle.model_name, bundle.threshold)

    scores = out_of_sample_scores(df, bundle, config)
    logger.info(
        "mean out-of-sample predicted return probability %.4f against an actual rate of %.4f",
        scores.mean(),
        df[config.features.target].mean(),
    )

    customers = customer_table(df, scores, config)
    customers = compute_clv(customers, config.economics, config.clv)
    customers = assign_risk_bands(customers, config.clv)

    # --- policy simulation: intervene on the high-risk band ------------------
    targeted = (customers["risk_segment"] == "High_Risk").to_numpy()
    customers = compute_clv(
        customers,
        config.economics,
        config.clv,
        intervention=targeted,
        suffix="_policy",
    )
    customers["clv_uplift_gbp"] = customers["clv_gbp_policy"] - customers["clv_gbp"]

    summary_table = segment_summary(customers)
    uplift_by_segment = customers.groupby("risk_segment", observed=False).agg(
        customers=("customer_id", "count"),
        mean_clv_gbp=("clv_gbp", "mean"),
        mean_clv_policy_gbp=("clv_gbp_policy", "mean"),
        mean_uplift_gbp=("clv_uplift_gbp", "mean"),
        total_uplift_gbp=("clv_uplift_gbp", "sum"),
    )

    # --- figures and tables --------------------------------------------------
    figures = config.paths.figures_path
    reports = config.paths.reports_path
    plots.plot_clv_by_segment(summary_table, figures / "08_clv_by_risk_segment.png")
    plots.plot_clv_distribution(customers["clv_gbp"], figures / "09_clv_distribution.png")
    plots.plot_policy_impact(
        uplift_by_segment.rename(columns={"total_uplift_gbp": "net_saving_gbp"}),
        figures / "10_clv_policy_uplift.png",
    )

    customers.round(4).to_csv(reports / "customer_clv.csv", index=False)
    summary_table.round(4).to_csv(reports / "clv_by_risk_segment.csv")
    uplift_by_segment.round(4).to_csv(reports / "clv_policy_uplift.csv")

    total_clv = float(customers["clv_gbp"].sum())
    total_uplift = float(customers["clv_uplift_gbp"].sum())
    summary: dict[str, Any] = {
        "model": bundle.model_name,
        "assumptions": {
            "horizon_years": config.clv.horizon_years,
            "discount_rate": config.clv.discount_rate,
            "annual_retention_rate": config.clv.annual_retention_rate,
            "gross_margin_rate": config.economics.gross_margin_rate,
            "return_logistics_cost_gbp": config.economics.return_logistics_cost_gbp,
            "return_restocking_loss_rate": config.economics.return_restocking_loss_rate,
            "intervention_cost_gbp": config.economics.intervention_cost_gbp,
            "intervention_effectiveness": config.economics.intervention_effectiveness,
            "frequency_source": "observed transactions per active year",
        },
        "n_customers": int(len(customers)),
        "portfolio_clv_gbp": total_clv,
        "mean_clv_gbp": float(customers["clv_gbp"].mean()),
        "median_clv_gbp": float(customers["clv_gbp"].median()),
        "negative_clv_customers": int((customers["clv_gbp"] < 0).sum()),
        "expected_return_cost_gbp": float(customers["expected_return_cost_gbp"].sum()),
        "risk_segments": summary_table.round(4).reset_index().to_dict(orient="records"),
        "policy": {
            "targeted_segment": "High_Risk",
            "customers_targeted": int(targeted.sum()),
            "portfolio_clv_after_gbp": float(customers["clv_gbp_policy"].sum()),
            "total_uplift_gbp": total_uplift,
            "uplift_pct": total_uplift / total_clv if total_clv else float("nan"),
            "by_segment": uplift_by_segment.round(4).reset_index().to_dict(orient="records"),
        },
    }
    write_json(summary, reports / "clv_summary.json")

    logger.info("portfolio CLV £%s across %s customers", f"{total_clv:,.0f}", f"{len(customers):,}")
    logger.info(
        "targeting the %s high-risk customers changes portfolio CLV by £%s (%.2f%%)",
        f"{int(targeted.sum()):,}",
        f"{total_uplift:,.0f}",
        100 * total_uplift / total_clv if total_clv else float("nan"),
    )
    return summary
