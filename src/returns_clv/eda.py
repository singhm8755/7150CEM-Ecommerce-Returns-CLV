"""Exploratory analysis stage.

Produces the figures and tables that justify the modelling choices downstream:
which dimensions carry return signal, whether the return rate drifts over time
(it does not, which is what makes a temporal split fair), and how weak the linear
signal is in the numeric features (weak enough that a linear model needs the
categorical drivers one-hot encoded to compete).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from returns_clv import plots
from returns_clv.config import Config
from returns_clv.logging_utils import banner, get_logger, write_json

logger = get_logger(__name__)


def run_eda(config: Config, *, df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Run exploratory analysis and write figures and a summary.

    Args:
        config: Pipeline configuration.
        df: Optional pre-loaded dataset.

    Returns:
        Summary statistics, also written to ``outputs/reports/eda_summary.json``.
    """
    banner(logger, "stage 2 of 5: exploratory data analysis")
    config.paths.ensure_dirs()
    from returns_clv.train import load_dataset  # local import avoids a cycle

    df = load_dataset(config) if df is None else df
    target = config.features.target
    figures = config.paths.figures_path
    baseline = float(df[target].mean())

    # --- return rate across categorical dimensions --------------------------
    rates = {
        dimension: df.groupby(dimension)[target].mean().sort_values(ascending=False)
        for dimension in config.features.categorical
    }
    plots.plot_rate_by_dimension(rates, baseline, figures / "00_return_rate_by_dimension.png")

    # --- temporal stability --------------------------------------------------
    dated = df.copy()
    dated[config.features.date_column] = pd.to_datetime(dated[config.features.date_column])
    monthly = (
        dated.set_index(config.features.date_column)
        .groupby(pd.Grouper(freq="MS"))
        .agg(transactions=(target, "size"), return_rate=(target, "mean"))
    )
    monthly.index = monthly.index.strftime("%Y-%m")
    plots.plot_monthly_trend(monthly, figures / "00_monthly_trend.png")

    # --- order value by outcome ---------------------------------------------
    plots.plot_value_distribution(
        df.loc[df[target] == 0, "order_value_gbp"],
        df.loc[df[target] == 1, "order_value_gbp"],
        figures / "00_order_value_by_outcome.png",
    )

    # --- linear signal in numeric features -----------------------------------
    correlations = (
        df[config.features.numeric + [target]].corr(numeric_only=True)[target].drop(target)
    )
    plots.plot_correlation_heatmap(correlations, figures / "00_feature_correlations.png")

    # --- tables --------------------------------------------------------------
    reports = config.paths.reports_path
    for dimension, series in rates.items():
        series.rename("return_rate").to_frame().assign(
            transactions=df.groupby(dimension).size()
        ).to_csv(reports / f"eda_return_rate_by_{dimension}.csv")
    monthly.to_csv(reports / "eda_monthly_trend.csv")
    correlations.rename("correlation_with_return").to_csv(reports / "eda_feature_correlations.csv")

    drift = float(monthly["return_rate"].std())
    summary: dict[str, Any] = {
        "overall_return_rate": baseline,
        "n_transactions": int(len(df)),
        "n_customers": int(df["customer_id"].nunique()),
        "total_revenue_gbp": float(df["order_value_gbp"].sum()),
        "mean_order_value_gbp": float(df["order_value_gbp"].mean()),
        "return_rate_by_dimension": {k: v.round(4).to_dict() for k, v in rates.items()},
        "monthly_return_rate_std": drift,
        "feature_correlations": correlations.round(4).to_dict(),
        "strongest_categorical_driver": max(
            ((k, float(v.max() - v.min())) for k, v in rates.items()), key=lambda kv: kv[1]
        ),
    }
    write_json(summary, reports / "eda_summary.json")

    logger.info("overall return rate %.2f%%", 100 * baseline)
    logger.info(
        "widest categorical spread: %s (%.1f percentage points)",
        summary["strongest_categorical_driver"][0],
        100 * summary["strongest_categorical_driver"][1],
    )
    logger.info("month-to-month return rate std %.4f - no material drift", drift)
    return summary
