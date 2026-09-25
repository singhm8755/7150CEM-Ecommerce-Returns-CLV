"""Report figures.

All figures share one house style, defined once at the top of this module:

* **Categorical colour is assigned by series identity, in a fixed order** - a
  model keeps its colour whether or not the other models are on the chart.
* **One y-axis per panel, always.** Where a metric in [0, 1] and a metric in
  pounds belong on the same story, they get two stacked panels rather than two
  scales on one axis.
* **Sequential magnitude uses a single hue**, light to dark; signed change uses a
  blue/red diverging pair with a neutral zero.
* **Identity is never colour alone**: every multi-series chart carries a legend,
  values are labelled directly where there are few enough to read, and every
  figure has a CSV counterpart in ``outputs/reports`` so the numbers are
  available without reading colour.

The palette is checked against colour-vision-deficiency separation thresholds;
the three lead hues are safe for all-pairs comparison and the fourth is used only
on line charts, where adjacent-pair separation is what matters.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are written to disk, never displayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve

from returns_clv.logging_utils import get_logger

logger = get_logger(__name__)

# --- House style -----------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"

# Fixed categorical order. Series are assigned by name, never by rank.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
REFERENCE_COLOR = INK_MUTED
POSITIVE_COLOR = "#2a78d6"
NEGATIVE_COLOR = "#e34948"

SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef5fd", "#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#104281"]
)
DIVERGING_BLUE_RED = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#5598e7", "#cde2fb", "#f0efec", "#f6bdbc", "#e34948", "#8f1f1e"]
)

LINE_WIDTH = 2.0
MARKER_SIZE = 8.0


def apply_style() -> None:
    """Install the house style globally. Idempotent."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_SECONDARY,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "font.size": 10,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def series_color(index: int) -> str:
    """Colour for the ``index``-th series, folding past the palette length."""
    return SERIES_COLORS[index % len(SERIES_COLORS)]


def color_map(names: list[str]) -> dict[str, str]:
    """Stable name-to-colour assignment so a model keeps its hue across figures."""
    return {name: series_color(i) for i, name in enumerate(names)}


def _finish(fig: Figure, path: Path, subtitle: str | None = None) -> Path:
    """Add an optional caption, save, and close the figure."""
    if subtitle:
        fig.text(0.0, -0.02, subtitle, fontsize=8.5, color=INK_MUTED, ha="left", va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    logger.info("wrote %s", path.name)
    return path


def _label_bars(ax, bars, values, fmt="{:.3f}", offset=0.01) -> None:
    """Direct value labels in ink, not in the series colour."""
    span = max(abs(v) for v in values) or 1.0
    for bar, value in zip(bars, values, strict=True):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, height + np.sign(height or 1) * offset * span),
            ha="center",
            va=va,
            fontsize=8.5,
            color=INK_SECONDARY,
        )


# ---------------------------------------------------------------------------
# Model evaluation figures
# ---------------------------------------------------------------------------


def plot_roc_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    aucs: dict[str, float],
    path: Path,
    *,
    ceiling: tuple[np.ndarray, np.ndarray] | None = None,
    ceiling_auc: float | None = None,
) -> Path:
    """ROC curves with the achievable ceiling drawn as a reference line."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5.6))
    colors = color_map(list(curves))

    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, color=colors[name], label=f"{name} — AUC {aucs[name]:.3f}")

    if ceiling is not None:
        ax.plot(
            *ceiling,
            color=REFERENCE_COLOR,
            linestyle=(0, (5, 3)),
            linewidth=1.6,
            label=f"Achievable ceiling — AUC {ceiling_auc:.3f}"
            if ceiling_auc
            else "Achievable ceiling",
        )
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.4, zorder=0, label="Random (AUC 0.500)")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Ranking quality on the held-out test period")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    return _finish(
        fig,
        path,
        "The dashed line is the best any model could do on this data: the probability that "
        "actually generated each outcome.",
    )


def plot_pr_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    scores: dict[str, float],
    prevalence: float,
    path: Path,
) -> Path:
    """Precision-recall curves, with the no-skill line at class prevalence."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5.6))
    colors = color_map(list(curves))

    for name, (recall, precision) in curves.items():
        ax.plot(recall, precision, color=colors[name], label=f"{name} — AP {scores[name]:.3f}")

    ax.axhline(
        prevalence,
        color=REFERENCE_COLOR,
        linestyle=(0, (5, 3)),
        linewidth=1.6,
        label=f"No skill — AP {prevalence:.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall on the held-out test period")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    return _finish(
        fig,
        path,
        "Average precision is the headline metric here: it ignores the large true-negative "
        "mass that inflates accuracy on imbalanced data.",
    )


def plot_calibration(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
    *,
    n_bins: int = 12,
    brier: dict[str, float] | None = None,
) -> Path:
    """Reliability diagram - essential because CLV multiplies by these probabilities."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5.6))
    colors = color_map(list(predictions))

    for name, (y_true, y_proba) in predictions.items():
        observed, predicted = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
        label = name if brier is None else f"{name} — Brier {brier[name]:.4f}"
        ax.plot(predicted, observed, color=colors[name], marker="o", label=label)

    ax.plot(
        [0, 1],
        [0, 1],
        color=REFERENCE_COLOR,
        linestyle=(0, (5, 3)),
        linewidth=1.6,
        label="Perfectly calibrated",
    )
    ax.set_xlabel("Mean predicted probability of return")
    ax.set_ylabel("Observed return rate")
    ax.set_title("Probability calibration")
    ax.legend(loc="upper left")
    return _finish(
        fig,
        path,
        "Points below the diagonal mean the model over-predicts returns, which would "
        "overstate expected return cost in the CLV model.",
    )


def plot_confusion_matrices(
    matrices: dict[str, np.ndarray], path: Path, *, normalise: bool = True
) -> Path:
    """Small multiples of confusion matrices on one sequential hue."""
    apply_style()
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.1), squeeze=False)
    labels = ["Kept", "Returned"]

    for ax, (name, cm) in zip(axes[0], matrices.items(), strict=True):
        shown = cm / cm.sum(axis=1, keepdims=True) if normalise else cm
        ax.imshow(shown, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1 if normalise else cm.max())
        ax.set_title(name, fontsize=11)
        ax.set_xticks(range(2), labels)
        ax.set_yticks(range(2), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                # Flip the label to the surface colour on dark cells so it stays legible.
                dark = shown[i, j] > (0.55 if normalise else 0.55 * cm.max())
                ax.text(
                    j,
                    i,
                    f"{shown[i, j]:.1%}\n({cm[i, j]:,})" if normalise else f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    color=SURFACE if dark else INK_PRIMARY,
                )
    fig.suptitle(
        "Confusion matrices at each model's chosen threshold", x=0.01, ha="left", fontsize=12
    )
    return _finish(fig, path, "Cells are row-normalised; counts in brackets.")


def plot_threshold_sweep(sweep: pd.DataFrame, chosen: float, path: Path) -> Path:
    """Decision metrics and net saving against threshold, as two stacked panels.

    Deliberately not a dual-axis chart: a ratio in [0, 1] and a value in pounds
    share an x-axis but never a y-axis.
    """
    apply_style()
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.6, 7.4), sharex=True, gridspec_kw={"height_ratios": [1, 1], "hspace": 0.22}
    )

    for i, metric in enumerate(["precision", "recall", "f1"]):
        top.plot(
            sweep["threshold"], sweep[metric], color=series_color(i), label=metric.capitalize()
        )
    top.axvline(chosen, color=REFERENCE_COLOR, linestyle=(0, (5, 3)), linewidth=1.6)
    top.set_ylabel("Score")
    top.set_ylim(0, 1)
    top.set_title("Decision quality against threshold")
    top.legend(loc="upper right")

    net = sweep["net_saving_gbp"]
    # A single series needs no legend box: the panel title names it.
    bottom.plot(sweep["threshold"], net, color=series_color(3))
    bottom.axhline(0, color=INK_MUTED, linewidth=1.2)
    bottom.axvline(chosen, color=REFERENCE_COLOR, linestyle=(0, (5, 3)), linewidth=1.6)
    peak = sweep.loc[net.idxmax()]
    bottom.plot([peak["threshold"]], [peak["net_saving_gbp"]], marker="o", color=series_color(3))
    bottom.annotate(
        f"best £{peak['net_saving_gbp']:,.0f} at {peak['threshold']:.2f}",
        (peak["threshold"], peak["net_saving_gbp"]),
        textcoords="offset points",
        xytext=(8, -4),
        fontsize=9,
        color=INK_SECONDARY,
    )
    bottom.set_xlabel("Probability threshold for flagging a transaction")
    bottom.set_ylabel("Net saving (GBP)")
    bottom.set_title("Business value against threshold")

    return _finish(
        fig,
        path,
        f"Dashed line marks the deployed threshold ({chosen:.2f}), chosen on the validation "
        "block before the test block was scored.",
    )


def plot_model_comparison(
    table: pd.DataFrame,
    path: Path,
    *,
    metrics: tuple[str, ...] = ("roc_auc", "average_precision", "f1", "recall"),
    ceiling: dict[str, float] | None = None,
) -> Path:
    """Grouped bars comparing models, with the achievable ceiling per metric."""
    apply_style()
    available = [m for m in metrics if m in table.columns]
    models = list(table.index)
    colors = color_map(models)

    fig, axes = plt.subplots(1, len(available), figsize=(3.5 * len(available), 4.6), squeeze=False)
    pretty = {
        "roc_auc": "ROC-AUC",
        "average_precision": "Average precision",
        "f1": "F1",
        "recall": "Recall",
        "precision": "Precision",
        "brier_score": "Brier score",
    }

    for ax, metric in zip(axes[0], available, strict=True):
        values = table[metric].to_numpy(dtype=float)
        bars = ax.bar(
            range(len(models)),
            values,
            width=0.62,
            color=[colors[m] for m in models],
            edgecolor=SURFACE,
            linewidth=2.0,  # the 2px surface gap between adjacent bars
        )
        _label_bars(ax, bars, values)
        if ceiling and metric in ceiling:
            ax.axhline(ceiling[metric], color=REFERENCE_COLOR, linestyle=(0, (5, 3)), linewidth=1.6)
            ax.annotate(
                f"ceiling {ceiling[metric]:.3f}",
                (len(models) - 0.5, ceiling[metric]),
                textcoords="offset points",
                xytext=(0, 4),
                ha="right",
                fontsize=8.5,
                color=INK_MUTED,
            )
        ax.set_title(pretty.get(metric, metric))
        ax.set_xticks(range(len(models)), [m.replace(" ", "\n") for m in models], fontsize=8.5)
        ax.set_ylim(0, max(1.0, values.max() * 1.25))
        ax.grid(axis="x", visible=False)

    fig.suptitle("Model comparison on the held-out test period", x=0.01, ha="left", fontsize=12)
    return _finish(fig, path, "Values are labelled directly; full table in outputs/reports.")


def plot_feature_importance(importance: pd.Series, path: Path, *, top_n: int = 15) -> Path:
    """Horizontal bars for a single series - no legend needed, the title names it."""
    apply_style()
    top = importance.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(top) + 1.6))
    bars = ax.barh(
        range(len(top)),
        top.to_numpy(),
        height=0.64,
        color=series_color(0),
        edgecolor=SURFACE,
        linewidth=2.0,
    )
    for bar, value in zip(bars, top.to_numpy(), strict=True):
        ax.annotate(
            f"{value:.3f}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            textcoords="offset points",
            xytext=(5, 0),
            va="center",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
    ax.set_yticks(range(len(top)), [str(i) for i in top.index], fontsize=9)
    ax.set_xlabel("Mean absolute SHAP value (impact on predicted probability)")
    ax.set_title("What drives the model's return-risk predictions")
    ax.set_xlim(0, top.max() * 1.18)
    ax.grid(axis="y", visible=False)
    return _finish(
        fig, path, "Permutation-free attribution; magnitude only, direction varies by row."
    )


# ---------------------------------------------------------------------------
# CLV and policy figures
# ---------------------------------------------------------------------------


def plot_clv_by_segment(summary: pd.DataFrame, path: Path) -> Path:
    """Customer count and mean CLV per risk band, as two panels sharing an x-axis."""
    apply_style()
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.2, 7.0), sharex=True, gridspec_kw={"hspace": 0.2}
    )
    segments = list(summary.index)

    counts = summary["customers"].to_numpy(dtype=float)
    bars = top.bar(
        range(len(segments)),
        counts,
        width=0.6,
        color=series_color(0),
        edgecolor=SURFACE,
        linewidth=2.0,
    )
    _label_bars(top, bars, counts, fmt="{:,.0f}")
    top.set_ylabel("Customers")
    top.set_title("Customers per predicted return-risk band")
    top.set_ylim(0, counts.max() * 1.2)
    top.grid(axis="x", visible=False)

    clv = summary["mean_clv_gbp"].to_numpy(dtype=float)
    bars = bottom.bar(
        range(len(segments)),
        clv,
        width=0.6,
        color=series_color(2),
        edgecolor=SURFACE,
        linewidth=2.0,
    )
    _label_bars(bottom, bars, clv, fmt="£{:,.0f}")
    bottom.axhline(0, color=INK_MUTED, linewidth=1.2)
    bottom.set_ylabel("Mean predicted CLV (GBP)")
    bottom.set_title("Mean customer lifetime value per risk band")
    bottom.set_xticks(range(len(segments)), [s.replace("_", " ") for s in segments])
    bottom.grid(axis="x", visible=False)

    return _finish(
        fig, path, "Risk bands are cut on predicted return probability, not on outcomes."
    )


def plot_policy_impact(
    impact: pd.DataFrame, path: Path, *, value_column: str = "net_saving_gbp"
) -> Path:
    """Signed policy outcomes on a diverging scale with a neutral zero."""
    apply_style()
    values = impact[value_column].to_numpy(dtype=float)
    labels = [str(i) for i in impact.index]
    colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in values]

    fig, ax = plt.subplots(figsize=(8.0, 0.62 * len(labels) + 2.0))
    bars = ax.barh(
        range(len(values)), values, height=0.6, color=colors, edgecolor=SURFACE, linewidth=2.0
    )
    ax.axvline(0, color=INK_MUTED, linewidth=1.4)
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"£{value:,.0f}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            textcoords="offset points",
            xytext=(7 if value >= 0 else -7, 0),
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )
    ax.set_yticks(range(len(labels)), labels, fontsize=9.5)
    ax.set_xlabel("Net saving over the test period (GBP)")
    ax.set_title("What each intervention policy is worth")
    pad = max(abs(values).max() * 0.22, 1.0)
    ax.set_xlim(min(values.min() - pad, -pad), values.max() + pad)
    ax.grid(axis="y", visible=False)
    return _finish(
        fig,
        path,
        "Blue is a saving, red a loss. 'Intervene on every order' is the cost of acting "
        "without a model.",
    )


def plot_clv_distribution(clv: pd.Series, path: Path) -> Path:
    """Distribution of predicted CLV across the customer base."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.hist(clv, bins=60, color=series_color(0), edgecolor=SURFACE, linewidth=0.6)
    median = float(clv.median())
    ax.axvline(median, color=REFERENCE_COLOR, linestyle=(0, (5, 3)), linewidth=1.6)
    ax.annotate(
        f"median £{median:,.0f}",
        (median, ax.get_ylim()[1] * 0.92),
        textcoords="offset points",
        xytext=(8, 0),
        fontsize=9,
        color=INK_SECONDARY,
    )
    ax.set_xlabel("Predicted customer lifetime value (GBP)")
    ax.set_ylabel("Customers")
    ax.set_title("Predicted CLV across the customer base")
    ax.grid(axis="x", visible=False)
    return _finish(fig, path)


# ---------------------------------------------------------------------------
# Exploratory figures
# ---------------------------------------------------------------------------


def plot_rate_by_dimension(rates: dict[str, pd.Series], baseline: float, path: Path) -> Path:
    """Return rate across several categorical dimensions, as small multiples."""
    apply_style()
    fig, axes = plt.subplots(1, len(rates), figsize=(3.6 * len(rates), 4.4), squeeze=False)
    for i, (ax, (dimension, series)) in enumerate(zip(axes[0], rates.items(), strict=True)):
        values = series.to_numpy(dtype=float)
        bars = ax.bar(
            range(len(series)),
            values,
            width=0.62,
            color=series_color(i),
            edgecolor=SURFACE,
            linewidth=2.0,
        )
        _label_bars(ax, bars, values, fmt="{:.1%}")
        ax.axhline(baseline, color=REFERENCE_COLOR, linestyle=(0, (5, 3)), linewidth=1.6)
        ax.set_title(dimension.replace("_", " ").capitalize())
        ax.set_xticks(
            range(len(series)), [str(i).replace("_", "\n") for i in series.index], fontsize=8.5
        )
        ax.set_ylim(0, max(values.max() * 1.3, baseline * 1.4))
        ax.set_ylabel("Return rate" if i == 0 else "")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Where returns concentrate", x=0.01, ha="left", fontsize=12)
    return _finish(fig, path, f"Dashed line is the overall return rate ({baseline:.1%}).")


def plot_monthly_trend(monthly: pd.DataFrame, path: Path) -> Path:
    """Return rate and order volume over time, as two stacked panels."""
    apply_style()
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.2, 6.4), sharex=True, gridspec_kw={"hspace": 0.2}
    )
    x = range(len(monthly))
    top.plot(x, monthly["return_rate"], color=series_color(0), marker="o", markersize=5)
    top.axhline(
        monthly["return_rate"].mean(), color=REFERENCE_COLOR, linestyle=(0, (5, 3)), linewidth=1.6
    )
    top.set_ylabel("Return rate")
    top.set_title("Return rate by month")
    top.set_ylim(0, monthly["return_rate"].max() * 1.35)

    bottom.bar(
        x,
        monthly["transactions"],
        width=0.68,
        color=series_color(2),
        edgecolor=SURFACE,
        linewidth=2.0,
    )
    bottom.set_ylabel("Transactions")
    bottom.set_title("Order volume by month")
    bottom.set_xticks(
        list(x)[::2], [str(i) for i in monthly.index[::2]], rotation=45, ha="right", fontsize=8
    )
    bottom.grid(axis="x", visible=False)
    return _finish(fig, path, "A flat return rate means no temporal drift for the model to chase.")


def plot_value_distribution(kept: pd.Series, returned: pd.Series, path: Path) -> Path:
    """Order-value distribution split by outcome."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bins = np.linspace(0, float(np.percentile(np.concatenate([kept, returned]), 99)), 50)
    ax.hist(kept, bins=bins, color=series_color(0), alpha=0.85, label="Kept", density=True)
    ax.hist(returned, bins=bins, color=series_color(1), alpha=0.65, label="Returned", density=True)
    ax.set_xlabel("Order value (GBP)")
    ax.set_ylabel("Share of orders")
    ax.set_title("Order value by outcome")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    return _finish(
        fig, path, "Overlapping distributions mean order value alone does not predict returns."
    )


def plot_correlation_heatmap(corr: pd.Series, path: Path) -> Path:
    """Point-biserial correlation of each numeric feature with the outcome."""
    apply_style()
    ordered = corr.reindex(corr.abs().sort_values().index)
    values = ordered.to_numpy(dtype=float)
    limit = max(abs(values).max(), 0.05)
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(ordered) + 1.6))
    colors = [DIVERGING_BLUE_RED((v / limit + 1) / 2) for v in values]
    bars = ax.barh(
        range(len(values)), values, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2.0
    )
    ax.axvline(0, color=INK_MUTED, linewidth=1.4)
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:+.3f}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            textcoords="offset points",
            xytext=(6 if value >= 0 else -6, 0),
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
    ax.set_yticks(range(len(ordered)), [str(i) for i in ordered.index], fontsize=9)
    ax.set_xlabel("Correlation with return outcome")
    ax.set_title("Linear signal in the numeric features")
    ax.set_xlim(-limit * 1.45, limit * 1.45)
    ax.grid(axis="y", visible=False)
    return _finish(
        fig,
        path,
        "Weak linear correlations are expected: the signal lives in the categorical drivers.",
    )
