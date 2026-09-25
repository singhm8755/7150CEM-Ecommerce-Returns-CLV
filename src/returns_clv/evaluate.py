"""Metrics, decision thresholds and the money model.

Three ideas run through this module.

**Rank quality and decision quality are separate questions.** ROC-AUC and
average precision measure how well the model orders transactions and are
threshold-free. Precision, recall and F1 measure a *decision* and change
completely with the threshold. Both are reported, never conflated.

**Probabilities must be calibrated, because the CLV model multiplies by them.**
A model can rank perfectly and still be useless for expected-cost arithmetic if
its probabilities are systematically too high. Brier score and the calibration
curve keep that honest.

**The threshold is a business choice, not a modelling one.** Maximising F1 is
arbitrary. The cost-optimal rule flags a transaction when the expected saving
from intervening exceeds the cost of intervening, which - because the saving
scales with order value - makes the optimal rule value-aware rather than a single
global cut-off. Both are computed and compared in pounds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from returns_clv.config import Config, EconomicsConfig
from returns_clv.logging_utils import get_logger

logger = get_logger(__name__)

THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


@dataclass(frozen=True)
class ThresholdChoice:
    """A chosen operating point and why it was chosen."""

    threshold: float
    criterion: str
    score: float
    fitted_on: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Full metric set at a given operating point.

    Args:
        y_true: Binary ground truth.
        y_proba: Predicted probability of the positive class.
        threshold: Probability above which a transaction is flagged.

    Returns:
        Threshold-free ranking metrics (``roc_auc``, ``average_precision``),
        calibration (``brier_score``), and threshold-dependent decision metrics
        including the raw confusion-matrix counts.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "positive_rate_predicted": float(y_pred.mean()),
        "positive_rate_actual": float(y_true.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ---------------------------------------------------------------------------
# Unit economics
# ---------------------------------------------------------------------------


def return_cost(order_values: np.ndarray | pd.Series, economics: EconomicsConfig) -> np.ndarray:
    """Loss incurred when an order is returned, per transaction.

    Three components: the gross margin the retailer no longer earns, the fixed
    two-way logistics and processing cost, and a restocking loss proportional to
    order value (markdown, damage, or write-off).
    """
    values = np.asarray(order_values, dtype=float)
    return (
        values * economics.gross_margin_rate
        + economics.return_logistics_cost_gbp
        + values * economics.return_restocking_loss_rate
    )


def net_benefit(
    y_true: np.ndarray | pd.Series,
    flagged: np.ndarray | pd.Series,
    order_values: np.ndarray | pd.Series,
    economics: EconomicsConfig,
) -> dict[str, float]:
    """Pounds saved by intervening on the flagged transactions.

    The intervention (extra sizing guidance, address verification, withholding a
    free-returns offer) costs ``intervention_cost_gbp`` whenever it is applied
    and prevents a fraction ``intervention_effectiveness`` of the returns that
    would otherwise have happened. So a flag on a transaction that would have
    been returned saves ``effectiveness x return_cost``; a flag on one that would
    have been kept is pure cost.

    Args:
        y_true: Whether each transaction was actually returned.
        flagged: Whether the policy flags each transaction.
        order_values: Order value in GBP, used to size the avoided loss.
        economics: Unit economics.

    Returns:
        Gross saving, intervention spend, net saving, and the realised return
        cost before and after the policy.
    """
    y_true = np.asarray(y_true).astype(int)
    flagged = np.asarray(flagged).astype(bool)
    costs = return_cost(order_values, economics)

    baseline_cost = float((y_true * costs).sum())
    gross_saving = float((flagged & (y_true == 1)) @ costs * economics.intervention_effectiveness)
    intervention_spend = float(flagged.sum() * economics.intervention_cost_gbp)
    net = gross_saving - intervention_spend

    return {
        "n_flagged": int(flagged.sum()),
        "flag_rate": float(flagged.mean()),
        "baseline_return_cost_gbp": baseline_cost,
        "gross_saving_gbp": gross_saving,
        "intervention_spend_gbp": intervention_spend,
        "net_saving_gbp": net,
        "residual_return_cost_gbp": baseline_cost - gross_saving,
        "saving_per_transaction_gbp": net / len(y_true) if len(y_true) else 0.0,
        "roi": gross_saving / intervention_spend if intervention_spend else float("nan"),
    }


def value_aware_flags(
    y_proba: np.ndarray | pd.Series,
    order_values: np.ndarray | pd.Series,
    economics: EconomicsConfig,
) -> np.ndarray:
    """Flag a transaction when intervening has positive expected value.

    Intervene iff ``p x effectiveness x return_cost > intervention_cost``. Because
    ``return_cost`` grows with order value, the implied probability cut-off is
    lower for expensive orders - a single global threshold cannot express this.
    """
    proba = np.asarray(y_proba, dtype=float)
    costs = return_cost(order_values, economics)
    expected_saving = proba * economics.intervention_effectiveness * costs
    return expected_saving > economics.intervention_cost_gbp


def threshold_sweep(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    order_values: np.ndarray | pd.Series,
    economics: EconomicsConfig,
    grid: np.ndarray = THRESHOLD_GRID,
) -> pd.DataFrame:
    """Decision metrics and net saving across a grid of global thresholds."""
    rows = []
    for threshold in grid:
        metrics = classification_metrics(y_true, y_proba, threshold)
        flags = np.asarray(y_proba, dtype=float) >= threshold
        metrics.update(net_benefit(y_true, flags, order_values, economics))
        rows.append(metrics)
    return pd.DataFrame(rows)


def choose_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    order_values: np.ndarray | pd.Series,
    economics: EconomicsConfig,
    *,
    criterion: str = "net_saving",
    fitted_on: str = "validation",
) -> tuple[ThresholdChoice, pd.DataFrame]:
    """Pick an operating point on held-out data.

    Args:
        criterion: ``net_saving`` maximises pounds saved; ``f1`` reproduces the
            coursework objective; ``balanced_accuracy`` weighs both error types
            equally regardless of prevalence.
        fitted_on: Recorded on the result so a report can never imply a threshold
            was chosen on the test set when it was not.

    Returns:
        The chosen operating point and the full sweep behind it.
    """
    sweep = threshold_sweep(y_true, y_proba, order_values, economics)
    column = {"net_saving": "net_saving_gbp", "f1": "f1", "balanced_accuracy": "balanced_accuracy"}
    if criterion not in column:
        raise ValueError(f"unknown criterion {criterion!r}; expected one of {sorted(column)}")

    best = sweep.loc[sweep[column[criterion]].idxmax()]
    choice = ThresholdChoice(
        threshold=float(best["threshold"]),
        criterion=criterion,
        score=float(best[column[criterion]]),
        fitted_on=fitted_on,
    )
    logger.info(
        "threshold %.2f chosen on %s by %s (score %.4f)",
        choice.threshold,
        fitted_on,
        criterion,
        choice.score,
    )
    return choice, sweep


def policy_comparison(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    order_values: np.ndarray | pd.Series,
    economics: EconomicsConfig,
    *,
    threshold: float,
    heuristic_flags: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Compare the model against do-nothing, blanket and heuristic policies.

    The blanket and heuristic rows are what a business would otherwise do, so
    they are the comparison that decides whether the model is worth deploying.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(y_proba, dtype=float)
    n = len(y_true)

    policies: dict[str, np.ndarray] = {
        "Do nothing": np.zeros(n, dtype=bool),
        "Intervene on every order": np.ones(n, dtype=bool),
    }
    if heuristic_flags is not None:
        policies["Rule of thumb (no model)"] = np.asarray(heuristic_flags).astype(bool)
    policies[f"Model, global threshold {threshold:.2f}"] = proba >= threshold
    policies["Model, value-aware rule"] = value_aware_flags(proba, order_values, economics)

    rows = []
    for name, flags in policies.items():
        row = {"policy": name}
        row.update(net_benefit(y_true, flags, order_values, economics))
        rows.append(row)
    return pd.DataFrame(rows).set_index("policy")


# ---------------------------------------------------------------------------
# Performance ceiling
# ---------------------------------------------------------------------------


def bayes_ceiling(
    y_true: np.ndarray | pd.Series, true_probability: np.ndarray | pd.Series
) -> dict[str, float]:
    """Metrics achieved by the true data-generating probability.

    Because this dataset is synthetic, the probability behind every outcome is
    recoverable (see :func:`returns_clv.data.generate.true_return_probability`).
    Scoring *that* gives the best performance any model could reach: the rest of
    the gap to 1.0 is coin-flip noise, not a modelling failure. Reporting a
    model's AUC without this ceiling invites the reader to assume 0.95 was
    available when it never was.
    """
    y_true = np.asarray(y_true).astype(int)
    truth = np.asarray(true_probability, dtype=float)
    return {
        "roc_auc": float(roc_auc_score(y_true, truth)),
        "average_precision": float(average_precision_score(y_true, truth)),
        "brier_score": float(brier_score_loss(y_true, truth)),
    }


def attainment(model_metrics: dict[str, float], ceiling: dict[str, float]) -> dict[str, float]:
    """How much of the *available* signal a model captured.

    For AUC the useful scale runs from 0.5 (random) to the ceiling, so
    attainment is ``(model - 0.5) / (ceiling - 0.5)``. A model at 0.90
    attainment has captured nine tenths of the learnable signal, which is a far
    more informative statement than "AUC 0.68".
    """
    result = {}
    for metric, floor in (("roc_auc", 0.5), ("average_precision", None)):
        if metric not in model_metrics or metric not in ceiling:
            continue
        base = floor if floor is not None else 0.0
        headroom = ceiling[metric] - base
        result[f"{metric}_attainment"] = (
            float((model_metrics[metric] - base) / headroom) if headroom > 0 else float("nan")
        )
    return result


def summarise_models(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Tidy a ``{model_name: metrics}`` mapping into a comparison table."""
    frame = pd.DataFrame(results).T
    preferred = [
        "roc_auc",
        "average_precision",
        "brier_score",
        "f1",
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
        "accuracy",
        "mcc",
        "threshold",
    ]
    ordered = [c for c in preferred if c in frame.columns]
    return frame[ordered + [c for c in frame.columns if c not in ordered]]


def heuristic_baseline_flags(df: pd.DataFrame) -> np.ndarray:
    """A plausible no-model rule: flag cash-on-delivery or first-time buyers.

    This is the policy a category manager could write down without any data
    science, so it is the bar the model has to clear to justify itself.
    """
    return (
        (df["payment_method"] == "Cash_on_Delivery") | (df["customer_segment"] == "First_Time")
    ).to_numpy()


def evaluation_config_summary(config: Config) -> dict[str, Any]:
    """Record the evaluation protocol alongside results, for reproducibility."""
    return {
        "split_strategy": config.split.strategy,
        "val_size": config.split.val_size,
        "test_size": config.split.test_size,
        "resampler": config.training.resampler,
        "calibration": config.training.calibration,
        "cv_folds": config.training.cv_folds,
        "search_iterations": config.training.search_iterations,
        "scoring": config.training.scoring,
        "economics": asdict(config.economics),
        "random_seed": config.seed,
    }
