"""Train / validation / test splitting strategies.

The default is a **temporal** split: train on the earliest transactions,
validate on the next block, test on the most recent block. This is the only
protocol that answers the question a retailer actually has — "will this model
work on next quarter's orders?" — and it is strictly harder than a random split,
because any drift between periods counts against the model.

Two alternatives are available for comparison:

* ``random``   - stratified random split, as used in the coursework notebooks.
* ``grouped``  - split by ``customer_id`` so no customer appears in two sets,
  which isolates the model's ability to generalise to unseen customers.

A three-way split matters for this project: the decision threshold and the
probability calibrator are both fitted on the validation block, so the test
block stays untouched until the final report. Tuning a threshold on the test set
(as the original notebook did) reports an optimistically biased number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from returns_clv.config import Config
from returns_clv.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DataSplit:
    """Three disjoint transaction frames plus a description of how they were cut."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    strategy: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        overlap = (
            set(self.train.index) & set(self.validation.index)
            | set(self.train.index) & set(self.test.index)
            | set(self.validation.index) & set(self.test.index)
        )
        if overlap:
            raise ValueError(f"splits overlap on {len(overlap)} rows")

    def describe(self, target: str) -> dict[str, Any]:
        summary: dict[str, Any] = {"strategy": self.strategy, **self.details}
        total = len(self.train) + len(self.validation) + len(self.test)
        for name, frame in self.items():
            summary[name] = {
                "n_rows": int(len(frame)),
                "share": round(len(frame) / total, 4),
                "positive_rate": round(float(frame[target].mean()), 4),
            }
        return summary

    def items(self) -> list[tuple[str, pd.DataFrame]]:
        return [("train", self.train), ("validation", self.validation), ("test", self.test)]


def make_split(df: pd.DataFrame, config: Config) -> DataSplit:
    """Split ``df`` according to ``config.split.strategy``.

    Args:
        df: Transaction-level dataset. Not modified.
        config: Supplies the strategy and the validation/test proportions.

    Returns:
        A :class:`DataSplit` whose frames preserve ``df``'s index, so predictions
        can be joined back onto the full dataset.
    """
    strategy = config.split.strategy
    if strategy == "temporal":
        split = _temporal_split(df, config)
    elif strategy == "random":
        split = _random_split(df, config)
    elif strategy == "grouped":
        split = _grouped_split(df, config)
    else:  # pragma: no cover - guarded by SplitConfig validation
        raise ValueError(f"unknown split strategy {strategy!r}")

    logger.info(
        "%s split: train=%s validation=%s test=%s (positive rate %.3f / %.3f / %.3f)",
        strategy,
        f"{len(split.train):,}",
        f"{len(split.validation):,}",
        f"{len(split.test):,}",
        split.train[config.features.target].mean(),
        split.validation[config.features.target].mean(),
        split.test[config.features.target].mean(),
    )
    return split


def _temporal_split(df: pd.DataFrame, config: Config) -> DataSplit:
    date_col = config.features.date_column
    dates = pd.to_datetime(df[date_col])

    # Cut on calendar dates rather than row positions so a single day never
    # straddles two splits - otherwise near-identical same-day orders appear on
    # both sides of the boundary.
    train_share = 1.0 - config.split.val_size - config.split.test_size
    val_cutoff = dates.quantile(train_share)
    test_cutoff = dates.quantile(train_share + config.split.val_size)
    val_cutoff = pd.Timestamp(val_cutoff).normalize()
    test_cutoff = pd.Timestamp(test_cutoff).normalize()

    train_mask = dates < val_cutoff
    val_mask = (dates >= val_cutoff) & (dates < test_cutoff)
    test_mask = dates >= test_cutoff

    return DataSplit(
        train=df.loc[train_mask],
        validation=df.loc[val_mask],
        test=df.loc[test_mask],
        strategy="temporal",
        details={
            "train_period": [
                dates[train_mask].min().date().isoformat(),
                val_cutoff.date().isoformat(),
            ],
            "validation_period": [val_cutoff.date().isoformat(), test_cutoff.date().isoformat()],
            "test_period": [
                test_cutoff.date().isoformat(),
                dates[test_mask].max().date().isoformat(),
            ],
        },
    )


def _random_split(df: pd.DataFrame, config: Config) -> DataSplit:
    target = config.features.target
    holdout_size = config.split.val_size + config.split.test_size
    train, holdout = train_test_split(
        df,
        test_size=holdout_size,
        random_state=config.seed,
        stratify=df[target],
    )
    # Divide the holdout so that `test_size` of the *original* data ends up in test.
    test_fraction = config.split.test_size / holdout_size
    validation, test = train_test_split(
        holdout,
        test_size=test_fraction,
        random_state=config.seed,
        stratify=holdout[target],
    )
    return DataSplit(
        train=train,
        validation=validation,
        test=test,
        strategy="random",
        details={"stratified_on": target},
    )


def _grouped_split(df: pd.DataFrame, config: Config) -> DataSplit:
    groups = df["customer_id"].to_numpy()
    holdout_size = config.split.val_size + config.split.test_size

    outer = GroupShuffleSplit(n_splits=1, test_size=holdout_size, random_state=config.seed)
    train_idx, holdout_idx = next(outer.split(df, groups=groups))
    train = df.iloc[train_idx]
    holdout = df.iloc[holdout_idx]

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=config.split.test_size / holdout_size,
        random_state=config.seed,
    )
    val_idx, test_idx = next(inner.split(holdout, groups=holdout["customer_id"].to_numpy()))
    validation, test = holdout.iloc[val_idx], holdout.iloc[test_idx]

    shared = set(train["customer_id"]) & set(test["customer_id"])
    if shared:  # pragma: no cover - GroupShuffleSplit guarantees this
        raise AssertionError(f"{len(shared)} customers leaked across the grouped split")

    return DataSplit(
        train=train,
        validation=validation,
        test=test,
        strategy="grouped",
        details={"grouped_on": "customer_id", "n_customers": int(df["customer_id"].nunique())},
    )


def cv_indices(
    df: pd.DataFrame, config: Config, n_splits: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cross-validation folds matching the configured split strategy.

    Under a temporal protocol this returns expanding-window folds
    (``TimeSeriesSplit`` semantics), so hyperparameter search never trains on
    transactions that happen after the fold it scores.
    """
    from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

    if config.split.strategy == "temporal":
        ordered = np.argsort(
            pd.to_datetime(df[config.features.date_column]).to_numpy(), kind="stable"
        )
        splitter = TimeSeriesSplit(n_splits=n_splits)
        return [(ordered[train], ordered[test]) for train, test in splitter.split(ordered)]

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
    return list(splitter.split(df, df[config.features.target]))
