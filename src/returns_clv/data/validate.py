"""Dataset validation: schema, integrity, business logic and internal consistency.

Each check returns a :class:`CheckResult` with one of three statuses:

* ``PASS``  - the property holds.
* ``WARN``  - the property is violated but the data is still usable; the report
  records the size of the violation so it can be discussed rather than hidden.
* ``FAIL``  - the data cannot be modelled as-is.

The suite deliberately has teeth: run against the original coursework dataset it
surfaces two genuine integrity defects (see ``docs/methodology.md``) rather than
printing "PASS" for everything.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from returns_clv.config import Config
from returns_clv.data.generate import (
    CLICK_DEPTH_RANGE,
    DWELL_SECONDS_RANGE,
    ORDER_VALUE_RANGE,
    PAGE_VISITS_RANGE,
    TRAILING_WINDOW_DAYS,
    true_return_probability,
)
from returns_clv.logging_utils import get_logger

logger = get_logger(__name__)

Status = Literal["PASS", "WARN", "FAIL"]

FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "click_depth": CLICK_DEPTH_RANGE,
    "time_on_page_seconds": DWELL_SECONDS_RANGE,
    "product_page_visits": PAGE_VISITS_RANGE,
    "order_value_gbp": ORDER_VALUE_RANGE,
}

# Customer attributes that must not change between a customer's transactions.
CUSTOMER_INVARIANT_COLUMNS = ["customer_segment"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def symbol(self) -> str:
        return {"PASS": "ok", "WARN": "warn", "FAIL": "FAIL"}[self.status]


@dataclass
class ValidationReport:
    """Collected results of a validation run."""

    checks: list[CheckResult] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "FAIL"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "WARN"]

    @property
    def passed(self) -> bool:
        """True when nothing failed. Warnings do not block the pipeline."""
        return not self.failures

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "n_checks": len(self.checks),
            "n_warnings": len(self.warnings),
            "n_failures": len(self.failures),
            "summary": self.summary,
            "checks": [asdict(c) for c in self.checks],
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"check": c.name, "status": c.status, "detail": c.detail} for c in self.checks]
        )

    def to_markdown(self) -> str:
        lines = ["| Check | Status | Detail |", "| --- | --- | --- |"]
        lines += [f"| {c.name} | {c.status} | {c.detail} |" for c in self.checks]
        return "\n".join(lines)

    def log(self) -> None:
        for check in self.checks:
            level = {"PASS": logger.info, "WARN": logger.warning, "FAIL": logger.error}[
                check.status
            ]
            level("[%-4s] %-32s %s", check.symbol, check.name, check.detail)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_schema(df: pd.DataFrame, config: Config) -> CheckResult:
    required = {
        *config.features.id_columns,
        config.features.date_column,
        config.features.target,
        *config.features.all_features,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        return CheckResult("schema", "FAIL", f"missing columns: {missing}")
    return CheckResult("schema", "PASS", f"all {len(required)} required columns present")


def _check_missing_values(df: pd.DataFrame, config: Config) -> CheckResult:
    missing = df.isna().sum()
    offenders = missing[missing > 0]
    if offenders.empty:
        return CheckResult("missing_values", "PASS", "no missing values")
    return CheckResult(
        "missing_values",
        "FAIL",
        f"{int(offenders.sum()):,} missing across {len(offenders)} column(s): "
        f"{list(offenders.index)}",
        {"n_missing": float(offenders.sum())},
    )


def _check_duplicate_ids(df: pd.DataFrame, config: Config) -> CheckResult:
    id_col = config.features.id_columns[0]
    dupes = int(df[id_col].duplicated().sum())
    if dupes == 0:
        return CheckResult("duplicate_ids", "PASS", f"{id_col} is unique")
    return CheckResult(
        "duplicate_ids",
        "FAIL",
        f"{dupes:,} duplicate {id_col} values",
        {"n_duplicates": dupes},
    )


def _check_feature_bounds(df: pd.DataFrame, config: Config) -> CheckResult:
    violations = []
    for column, (low, high) in FEATURE_BOUNDS.items():
        if column not in df.columns:
            continue
        actual_low, actual_high = float(df[column].min()), float(df[column].max())
        if actual_low < low or actual_high > high:
            violations.append(
                f"{column} in [{actual_low:g}, {actual_high:g}] expected [{low:g}, {high:g}]"
            )
    if violations:
        return CheckResult("feature_bounds", "FAIL", "; ".join(violations))
    return CheckResult(
        "feature_bounds", "PASS", f"{len(FEATURE_BOUNDS)} features within documented bounds"
    )


def _check_target(df: pd.DataFrame, config: Config) -> CheckResult:
    target = config.features.target
    values = set(pd.unique(df[target]))
    if not values <= {0, 1}:
        return CheckResult(
            "target_binary", "FAIL", f"target has non-binary values: {sorted(values)}"
        )
    return CheckResult("target_binary", "PASS", f"{target} is binary")


def _check_class_balance(df: pd.DataFrame, config: Config) -> CheckResult:
    rate = float(df[config.features.target].mean())
    if rate in (0.0, 1.0):
        return CheckResult("class_balance", "FAIL", "target has a single class")
    ratio = (1 - rate) / rate
    status: Status = "PASS" if 1.0 <= ratio <= 10.0 else "WARN"
    detail = f"positive rate {rate:.2%}, majority:minority {ratio:.2f}:1"
    if status == "WARN":
        detail += " (outside the 1:1-10:1 band resampling handles comfortably)"
    return CheckResult(
        "class_balance", status, detail, {"positive_rate": rate, "imbalance_ratio": ratio}
    )


def _check_customer_invariants(df: pd.DataFrame, config: Config) -> CheckResult:
    """Attributes describing the customer must be constant within a customer."""
    grouped = df.groupby("customer_id")
    offenders = {}
    for column in CUSTOMER_INVARIANT_COLUMNS:
        if column not in df.columns:
            continue
        inconsistent = int((grouped[column].nunique() > 1).sum())
        if inconsistent:
            offenders[column] = inconsistent
    if not offenders:
        return CheckResult(
            "customer_invariants",
            "PASS",
            f"{', '.join(CUSTOMER_INVARIANT_COLUMNS)} constant per customer",
        )
    detail = "; ".join(
        f"{col}: {n:,} customers with conflicting values" for col, n in offenders.items()
    )
    return CheckResult(
        "customer_invariants", "FAIL", detail, {k: float(v) for k, v in offenders.items()}
    )


def _check_tenure_consistency(df: pd.DataFrame, config: Config) -> CheckResult:
    """Tenure must increase with transaction date within a customer.

    A constant tenure means the field was stamped once per customer instead of
    measured at transaction time, which leaks nothing but is simply wrong.
    """
    if "customer_tenure_days" not in df.columns:
        return CheckResult("tenure_consistency", "WARN", "customer_tenure_days absent")
    frame = df[["customer_id", config.features.date_column, "customer_tenure_days"]].copy()
    frame[config.features.date_column] = pd.to_datetime(frame[config.features.date_column])
    frame = frame.sort_values(["customer_id", config.features.date_column])
    grouped = frame.groupby("customer_id")["customer_tenure_days"]
    multi_tx = grouped.size() > 1
    if not multi_tx.any():
        return CheckResult("tenure_consistency", "PASS", "no repeat customers to check")
    constant = int((grouped.nunique() == 1)[multi_tx].sum())
    n_multi = int(multi_tx.sum())
    decreasing = int((grouped.diff() < 0).groupby(frame["customer_id"]).any().sum())
    if constant == n_multi:
        return CheckResult(
            "tenure_consistency",
            "WARN",
            f"tenure is constant for all {n_multi:,} repeat customers - stamped per customer, "
            "not measured at transaction time",
            {"constant_customers": float(constant)},
        )
    if decreasing:
        return CheckResult(
            "tenure_consistency", "FAIL", f"tenure decreases over time for {decreasing:,} customers"
        )
    return CheckResult("tenure_consistency", "PASS", "tenure increases with transaction date")


def _check_order_frequency_consistency(df: pd.DataFrame, config: Config) -> CheckResult:
    """``order_frequency_12m`` should match the customer's own trailing history."""
    if "order_frequency_12m" not in df.columns:
        return CheckResult("order_frequency_consistency", "WARN", "order_frequency_12m absent")

    frame = df[["customer_id", config.features.date_column, "order_frequency_12m"]].copy()
    frame[config.features.date_column] = pd.to_datetime(frame[config.features.date_column])
    frame = frame.sort_values(["customer_id", config.features.date_column])

    window = pd.Timedelta(days=TRAILING_WINDOW_DAYS)
    observed = np.empty(len(frame), dtype=int)
    position = 0
    for _, dates in frame.groupby("customer_id")[config.features.date_column]:
        values = dates.to_numpy()
        first_in_window = np.searchsorted(values, values - window.to_timedelta64(), side="left")
        observed[position : position + len(values)] = np.arange(len(values)) - first_in_window + 1
        position += len(values)

    stated = frame["order_frequency_12m"].to_numpy()
    mismatch_rate = float((observed != stated).mean())
    median_gap = float(np.median(np.abs(observed.astype(float) - stated)))
    metrics = {"mismatch_rate": mismatch_rate, "median_absolute_gap": median_gap}

    if mismatch_rate <= 0.01:
        return CheckResult(
            "order_frequency_consistency",
            "PASS",
            "matches observed trailing-12-month counts",
            metrics,
        )
    return CheckResult(
        "order_frequency_consistency",
        "WARN",
        f"{mismatch_rate:.1%} of rows disagree with the observed trailing-12-month order count "
        f"(median gap {median_gap:.0f} orders) - the field is not derivable from this history, "
        "so CLV uses observed frequency instead",
        metrics,
    )


def _check_business_logic(df: pd.DataFrame, config: Config) -> CheckResult:
    """The documented effect directions should be visible in the raw rates."""
    target = config.features.target
    findings: list[str] = []
    status: Status = "PASS"

    if "payment_method" in df.columns:
        rates = df.groupby("payment_method")[target].mean()
        if "Cash_on_Delivery" in rates.index:
            cod = rates["Cash_on_Delivery"]
            others = df.loc[df["payment_method"] != "Cash_on_Delivery", target].mean()
            ok = cod > others
            findings.append(
                f"COD {cod:.1%} vs other {others:.1%} ({'as expected' if ok else 'UNEXPECTED'})"
            )
            status = status if ok else "WARN"

    if "customer_segment" in df.columns:
        rates = df.groupby("customer_segment")[target].mean()
        needed = {"First_Time", "Repeat", "Wholesale"}
        if needed <= set(rates.index):
            ordered = rates["First_Time"] > rates["Repeat"] > rates["Wholesale"]
            findings.append(
                f"segments First_Time {rates['First_Time']:.1%} > Repeat {rates['Repeat']:.1%} > "
                f"Wholesale {rates['Wholesale']:.1%} "
                f"({'confirmed' if ordered else 'NOT confirmed'})"
            )
            status = status if ordered else "WARN"

    return CheckResult(
        "business_logic", status, "; ".join(findings) or "no directional checks applicable"
    )


def _check_dgp_calibration(df: pd.DataFrame, config: Config) -> CheckResult:
    """Realised return rate should track the DGP probability that produced it."""
    try:
        expected = float(true_return_probability(df, config).mean())
    except (KeyError, ValueError) as exc:
        return CheckResult("dgp_calibration", "WARN", f"cannot recover DGP probability: {exc}")
    actual = float(df[config.features.target].mean())
    gap = abs(actual - expected)
    status: Status = "PASS" if gap < 0.02 else "WARN"
    return CheckResult(
        "dgp_calibration",
        status,
        f"observed {actual:.2%} vs DGP expectation {expected:.2%} (gap {gap:.2%})",
        {"observed_rate": actual, "expected_rate": expected, "gap": gap},
    )


CHECKS: tuple[Callable[[pd.DataFrame, Config], CheckResult], ...] = (
    _check_schema,
    _check_missing_values,
    _check_duplicate_ids,
    _check_feature_bounds,
    _check_target,
    _check_class_balance,
    _check_customer_invariants,
    _check_tenure_consistency,
    _check_order_frequency_consistency,
    _check_business_logic,
    _check_dgp_calibration,
)


def validate_dataset(df: pd.DataFrame, config: Config) -> ValidationReport:
    """Run every check against ``df``.

    Args:
        df: Transaction-level dataset.
        config: Supplies the expected schema, bounds and DGP parameters.

    Returns:
        A :class:`ValidationReport`. ``report.passed`` is False only if a check
        returned ``FAIL``; warnings are recorded for discussion.
    """
    report = ValidationReport()
    schema = _check_schema(df, config)
    report.add(schema)
    if schema.status == "FAIL":
        logger.error("schema check failed, skipping remaining checks")
        return report

    for check in CHECKS[1:]:
        report.add(check(df, config))

    dates = pd.to_datetime(df[config.features.date_column])
    report.summary = {
        "n_transactions": int(len(df)),
        "n_customers": int(df["customer_id"].nunique()),
        "return_rate": float(df[config.features.target].mean()),
        "mean_order_value_gbp": float(df["order_value_gbp"].mean()),
        "total_revenue_gbp": float(df["order_value_gbp"].sum()),
        "date_min": dates.min().date().isoformat(),
        "date_max": dates.max().date().isoformat(),
        "n_warnings": len(report.warnings),
        "n_failures": len(report.failures),
    }
    return report
