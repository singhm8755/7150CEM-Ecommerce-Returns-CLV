"""Synthetic e-commerce transaction generator.

The generator encodes an explicit, documented data-generating process (DGP): a
base return rate per product category, modified multiplicatively by customer
segment, payment method, device and two browsing-behaviour signals. Because the
DGP is known, :func:`true_return_probability` can recover the exact probability
behind every row, which the evaluation stage uses to compute the *Bayes-optimal
ceiling* — the best AUC any model could achieve on this data.

Two properties distinguish this implementation from the original coursework
notebook (see ``docs/methodology.md``):

1. **Customer-level fields are consistent.** A customer has one first-purchase
   date and one purchase intensity; the original drew ``order_frequency_12m``
   independently per transaction, so 60% of customers carried contradictory
   values.
2. **History features are computed as of transaction time.** ``order_frequency_12m``
   counts a customer's orders in the 365 days *preceding* each transaction, and
   ``customer_tenure_days`` is measured from their first purchase. Neither can
   see the future, so a temporal train/test split stays honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from returns_clv.config import Config
from returns_clv.logging_utils import get_logger

logger = get_logger(__name__)

# Behavioural feature bounds, inclusive. Kept as module constants so the
# validation stage can assert against exactly the same numbers.
CLICK_DEPTH_RANGE = (1, 10)
DWELL_SECONDS_RANGE = (5, 300)
PAGE_VISITS_RANGE = (1, 20)
ORDER_VALUE_RANGE = (10.0, 1000.0)
ORDER_VALUE_LOGNORMAL = (4.5, 0.8)  # mean, sigma of the underlying normal

# Relative purchase intensity by segment, later rescaled to hit the configured
# transaction volume. Wholesale accounts buy far more often than one-off buyers.
SEGMENT_INTENSITY_RANGE = {
    "First_Time": (1, 2),
    "Repeat": (3, 20),
    "Wholesale": (20, 50),
}

# Days before the observation window that a customer first purchased. A negative
# range means "joined during the window", which is how first-time buyers arise.
SEGMENT_FIRST_PURCHASE_OFFSET_DAYS = {
    "First_Time": (0, None),  # uniform inside the window
    "Repeat": (-730, -90),
    "Wholesale": (-1095, -365),
}

TRAILING_WINDOW_DAYS = 365

OUTPUT_COLUMNS = [
    "transaction_id",
    "customer_id",
    "transaction_date",
    "product_category",
    "payment_method",
    "device_type",
    "customer_segment",
    "customer_tenure_days",
    "order_frequency_12m",
    "order_value_gbp",
    "click_depth",
    "time_on_page_seconds",
    "product_page_visits",
    "returned",
]


def _categories_and_probs(mix: dict[str, float]) -> tuple[list[str], np.ndarray]:
    """Split a ``{label: weight}`` mapping into aligned labels and probabilities."""
    labels = list(mix)
    probs = np.asarray([mix[label] for label in labels], dtype=float)
    total = probs.sum()
    if not np.isclose(total, 1.0):
        logger.warning("mixture weights sum to %.4f, normalising", total)
    return labels, probs / total


def true_return_probability(df: pd.DataFrame, config: Config) -> pd.Series:
    """Recover the DGP return probability for each row.

    Every input to the probability is an observed column, so this works on any
    dataset produced by this generator — including the original coursework CSV.
    The result is the irreducible signal: a model cannot beat it, and the gap
    between a model's AUC and the AUC of this series quantifies how much
    learnable structure is left on the table.

    Args:
        df: Transactions containing the categorical and behavioural columns.
        config: Supplies base rates, effect multipliers and the probability cap.

    Returns:
        Probability of return per row, indexed like ``df``.
    """
    gen = config.data_generation
    effects = gen.effects

    missing = {
        "product_category",
        "customer_segment",
        "payment_method",
        "device_type",
        "click_depth",
        "time_on_page_seconds",
    } - set(df.columns)
    if missing:
        raise KeyError(f"cannot recover true probability, missing columns: {sorted(missing)}")

    prob = df["product_category"].map(gen.category_base_return_rate)
    if prob.isna().any():
        unknown = sorted(df.loc[prob.isna(), "product_category"].unique())
        raise ValueError(f"no base return rate configured for categories {unknown}")
    prob = prob.astype(float)

    prob = prob.where(df["customer_segment"] != "First_Time", prob * effects.segment_First_Time)
    prob = prob.where(df["customer_segment"] != "Wholesale", prob * effects.segment_Wholesale)
    prob = prob.where(
        df["payment_method"] != "Cash_on_Delivery", prob * effects.payment_Cash_on_Delivery
    )
    prob = prob.where(df["device_type"] != "Mobile", prob * effects.device_Mobile)
    prob = prob.where(df["click_depth"] > 3, prob * effects.low_click_depth)
    prob = prob.where(df["time_on_page_seconds"] >= 60, prob * effects.short_dwell)

    return prob.clip(upper=gen.max_return_probability).rename("true_return_prob")


def _draw_customers(config: Config, rng: np.random.Generator) -> pd.DataFrame:
    """Build the customer table: segment, first purchase date, purchase intensity."""
    gen = config.data_generation
    start = pd.Timestamp(gen.start_date)
    end = pd.Timestamp(gen.end_date)
    window_days = int((end - start).days)

    segment_labels, segment_probs = _categories_and_probs(gen.segment_mix)
    segments = rng.choice(segment_labels, size=gen.n_customers, p=segment_probs)

    intensity = np.empty(gen.n_customers, dtype=float)
    first_offset = np.empty(gen.n_customers, dtype=int)

    for segment in segment_labels:
        mask = segments == segment
        n = int(mask.sum())
        if n == 0:
            continue
        lo, hi = SEGMENT_INTENSITY_RANGE.get(segment, (1, 5))
        intensity[mask] = rng.integers(lo, hi + 1, size=n)

        offset_lo, offset_hi = SEGMENT_FIRST_PURCHASE_OFFSET_DAYS.get(segment, (-365, 0))
        if offset_hi is None:  # joined during the observation window
            first_offset[mask] = rng.integers(0, window_days + 1, size=n)
        else:
            first_offset[mask] = rng.integers(offset_lo, offset_hi + 1, size=n)

    first_purchase = start + pd.to_timedelta(first_offset, unit="D")
    # Days the customer is observable inside the window.
    active_start = first_purchase.where(first_purchase > start, start)
    active_days = (end - active_start).days.to_numpy().clip(min=1)

    return pd.DataFrame(
        {
            "customer_id": np.arange(1, gen.n_customers + 1),
            "customer_segment": segments,
            "first_purchase_date": first_purchase,
            "active_start": active_start,
            "active_days": active_days,
            "intensity": intensity,
        }
    )


def _allocate_transaction_counts(
    customers: pd.DataFrame, target_total: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw per-customer transaction counts summing exactly to ``target_total``.

    Counts are Poisson draws around each customer's intensity scaled to the
    configured volume, then nudged to hit the target exactly so the dataset size
    is deterministic.
    """
    active_years = customers["active_days"].to_numpy() / 365.0
    weights = customers["intensity"].to_numpy() * active_years
    scale = target_total / weights.sum()
    expected = weights * scale

    counts = rng.poisson(expected).astype(int)

    # Give every customer at least one transaction, but only when the requested
    # volume leaves room for it: asking for fewer transactions than customers
    # (as a fast test might) must not make the target unreachable.
    floor = 1 if target_total >= len(counts) else 0
    if floor:
        counts = np.maximum(counts, 1)

    # Reconcile to the exact target by adding to / removing from customers in
    # proportion to their intensity.
    probs = expected / expected.sum()
    delta = target_total - int(counts.sum())
    if delta > 0:
        extra = rng.choice(len(counts), size=delta, p=probs)
        np.add.at(counts, extra, 1)

    while delta < 0:
        removable = np.flatnonzero(counts > floor)
        if removable.size == 0:
            # Every remaining customer sits on the floor. Drop the floor so the
            # loop converges instead of spinning forever.
            if floor == 0:
                break
            floor = 0
            continue
        take = min(-delta, removable.size)
        chosen = rng.choice(removable, size=take, replace=False)
        counts[chosen] -= 1
        delta += take

    return counts


def _trailing_order_counts(customer_ids: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Orders placed by the same customer in the 365 days before each transaction.

    Computed with a per-customer ``searchsorted`` on sorted dates, so the value
    only ever depends on the past.
    """
    order = np.lexsort((dates, customer_ids))
    sorted_ids = customer_ids[order]
    sorted_dates = dates[order]
    result = np.zeros(len(dates), dtype=int)

    boundaries = np.flatnonzero(np.diff(sorted_ids)) + 1
    for group_dates, positions in zip(
        np.split(sorted_dates, boundaries),
        np.split(np.arange(len(dates)), boundaries),
        strict=True,
    ):
        window_start = group_dates - np.timedelta64(TRAILING_WINDOW_DAYS, "D")
        # Number of earlier orders inside the trailing window: index of the
        # current order minus the first order still within the window.
        current_index = np.arange(len(group_dates))
        first_in_window = np.searchsorted(group_dates, window_start, side="left")
        result[order[positions]] = current_index - first_in_window
    return result


def generate_dataset(
    config: Config,
    *,
    seed: int | None = None,
    n_transactions: int | None = None,
) -> pd.DataFrame:
    """Generate a synthetic transaction dataset.

    Args:
        config: Pipeline configuration; ``config.data_generation`` supplies every
            distribution and effect size.
        seed: Overrides ``config.project.random_seed``.
        n_transactions: Overrides the configured transaction volume. Useful for
            fast tests.

    Returns:
        Transactions sorted by date, with columns in :data:`OUTPUT_COLUMNS`.
    """
    gen = config.data_generation
    rng = np.random.default_rng(config.seed if seed is None else seed)
    target_total = int(n_transactions or gen.n_transactions)

    logger.info(
        "generating %s transactions for %s customers (%s to %s)",
        f"{target_total:,}",
        f"{gen.n_customers:,}",
        gen.start_date,
        gen.end_date,
    )

    customers = _draw_customers(config, rng)
    counts = _allocate_transaction_counts(customers, target_total, rng)

    # Expand the customer table to one row per transaction.
    idx = np.repeat(np.arange(len(customers)), counts)
    customer_ids = customers["customer_id"].to_numpy()[idx]
    segments = customers["customer_segment"].to_numpy()[idx]
    first_purchase = customers["first_purchase_date"].to_numpy()[idx]
    active_start = customers["active_start"].to_numpy()[idx]
    active_days = customers["active_days"].to_numpy()[idx]

    # Transaction dates: uniform across each customer's observable window.
    day_offset = np.floor(rng.random(target_total) * active_days).astype(int)
    dates = active_start + day_offset.astype("timedelta64[D]")

    # Causally-computed history features.
    tenure_days = ((dates - first_purchase) / np.timedelta64(1, "D")).astype(int).clip(min=0)
    trailing_orders = _trailing_order_counts(customer_ids, dates)

    category_labels, category_probs = _categories_and_probs(gen.category_mix)
    payment_labels, payment_probs = _categories_and_probs(gen.payment_mix)
    device_labels, device_probs = _categories_and_probs(gen.device_mix)

    order_value = np.round(rng.lognormal(*ORDER_VALUE_LOGNORMAL, size=target_total), 2).clip(
        *ORDER_VALUE_RANGE
    )

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "transaction_date": dates,
            "product_category": rng.choice(category_labels, target_total, p=category_probs),
            "payment_method": rng.choice(payment_labels, target_total, p=payment_probs),
            "device_type": rng.choice(device_labels, target_total, p=device_probs),
            "customer_segment": segments,
            "customer_tenure_days": tenure_days,
            # A customer's trailing-12-month order count, by definition at least
            # their first order in that window.
            "order_frequency_12m": trailing_orders + 1,
            "order_value_gbp": order_value,
            "click_depth": rng.integers(
                CLICK_DEPTH_RANGE[0], CLICK_DEPTH_RANGE[1] + 1, target_total
            ),
            "time_on_page_seconds": rng.integers(
                DWELL_SECONDS_RANGE[0], DWELL_SECONDS_RANGE[1] + 1, target_total
            ),
            "product_page_visits": rng.integers(
                PAGE_VISITS_RANGE[0], PAGE_VISITS_RANGE[1] + 1, target_total
            ),
        }
    )

    probability = true_return_probability(df, config)
    df["returned"] = (rng.random(target_total) < probability.to_numpy()).astype(int)

    df = df.sort_values(["transaction_date", "customer_id"], kind="mergesort").reset_index(
        drop=True
    )
    df.insert(0, "transaction_id", np.arange(1, len(df) + 1))
    df["transaction_date"] = df["transaction_date"].dt.strftime("%Y-%m-%d")

    logger.info(
        "generated %s rows | return rate %.2f%% | mean order value GBP %.2f",
        f"{len(df):,}",
        100 * df["returned"].mean(),
        df["order_value_gbp"].mean(),
    )
    return df[OUTPUT_COLUMNS]
