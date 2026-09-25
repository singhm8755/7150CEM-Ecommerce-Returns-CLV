# Data dictionary

`data/synthetic_ecommerce.csv` — 120,000 transactions from 12,000 customers,
1 January 2024 to 30 December 2025. One row per transaction.

## Columns

| Column | Type | Range / values | Description |
| --- | --- | --- | --- |
| `transaction_id` | int | 1 … n | Unique transaction identifier. |
| `customer_id` | int | 1 … 12,000 | Customer the order belongs to. |
| `transaction_date` | date | 2024-01-01 … 2025-12-30 | Order date. Drives the temporal split. |
| `product_category` | categorical | `Fashion`, `Electronics`, `Home_Garden` | Product category. The strongest single driver of the base return rate. |
| `payment_method` | categorical | `Credit_Card`, `PayPal`, `Cash_on_Delivery` | Payment method chosen at checkout. |
| `device_type` | categorical | `Mobile`, `Desktop`, `Tablet` | Device used to place the order. |
| `customer_segment` | categorical | `First_Time`, `Repeat`, `Wholesale` | Customer type. |
| `customer_tenure_days` | int | ≥ 0 | Days since the customer's first purchase. |
| `order_frequency_12m` | int | ≥ 1 | Orders placed by the customer in the preceding 12 months. |
| `order_value_gbp` | float | 10.00 … 1000.00 | Order value in pounds. Log-normal by construction. |
| `click_depth` | int | 1 … 10 | Pages viewed before purchase. |
| `time_on_page_seconds` | int | 5 … 300 | Dwell time on the product page. |
| `product_page_visits` | int | 1 … 20 | Times the customer viewed this product. |
| `returned` | int | 0 or 1 | **Target.** 1 if the order was returned. |

## Target

Overall return rate **29.5%** — a 2.4:1 majority-to-minority ratio, moderate
imbalance rather than the extreme imbalance that needs specialist handling.

| Dimension | Return rate |
| --- | --- |
| Fashion | 38.7% |
| Home_Garden | 25.6% |
| Electronics | 19.6% |
| First_Time | 37.6% |
| Repeat | 27.2% |
| Wholesale | 7.8% |
| Cash on delivery | 41.1% |
| All other payment methods | 28.1% |

## Data-generating process

The dataset is synthetic, and the process behind it is documented rather than
hidden. Each transaction's probability of return starts at its category's base
rate and is adjusted multiplicatively:

| Condition | Multiplier |
| --- | --- |
| Base rate: Fashion / Electronics / Home_Garden | 0.30 / 0.15 / 0.20 |
| `customer_segment == First_Time` | ×1.40 |
| `customer_segment == Wholesale` | ×0.30 |
| `payment_method == Cash_on_Delivery` | ×1.50 |
| `device_type == Mobile` | ×1.10 |
| `click_depth <= 3` | ×1.20 |
| `time_on_page_seconds < 60` | ×1.15 |
| Cap | 0.70 |

The outcome is then a Bernoulli draw from that probability. Because every input
is an observed column, `returns_clv.data.generate.true_return_probability`
recovers the exact probability behind any row — which is what makes the
**achievable performance ceiling** in `docs/model_card.md` computable rather
than guessed.

Two features are *not* in this list — `order_value_gbp` and
`product_page_visits` do not affect the probability at all. They are in the data
because a real dataset contains columns that turn out not to matter, and a
useful model has to discover that rather than be told.

## Known defects in the supplied dataset

`returns-clv validate` reports two warnings against the committed CSV. Both are
artefacts of how the original coursework notebook generated it, and both are
handled explicitly downstream rather than ignored.

| Defect | Detail | How the pipeline handles it |
| --- | --- | --- |
| Static tenure | `customer_tenure_days` is constant across all 11,993 repeat customers' transactions — stamped once per customer instead of measured at order time. | Kept as a feature (it still separates customer types) but not treated as a time-varying signal. |
| Inconsistent frequency | `order_frequency_12m` disagrees with the observed transaction history on 93.5% of rows, median gap 5 orders. First-time buyers carry a stated frequency of 1 while averaging ten orders on file. | Kept as a model feature, since the model can only use what it is given. **Excluded from the CLV calculation**, which uses each customer's observed orders per active year instead. |

`returns-clv generate` produces a dataset free of both: purchase intensity is
assigned once per customer, and `order_frequency_12m` and
`customer_tenure_days` are computed as of each transaction date from that
customer's own prior history.
