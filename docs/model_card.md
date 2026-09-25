# Model card — e-commerce return risk

> Results in this document are generated from `outputs/reports/` by
> `make results`. They reflect the most recent pipeline run.

## Overview

| | |
| --- | --- |
| **Task** | Binary classification: will this order be returned? |
| **Output** | A calibrated probability, plus a flag and a recommended action |
| **Unit of prediction** | One transaction, at the moment of checkout |
| **Training data** | 120,000 synthetic transactions, 12,000 customers, Jan 2024 – Dec 2025 |
| **Evaluation** | Temporal hold-out: trained on the earliest 70%, tested on the most recent 15% |
| **Intended use** | Prioritising pre-dispatch interventions and informing customer lifetime value |
| **Owner** | Coventry University 7150CEM project |

## Performance

<!-- BEGIN:headline -->
_Run `make all` to populate this table._
<!-- END:headline -->

### Candidate comparison

<!-- BEGIN:models -->
_Run `make all` to populate this table._
<!-- END:models -->

All candidates land within a narrow band of each other, and of the achievable
ceiling. That is the finding, not a disappointment: the data's signal is close
to exhausted, so gradient boosting buys nothing over logistic regression and the
simplest model is preferred on interpretability, latency and maintenance
grounds.

### What the model is worth

The model exists to support a decision, so it is scored in pounds against the
alternatives a business would otherwise choose.

<!-- BEGIN:policies -->
_Run `make all` to populate this table._
<!-- END:policies -->

The value-aware rule flags an order when
`P(return) × effectiveness × cost_of_return > intervention_cost`. Because the
saving scales with order value and the intervention does not, it intervenes on
fewer orders than a global threshold and saves more.

## Drivers

<!-- BEGIN:features -->
_Run `make all` to populate this table._
<!-- END:features -->

These match the documented data-generating process, which is the check that
matters: a model whose attributions contradicted the known process would be
fitting noise regardless of its metrics.

## Customer lifetime value

<!-- BEGIN:clv_segments -->
_Run `make all` to populate this table._
<!-- END:clv_segments -->

Predicted and actual return rates track each other closely within each band,
which is what makes the bands usable for targeting.

## Training details

| | |
| --- | --- |
| **Algorithm** | Selected per run from logistic regression, random forest and XGBoost |
| **Preprocessing** | One-hot encoding, standardisation, six engineered behavioural features — all inside the estimator |
| **Class imbalance** | SMOTE applied inside the cross-validation pipeline (29.5% positive, 2.4:1) |
| **Hyperparameter search** | Randomised search, expanding-window cross-validation, scored by average precision |
| **Calibration** | Isotonic regression on one half of the validation block |
| **Threshold** | Chosen on the other half, by expected net saving |
| **Seed** | 42, throughout |

## Limitations

**The data is synthetic.** Every return here is a draw from a documented
probability. Real return behaviour carries seasonality, fraud rings, sizing
inconsistencies between brands, and customers who learn a returns policy and
exploit it — none of which is in this dataset. The pipeline, evaluation
protocol and serving path transfer; the coefficients do not.

**The performance ceiling is low by construction.** Roughly a third of the
outcome is unexplainable from the recorded features, so no model can do much
better than the numbers above. On real data the ceiling is unknown, and the
substitute is a strong simple baseline plus an explicit irreducible-error
argument.

**The unit economics are assumptions.** Margin, logistics cost, intervention
cost and intervention effectiveness are plausible UK retail figures, not
measurements. Every pound figure inherits that uncertainty, which is why they
are configurable and exposed as dashboard sliders. In particular,
`intervention_effectiveness = 0.25` — the assumption that an intervention
prevents a quarter of the returns it targets — would need an A/B test to
establish.

**Two dataset defects are worked around, not fixed.** `customer_tenure_days`
is stamped per customer rather than measured at order time, and
`order_frequency_12m` disagrees with the transaction history on 93.5% of rows.
Both remain as model features because a model can only use what it is given;
the CLV calculation substitutes observed purchase frequency. See
`docs/data_dictionary.md`.

**Customers span the temporal split.** The default protocol holds out a time
period, not a set of customers, so a customer can appear in both training and
test. That matches deployment, where most orders come from customers already on
file. `--split grouped` answers the unseen-customer question separately.

## Ethical considerations

The model scores transactions, not people, and no protected attribute is used or
available. It nonetheless warrants care in deployment:

- **First-time buyers carry the highest predicted risk** and would be flagged
  most often. An intervention that adds friction for new customers could
  suppress acquisition — a cost that appears nowhere in the return-cost
  arithmetic. `intervention_cost_gbp` is set to a blended figure partly to
  represent it, but it is an estimate.
- **Cash on delivery correlates with return risk** and is used more by customers
  without access to credit. Restricting it on a model score risks an access
  effect along lines the model cannot see.
- **Interventions should reduce returns, not sales.** Better sizing guidance and
  clearer product imagery are the intended actions. Withholding free returns or
  blocking orders would shift the cost onto the customer and break the
  assumption behind the savings estimate.
- **Every score is explainable.** The `/explain` endpoint returns per-feature
  SHAP contributions, so a flagged order can always be justified.

## Maintenance

The deployed artefact records its training date, dataset fingerprint, library
versions and operating threshold (`GET /model`). Retraining should be triggered
when the monitored return rate drifts materially from the training-period rate,
when a new product category or payment method is introduced, or on a fixed
quarterly cadence — whichever comes first.
