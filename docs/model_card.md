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
| Result | Value | Context |
| --- | --- | --- |
| Test ROC-AUC | **0.677** | against an achievable ceiling of 0.678 |
| Share of achievable signal captured | **99.4%** | the remaining gap is irreducible noise |
| Average precision | 0.450 | no-skill baseline 0.297 |
| Brier score | 0.192 | calibrated, so the probabilities can be multiplied by money |
| Best intervention policy | **£15,899** | Model, value-aware rule, over the 18,121-order test period |
| Same intervention without a model | -£21,331 | intervening on every order destroys value |
| Portfolio lifetime value | £3,010,819 | 12,000 customers, 3-year horizon |
<!-- END:headline -->

### Candidate comparison

<!-- BEGIN:models -->
| Model | CV average precision | Test ROC-AUC | Test average precision | Fit time |
| --- | --- | --- | --- | --- |
| Majority-class baseline | — | 0.5000 | 0.2969 | 0s |
| **Logistic Regression** | 0.4635 | 0.6779 | 0.4617 | 23s |
| Random Forest | 0.4539 | 0.6693 | 0.4464 | 248s |
| XGBoost | 0.4537 | 0.6682 | 0.4480 | 42s |
| _Achievable ceiling_ | — | _0.6784_ | _0.4578_ | — |
<!-- END:models -->

The ceiling row is estimated on the same finite test sample, so a model can
land marginally above it by chance — as logistic regression does on average
precision. That is noise in the estimate rather than a model beating
information theory.

All candidates land within a narrow band of each other, and of the achievable
ceiling. That is the finding, not a disappointment: the data's signal is close
to exhausted, so gradient boosting buys nothing over logistic regression and the
simplest model is preferred on interpretability, latency and maintenance
grounds.

### What the model is worth

The model exists to support a decision, so it is scored in pounds against the
alternatives a business would otherwise choose.

<!-- BEGIN:policies -->
| Policy | Orders flagged | Intervention spend | Gross saving | **Net** |
| --- | ---: | ---: | ---: | ---: |
| Model, value-aware rule | 3,915 (22%) | £18,596 | £34,496 | **£15,899** |
| Model, global threshold 0.38 | 4,068 (22%) | £19,323 | £23,200 | **£3,877** |
| Do nothing | 0 (0%) | £0 | £0 | **£0** |
| Rule of thumb (no model) | 8,332 (46%) | £39,577 | £37,972 | **-£1,605** |
| Intervene on every order | 18,121 (100%) | £86,075 | £64,744 | **-£21,331** |
<!-- END:policies -->

The value-aware rule flags an order when
`P(return) × effectiveness × cost_of_return > intervention_cost`. Because the
saving scales with order value and the intervention does not, it intervenes on
fewer orders than a global threshold and saves more.

## Drivers

<!-- BEGIN:features -->
| Feature | Mean absolute SHAP | Pushes |
| --- | ---: | --- |
| `customer_segment_First_Time` | 0.3629 | towards a return |
| `product_category_Electronics` | 0.2435 | towards keeping |
| `product_category_Fashion` | 0.2382 | towards a return |
| `customer_segment_Wholesale` | 0.2095 | towards keeping |
| `customer_segment_Repeat` | 0.1354 | towards a return |
| `is_low_click_depth` | 0.1300 | towards a return |
| `payment_method_PayPal` | 0.1246 | towards keeping |
| `payment_method_Credit_Card` | 0.1243 | towards keeping |
| `device_type_Desktop` | 0.0687 | towards keeping |
| `product_category_Home_Garden` | 0.0643 | towards keeping |
<!-- END:features -->

These match the documented data-generating process, which is the check that
matters: a model whose attributions contradicted the known process would be
fitting noise regardless of its metrics.

## Customer lifetime value

<!-- BEGIN:clv_segments -->
| Risk band | Customers | Mean predicted return rate | Actual return rate | Mean CLV |
| --- | ---: | ---: | ---: | ---: |
| Low Risk | 2,835 | 16.7% | 17.2% | £310 |
| Medium Risk | 7,802 | 31.6% | 31.6% | £243 |
| High Risk | 1,363 | 43.1% | 42.9% | £175 |
<!-- END:clv_segments -->

Predicted and actual return rates track each other to within half a percentage
point in every band, which is what makes the bands usable for targeting. The
band edges are derived from the economics: 0.40 is the probability at which an
intervention breaks even on an order of average value.

Note that customer-level targeting is worth substantially less than order-level
targeting — averaging a customer's risk discards the order-to-order variation
that the intervention decision depends on. The customer view supports retention
and segmentation; the order view supports intervention.

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
