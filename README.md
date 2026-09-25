# Predicting e-commerce returns, and what it's worth

[![CI](https://github.com/singhm8755/7150CEM-Ecommerce-Returns-CLV/actions/workflows/ci.yml/badge.svg)](https://github.com/singhm8755/7150CEM-Ecommerce-Returns-CLV/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Returns are the quiet margin killer in online retail: the sale is recorded, the
stock comes back, and the profit is gone. This project predicts, at checkout,
which orders will be returned — then answers the question that actually decides
whether the model gets deployed: **is acting on that prediction worth more than
it costs?**

An end-to-end pipeline takes raw transactions through validation, exploratory
analysis, model training, customer lifetime value and explainability, and ships
the result as a REST API, a Docker image and an interactive dashboard.

> Coventry University 7150CEM data science project. The five coursework
> notebooks are preserved in [`notebooks/`](notebooks/); everything else is a
> production reimplementation of them, and
> [`docs/methodology.md`](docs/methodology.md) documents every place the two
> differ and why.

---

## Headline results

<!-- BEGIN:headline -->
_Run `make all` to populate this table._
<!-- END:headline -->

The second row is the one worth pausing on. The project proposal targeted
ROC-AUC ≥ 0.80. That target was never achievable — because this dataset is
synthetic with a documented generating process, the exact probability behind
every outcome is recoverable, and scoring *that* gives the best result any model
could possibly reach. On this data, that ceiling is ≈ 0.68. The remaining gap to
1.0 is a coin flip, not a modelling failure.

So "ROC-AUC 0.68, short of the 0.80 target" and "captures 99% of the available
signal" describe the same model. Only the second tells you anything.
[Section 8 of the methodology](docs/methodology.md#8-report-against-the-achievable-ceiling)
covers what to do when no such ceiling is computable.

---

## The commercial argument

A return-risk score is only useful if acting on it beats not acting on it. The
pipeline models the money directly:

```
cost of a return     = margin forgone + two-way logistics + restocking loss
value of intervening = P(return) × effectiveness × cost of return − intervention cost
```

Intervening pays when the expected saving clears its cost. Since the saving
scales with order value and the intervention costs the same either way, **the
optimal rule is value-aware rather than a single probability cut-off**: a 40%
risk on a £900 order is worth acting on, the same 40% on a £15 order is not.

Measured on the held-out test period, against the alternatives a retailer would
otherwise pick:

<!-- BEGIN:policies -->
_Run `make all` to populate this table._
<!-- END:policies -->

Intervening on everything loses money. A plausible hand-written rule — flag cash
on delivery and first-time buyers — also loses money. Targeting by expected
value is what turns the model into profit.

![Policy comparison](outputs/figures/07_policy_impact.png)

---

## What makes this more than a notebook

The original coursework produced respectable-looking metrics. Rebuilding it
surfaced five methodological problems that were inflating them, and two defects
in the dataset itself. Each fix is measured rather than asserted — run
`make experiments` to reproduce the numbers.

| Problem found | Why it inflates results | Fix |
| --- | --- | --- |
| SMOTE applied before cross-validation | Synthetic points interpolate across fold boundaries, so folds are scored on echoes of their own data | Resampling is a step inside an `imblearn` pipeline, applied per fold |
| Scaler fitted on all data before splitting | Test-set statistics inform the training transform | All preprocessing fitted inside the estimator |
| Decision threshold tuned on the test set | Reports the best cut-off in hindsight, which nobody could pick in advance | Threshold chosen on a held-out validation half |
| `LabelEncoder` on nominal categories | Asserts `Credit_Card < PayPal`, an ordering that does not exist | One-hot encoding, unknown-safe for serving |
| CLV scored in-sample | Customer risk scores are part memory, not prediction | Out-of-fold scoring, with the earliest block backcast |
| `order_frequency_12m` contradicts the transaction history on **93.5%** of rows | CLV multiplies by frequency, so the error propagates straight into pounds | Validation flags it; CLV uses observed orders per active year |
| `customer_tenure_days` constant within each customer | Stamped once per customer rather than measured at order time | Flagged by validation; the regenerated dataset computes it causally |

### Measured: what resampling before cross-validation costs you

<!-- BEGIN:experiment_leakage -->
_Run `make all` to populate this table._
<!-- END:experiment_leakage -->

The inflated cross-validation score is the dangerous part. The test score barely
moves, so nothing looks wrong — the model simply under-performs its own
documentation once deployed.

### Measured: what tuning a threshold on the test set costs you

<!-- BEGIN:experiment_threshold -->
_Run `make all` to populate this table._
<!-- END:experiment_threshold -->

### Measured: whether the engineered features earned their place

<!-- BEGIN:experiment_ablation -->
_Run `make all` to populate this table._
<!-- END:experiment_ablation -->

The answer depends on the model class, which is why the ablation runs across all
of them. A tree ensemble discovers a threshold effect like `click_depth <= 3` by
splitting on the raw column, so the indicator adds little; a linear model cannot
represent it at all and has to be handed it. Reporting only the tree result
would have made the features look worthless.

---

## Model comparison

<!-- BEGIN:models -->
_Run `make all` to populate this table._
<!-- END:models -->

Every candidate sits within a hair of the ceiling and of each other, so the
simplest model wins on interpretability, latency and maintenance cost. Knowing
*why* they converge — rather than reaching for a bigger model — is the point.

![ROC curves](outputs/figures/01_roc_curves.png)

Probability calibration is treated as a first-class requirement, not a
footnote: the CLV model multiplies these probabilities by pound amounts, and
ROC-AUC is completely blind to a model that ranks well but is systematically
overconfident.

![Calibration](outputs/figures/03_calibration.png)

---

## Customer lifetime value

Each customer's discounted three-year contribution, net of the returns the model
expects them to make:

```
CLV = Σ  annual contribution × retention^(t−1) / (1 + discount)^t
```

<!-- BEGIN:clv_segments -->
_Run `make all` to populate this table._
<!-- END:clv_segments -->

Predicted and actual return rates track each other within every band, which is
what makes the bands usable for targeting. The dashboard exposes the unit
economics as sliders, because an assumption the reader can move is more honest
than one buried in a config file.

![CLV by risk band](outputs/figures/08_clv_by_risk_segment.png)

---

## What drives a prediction

<!-- BEGIN:features -->
_Run `make all` to populate this table._
<!-- END:features -->

These match the documented data-generating process — the check that matters,
since attributions contradicting the known process would mean the model was
fitting noise whatever its metrics said. Every individual prediction is
explainable too, via `POST /explain`.

![Feature importance](outputs/figures/11_feature_importance.png)

---

## Quick start

```bash
git clone https://github.com/singhm8755/7150CEM-Ecommerce-Returns-CLV.git
cd 7150CEM-Ecommerce-Returns-CLV

make setup          # virtualenv + install
make all            # full pipeline: validate -> eda -> train -> clv -> explain
make api            # scoring API at http://localhost:8000/docs
make dashboard      # dashboard at http://localhost:8501
```

`make help` lists every target. The full pipeline takes about seven minutes on
four cores.

### Scoring a transaction

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '[{"product_category":"Fashion","payment_method":"Cash_on_Delivery",
        "device_type":"Mobile","customer_segment":"First_Time",
        "order_value_gbp":129.99,"click_depth":2,"time_on_page_seconds":35,
        "product_page_visits":3,"customer_tenure_days":12,"order_frequency_12m":1}]'
```

```json
{
  "model_name": "Logistic Regression",
  "predictions": [{
    "return_probability": 0.6012,
    "risk_band": "High_Risk",
    "flagged_by_expected_value": true,
    "expected_return_cost_gbp": 27.53,
    "expected_intervention_saving_gbp": 2.11,
    "recommended_action": "intervene: expected saving exceeds the intervention cost"
  }]
}
```

Every response carries both decisions — the global threshold and the
expected-value rule — plus the pounds behind the recommendation, so the caller
can apply whichever matches their operating policy.

### In Docker

```bash
make docker-build
make docker-run     # mounts ./models read-only; the image carries no data
```

---

## How it is put together

```mermaid
flowchart TB
    CSV[("data/synthetic_ecommerce.csv")] --> V

    subgraph PIPE["returns-clv pipeline"]
      direction LR
      V["validate<br/>11 checks"] --> E["eda"]
      E --> T["train<br/>search, select,<br/>calibrate, threshold"]
      T --> C["clv<br/>out-of-fold scoring"]
      C --> X["explain<br/>SHAP"]
    end

    CFG[["configs/default.yaml"]] -.drives.-> PIPE

    T -->|ModelBundle| M[("models/return_risk_model.joblib")]
    C --> R[("outputs/ reports + figures")]
    X --> R

    M --> API["FastAPI<br/>/predict  /explain"]
    M --> DASH["Streamlit dashboard"]
    R --> DASH
    R --> DOCS["README + model card<br/>via scripts/update_results.py"]
```

The bundle is the seam: training writes it, and the API, the dashboard and the
CLV stage all read it. Because it carries the fitted preprocessing as well as
the model, there is only ever one implementation of the feature logic.

```
├── src/returns_clv/          the pipeline, as an installable package
│   ├── config.py             typed configuration, validated on load
│   ├── data/
│   │   ├── generate.py       synthetic data with a documented generating process
│   │   └── validate.py       11 checks, PASS / WARN / FAIL
│   ├── features.py           engineering + preprocessing, inside the estimator
│   ├── splits.py             temporal (default), random and grouped protocols
│   ├── models.py             model zoo; resampling lives inside the pipeline
│   ├── train.py              search -> select -> calibrate -> threshold -> evaluate
│   ├── evaluate.py           metrics, thresholds and the money model
│   ├── clv.py                out-of-sample scoring, lifetime value, policy simulation
│   ├── explain.py            SHAP, global and per-prediction
│   ├── experiments.py        measures what each methodology fix is worth
│   ├── plots.py              one house style, colour-vision-safe
│   └── cli.py                `returns-clv <stage>`
├── api/main.py               FastAPI service
├── dashboard/app.py          Streamlit dashboard
├── notebooks/                the original coursework, de-Colab'd
├── configs/default.yaml      every parameter the pipeline depends on
├── tests/                    133 tests
└── docs/                     methodology, model card, data dictionary
```

Three properties hold throughout:

**Configuration over code.** Every distribution, effect size, split proportion
and pound figure lives in `configs/default.yaml` and is loaded into validated
dataclasses. An experiment is a config change.

**One code path from input to decision.** The saved artefact bundles the fitted
pipeline, the operating threshold, the expected input columns and its own
provenance. Training, the API, the dashboard and the CLV stage all load the same
bundle, so there is no second implementation of the feature logic to drift.

**Documentation generated from artefacts.** Every results table above is written
by `scripts/update_results.py` from the JSON in `outputs/reports/`. A stale
number is worse than no number, because nothing signals that it is wrong.

---

## Testing and CI

```bash
make test        # 133 tests
make test-fast   # skips the model-training tests
make lint        # ruff check + format
```

The tests cover behaviour rather than implementation — that a temporal split
never trains on the future, that a grouped split leaks no customer, that the
feature engineer produces identical values on a subset and the whole (so it
cannot leak), that CLV falls as return risk rises, that the API rejects
out-of-domain input, and that **no model ever exceeds the information-theoretic
ceiling**, which would be the signature of a leak.

CI runs lint, the suite on Python 3.10–3.12, a full pipeline smoke run on a
reduced budget, and a Docker build with a health check.

---

## Data

120,000 synthetic transactions from 12,000 customers over 24 months, 29.5%
returned. Schema, generating process and known defects are in
[`docs/data_dictionary.md`](docs/data_dictionary.md).

Validation against the committed dataset:

<!-- BEGIN:validation -->
_Run `make all` to populate this table._
<!-- END:validation -->

Two warnings, both genuine. A validation suite that passes everything is not
evidence that the data is clean — only that the suite is not looking.

`returns-clv generate` produces a dataset free of both defects, with history
features computed as of each transaction date.

---

## Reading further

| Document | What it covers |
| --- | --- |
| [`docs/methodology.md`](docs/methodology.md) | Every methodological decision, with measurements |
| [`docs/model_card.md`](docs/model_card.md) | Performance, limitations, ethical considerations |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | Schema, generating process, known defects |
| [`notebooks/`](notebooks/) | The original coursework analysis |

## Limitations

The data is synthetic, so the coefficients transfer to nothing — the pipeline,
evaluation protocol and serving path are the deliverable. The unit economics are
plausible assumptions rather than measurements, and every pound figure inherits
that uncertainty. `intervention_effectiveness = 0.25` in particular would need
an A/B test to establish. See the
[model card](docs/model_card.md#limitations) for the full list.

## Licence

MIT.
