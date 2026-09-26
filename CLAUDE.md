# CLAUDE.md

Guidance for working in this repository.

## What this is

Return-risk prediction and customer lifetime value for e-commerce, built for the
Coventry University 7150CEM project. `notebooks/` holds the original coursework;
everything else is a production reimplementation of it. `docs/methodology.md`
records every place the two differ and why — read it before changing modelling
code, because most of the non-obvious decisions are justified there.

## Commands

```bash
make setup          # virtualenv + editable install
make all            # full pipeline, then regenerate the documented results
make test           # full suite
make test-fast      # skips the model-training tests
make lint           # ruff check + format --check
make results        # regenerate README/docs tables from outputs/reports
returns-clv <stage> # generate | validate | eda | train | clv | explain | experiments | all
```

Run the suite as `pytest`, not `python -m pytest`. The latter silently adds the
working directory to `sys.path` and hides import problems that CI will catch.

## Conventions that matter

**Configuration over constants.** Every distribution, effect size, split
proportion and pound figure lives in `configs/default.yaml` and loads into frozen
dataclasses in `config.py`. Adding a parameter means adding it there, not
hardcoding it. Unknown keys are rejected on load.

**Preprocessing belongs inside the estimator.** Scalers, encoders and resamplers
are pipeline steps, never applied to a frame before splitting. This is the
project's central methodological fix; `experiments.py` measures what breaking it
costs. Do not "simplify" by moving a transform outside the pipeline.

**The test block fits nothing.** Hyperparameters come from the training block,
model selection and the threshold from validation, and the test block is scored
once. `ThresholdChoice.fitted_on` records the provenance so a report cannot imply
otherwise.

**One path from input to decision.** `ModelBundle` carries the fitted pipeline,
threshold, expected columns and provenance. Training writes it; the API, the
dashboard and the CLV stage all read it. Never reimplement feature logic at the
serving end.

**Results are generated, never typed.** Tables in `README.md` and `docs/` sit
between `<!-- BEGIN:key -->` markers and are written by
`scripts/update_results.py` from the JSON in `outputs/reports/`. Edit the
generator, not the Markdown. A test asserts every marker has a builder.

**Figures follow one house style.** `plots.py` defines it: categorical colour
assigned by series identity in fixed order, one y-axis per panel (never
dual-axis), single-hue sequential ramps, blue/red diverging with a neutral zero,
a legend whenever there are two or more series. The palette is checked for
colour-vision separation; do not substitute ad-hoc colours.

## Gotchas found the hard way

- `cross_val_predict` rejects expanding-window folds because they do not
  partition the data. `clv.py` loops over folds explicitly and backcasts the
  earliest block.
- SHAP needs a reference distribution. Explaining a single row against itself
  returns all zeros, which is why the bundle carries `explainer_background`.
- Out-of-fold scores must be calibrated the same way the deployed model is, or
  CLV multiplies two different probability scales by pounds.
- Averaging signed SHAP across a population inverts the apparent direction of a
  one-hot feature. Use the value/contribution correlation instead.
- `imblearn` pipelines reject nested `sklearn` pipelines as intermediate steps,
  so feature steps are spliced in flat via `features.feature_steps`.

## Testing

Tests assert behaviour, not implementation: that a temporal split never trains
on the future, that a grouped split leaks no customer, that the feature engineer
gives identical values on a subset and the whole, that CLV falls as risk rises,
and that no model exceeds the information-theoretic ceiling — which would be the
signature of a leak. Prefer adding a test in that style over asserting on
internals.
