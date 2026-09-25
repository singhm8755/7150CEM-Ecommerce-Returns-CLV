# Coursework notebooks

The five notebooks submitted for Coventry University 7150CEM, preserved with
their original analysis and recorded outputs.

| Notebook | Stage | Production equivalent |
| --- | --- | --- |
| `1_data_generation.ipynb` | Synthetic dataset generation | `returns-clv generate` |
| `2_data_validation.ipynb` | Data quality checks | `returns-clv validate` |
| `3_eda_analysis.ipynb` | Exploratory analysis | `returns-clv eda` |
| `4_model_training.ipynb` | Model training and evaluation | `returns-clv train` |
| `5_clv_analysis.ipynb` | Customer lifetime value | `returns-clv clv` |

## What changed

Only the plumbing. The Google Drive mounts are gone and paths now resolve
against the repository root, so the notebooks run locally. The analysis, the
code and the recorded outputs are as submitted.

## Why the pipeline differs from these notebooks

Rebuilding the analysis as a package surfaced several methodological problems —
resampling applied before cross-validation, a decision threshold tuned on the
test set, label encoding of nominal categories, in-sample scoring in the CLV
stage — along with two defects in the dataset itself.

The notebooks are kept unchanged as the record of the submitted work.
[`../docs/methodology.md`](../docs/methodology.md) documents each difference,
the reasoning, and what the fix is worth in measured terms.
