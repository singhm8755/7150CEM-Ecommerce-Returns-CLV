# Methodology

This document records what the pipeline does differently from the original
coursework notebooks, and why. Each section states the problem, the fix, and —
where it can be measured — the size of the difference. Every number here is
reproducible with `returns-clv experiments`.

---

## 1. Resampling belongs inside cross-validation

**The problem.** The original notebook applied SMOTE to the whole training set
and then ran `GridSearchCV` over the resampled matrix:

```python
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
grid = GridSearchCV(model, params, cv=5, scoring='f1')
grid.fit(X_train_smote, y_train_smote)          # leakage
```

SMOTE creates a synthetic minority point by interpolating between a real
minority point and one of its nearest neighbours. Run before the folds are cut,
it manufactures points that lie between a row in fold 3 and its neighbours in
folds 1 and 2. When fold 3 is held out for scoring, the model has already
trained on interpolations of the very rows it is being tested on. The
cross-validated score goes up; the model does not get better.

The same argument applies to `StandardScaler`, which the original fitted on the
full dataset before splitting, letting test-set means and variances inform the
training transformation.

**The fix.** Everything that learns from data — the scaler, the encoder, the
resampler — is a step inside an `imblearn.pipeline.Pipeline`. `imblearn`'s
pipeline applies resamplers on `fit` and skips them on `transform`/`predict`, so
each fold resamples only its own training portion and is scored on untouched
data.

```python
ImbPipeline([
    ("engineer",   BehaviouralFeatureEngineer()),
    ("preprocess", ColumnTransformer(...)),      # fitted per fold
    ("resample",   SMOTE(random_state=42)),      # fit only, per fold
    ("classifier", LogisticRegression()),
])
```

**The measurement.** `returns-clv experiments` runs both protocols on the same
data with the same model and compares each one's cross-validated score against
its score on genuinely unseen test data. The quantity that matters is the
**optimism gap** — how much a protocol flatters itself:

<!-- BEGIN:experiment_leakage -->
| Protocol | Cross-validated average precision | Test average precision | Optimism gap |
| --- | ---: | ---: | ---: |
| Resampling inside cross-validation (correct) | 0.4173 | 0.4068 | **+0.0104** |
| Resampling before cross-validation (leaky) | 0.8626 | 0.4068 | **+0.4558** |

Resampling first inflates the reported cross-validated score by +0.4454 without improving the model.
<!-- END:experiment_leakage -->

A protocol you can trust reports roughly what it delivers. The leaky one does
not, and the danger is not the inflated number itself but that it is invisible:
the test score barely moves, so nothing looks wrong until the model reaches
production and under-performs its own documentation.

---

## 2. One-hot encoding, not label encoding

**The problem.** The notebooks applied `LabelEncoder` to nominal columns:

```python
df['payment_method_encoded'] = LabelEncoder().fit_transform(df['payment_method'])
# Cash_on_Delivery -> 0, Credit_Card -> 1, PayPal -> 2
```

That asserts an ordering and a spacing that do not exist — that PayPal is
"twice" Credit_Card, and that Credit_Card sits exactly between the other two. A
linear model takes the claim literally. Tree models can partially recover by
splitting repeatedly, but they waste depth doing it.

**The fix.** `OneHotEncoder(handle_unknown="ignore")` for all four categorical
columns. `handle_unknown="ignore"` also means a category value that did not
exist at training time produces an all-zero block instead of a crash — a
deployed model should degrade, not fall over, when the catalogue gains a
category.

This matters more here than it usually would: the strongest drivers in this
dataset are all categorical, so mis-encoding them damages exactly the signal
that carries the model.

---

## 3. The test set is never used to fit anything

**The problem.** The original notebook swept 81 thresholds on the test set and
reported the best one:

```python
y_proba_xgb = xgb_best.predict_proba(X_test_scaled)[:, 1]
for thr in np.arange(0.10, 0.91, 0.01):
    ...                                   # metrics computed on the test set
best_thr = tuning_df.loc[tuning_df['f1'].idxmax()]   # chosen on the test set
```

The resulting F1 is the best that any threshold *could* have achieved on that
exact sample. Nobody deploying the model could have picked that value in
advance, so the number is not a forecast of anything.

**The fix.** A three-way split. Hyperparameters are searched on the training
block; model selection uses the validation block with a threshold-free metric;
the validation block is then halved, with isotonic calibration fitted on one
half and the decision threshold chosen on the other. The test block is scored
once, at the end. `ThresholdChoice` records where it was fitted, so a report can
never quietly imply otherwise.

**The measurement.**

<!-- BEGIN:experiment_threshold -->
| Threshold chosen on | Threshold | F1 reported on the test set |
| --- | ---: | ---: |
| Validation (correct) | 0.32 | 0.4831 |
| Test set (the mistake) | 0.25 | 0.4899 |

Tuning on the test set overstates F1 by +0.0068 — an improvement nobody could have obtained in advance.
<!-- END:experiment_threshold -->

---

## 4. Evaluate the way the model will be deployed

**The problem.** A random split lets the model train on December and be tested
on the previous March. Any temporal structure — a seasonal campaign, a policy
change, a catalogue refresh — leaks backwards.

**The fix.** The default protocol is a **temporal split**: train on the earliest
70% of transactions, validate on the next 15%, test on the most recent 15%. Cut
points fall on calendar dates so a single day never straddles a boundary.
Cross-validation within the training block uses expanding windows, so no fold is
ever trained on transactions that occur after the fold it scores.

Two alternatives are available for comparison (`--split random`, `--split
grouped`). The grouped split holds out whole customers, which answers a
different question: how well does the model generalise to people it has never
seen, rather than to orders it has never seen?

On this dataset the three agree closely, which is itself a finding — the return
rate is flat month to month (standard deviation 0.6 percentage points), so there
is no drift for a random split to conceal. The temporal protocol is still the
right default, because you cannot know that in advance without running it.

---

## 5. Probabilities have to mean something

**The problem.** The CLV model multiplies predicted probabilities by pound
amounts. A model that ranks perfectly but is systematically 15% too confident
will rank customers correctly and price them wrongly — and the ranking metrics
everyone reports, ROC-AUC and average precision, are both invariant to that
error.

Resampling makes this worse, not better: SMOTE inflates the minority class to
parity, so a model trained on resampled data predicts return probabilities
around 0.5 when the true base rate is 0.29.

**The fix.** Isotonic regression fitted on held-out validation data, and Brier
score plus a reliability curve reported alongside the ranking metrics. The
calibrator is fitted on a different half of the validation block from the one
used to pick the threshold, so the threshold is not fitted to the calibrator's
own residual error.

---

## 6. Score customers out-of-sample before valuing them

**The problem.** The original CLV notebook loaded the trained model and scored
the entire dataset — including the 80% the model had been fitted on:

```python
df_scoring['pred_return_prob'] = xgb_model.predict_proba(X_all_scaled)[:, 1]
```

The resulting probabilities are part memory, part prediction, and the model
appears to identify risky customers better than it can.

**The fix.** Every transaction receives a prediction from a model that never saw
it. Validation and test rows are scored by the deployed model, which trained
only on the training block. Training rows are scored by expanding-window
cross-validation. The earliest block — which sits inside every expanding window
and so cannot be predicted going forwards — is backcast by one extra model
fitted on the later blocks only.

---

## 7. The threshold is a business decision

Maximising F1 optimises a quantity with no commercial meaning: it weights a
false positive and a false negative equally, which they never are.

The pipeline models the money instead:

```
cost of a return      = order_value x margin_rate            (margin forgone)
                      + logistics_cost                       (two-way shipping)
                      + order_value x restocking_loss_rate   (markdown, damage)

value of intervening  = P(return) x effectiveness x cost_of_return
                      - intervention_cost
```

Intervening is worth it when the expected saving exceeds the cost. Because the
saving scales with order value while the intervention costs the same either way,
**the optimal rule is value-aware, not a single global cut-off** — a 40% risk on
a £900 order is worth acting on; the same 40% on a £15 order is not.

The pipeline reports both the best global threshold and the value-aware rule,
against three baselines a business would otherwise use: do nothing, intervene on
everything, and a plausible hand-written rule. See `docs/model_card.md` for what
each is worth.

The unit economics are assumptions, not measurements. They live in
`configs/default.yaml` and are exposed as sliders in the dashboard, because the
honest presentation of an assumption is one the reader can move.

---

## 8. Report against the achievable ceiling

Because the data is synthetic and its generating process is documented, the true
return probability behind every row is recoverable. Scoring *that* gives the
performance no model can exceed — the remaining gap to 1.0 is coin-flip noise.

This reframes the result. The project proposal targeted ROC-AUC ≥ 0.80. That
target was never reachable: the ceiling on this data is ≈ 0.68, so 0.80 would
have required predicting the outcome of a Bernoulli draw. A model reported as
"0.68 AUC, short of the 0.80 target" reads as a failure. The same model reported
as capturing **99% of the available signal** is close to the best result this
dataset admits. Both descriptions are arithmetically true; only the second is
informative.

The practical consequence shows up in model selection: logistic regression,
XGBoost and random forest land within 0.01 AUC of each other because all three
are pressed against the same ceiling. On this data, the honest conclusion is
that the extra machinery of gradient boosting buys nothing, and the simplest,
fastest, most interpretable model wins.

Real datasets do not come with a computable ceiling. The substitute is a
well-tuned simple baseline and an explicit irreducible-error argument — but the
discipline is the same: compare a model against what is achievable, not against
1.0.

---

## 9. Data defects are surfaced, not smoothed over

`returns-clv validate` runs eleven checks and classifies each PASS / WARN / FAIL.
Against the committed dataset it raises two warnings, both real (see
`docs/data_dictionary.md`). A validation suite that passes everything is not
evidence that the data is clean; it is evidence that the suite is not looking.

The consequential one is `order_frequency_12m`, which disagrees with the
transaction history on 93.5% of rows. Since CLV multiplies by purchase
frequency, taking that column at face value would misstate lifetime value for
most of the customer base. The CLV stage uses each customer's observed orders
per active year instead, and says so in its output.

---

## 10. Engineered features, and whether they earned their place

Six features are derived from the browsing columns: dwell time per click, visits
per click, log order value, value per page visit, and indicators for low click
depth and short dwell. All are row-wise, so they are stateless and cannot leak
between folds.

The honest question is whether they help. The ablation runs each candidate model
with and without them:

<!-- BEGIN:experiment_ablation -->
| Model | Raw columns only | With engineered features | Change |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.4580 | 0.4617 | **+0.0037** |
| Random Forest | 0.4069 | 0.4068 | **-0.0001** |
| XGBoost | 0.4344 | 0.4394 | **+0.0049** |

_Test-set average precision._
<!-- END:experiment_ablation -->

The answer depends on the model class, which is why the experiment runs across
all of them rather than one. A tree ensemble can discover a threshold effect
like `click_depth <= 3` by splitting on the raw column, so handing it the
indicator changes almost nothing. A linear model cannot represent that
non-linearity at all and has to be given it explicitly.

Reporting only the tree result would have made the features look worthless; the
per-model breakdown shows where they matter. The features are kept because the
deployed model is the one that benefits from them, and because they cost
nothing at inference time.

---

## Summary

| Decision | Coursework notebooks | This pipeline |
| --- | --- | --- |
| Resampling | Before cross-validation | Inside the estimator, per fold |
| Scaling | Fitted on all data | Fitted per fold inside the pipeline |
| Categorical encoding | `LabelEncoder` (implies an order) | `OneHotEncoder(handle_unknown="ignore")` |
| Split | Random | Temporal, with random and grouped for comparison |
| Threshold | Tuned on the test set | Tuned on a validation half, by expected value |
| Calibration | None | Isotonic, on a disjoint validation half |
| CLV scoring | In-sample | Out-of-fold, with the earliest block backcast |
| Purchase frequency | Dataset column | Observed orders per active year |
| Success criterion | Fixed target (AUC ≥ 0.80) | Share of the achievable ceiling captured |
| Reported value | Classification metrics | Pounds, against no-model baselines |
