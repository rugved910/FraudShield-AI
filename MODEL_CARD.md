# Model Card — FraudShield AI

Following the spirit of *Model Cards for Model Reporting* (Mitchell et al., 2019).

---

## 1. Model details

| Field | Value |
|---|---|
| Model name | FraudShield AI — Random Forest classifier (v1.0.0) |
| Model type | `sklearn.ensemble.RandomForestClassifier` (200 trees, `class_weight="balanced"`, `min_samples_leaf=2`) inside a `sklearn.pipeline.Pipeline` |
| Preprocessing | `ColumnTransformer`: `StandardScaler` on `Time` and `Amount` (fitted on the training fold only) + passthrough for the already-scaled PCA features `V1…V28` |
| Artifact | `artifacts/model_pipeline.joblib` (the exact preprocessing + model pipeline used at inference) |
| Trained | 2026-10-19 (UTC), seed 42 |
| Training data | 170,235 transactions (60 % of 283,726 deduplicated rows) |
| Validation data | 56,745 transactions (20 %) — used for model selection and threshold tuning |
| Test data | 56,746 transactions (20 %, 95 frauds) — touched once, after freezing |
| Decision threshold | 0.2997 (tuned on validation to maximise F1, ties broken toward higher recall) |

### Candidate models compared

1. Logistic Regression (`class_weight="balanced"`) — validation PR-AUC 0.787
2. Random Forest (`class_weight="balanced"`) — validation PR-AUC 0.875 ← **selected**
3. Balanced Random Forest (imbalanced-learn) — validation PR-AUC 0.778

Selection criterion: **validation PR-AUC** (accuracy is meaningless at a
0.17 % fraud rate). The test set was never used for model selection,
hyper-parameter tuning, or threshold tuning.

## 2. Intended use

**Intended**
- Undergraduate teaching and demonstration of an end-to-end fraud-detection
  ML workflow (imbalance handling, leakage prevention, threshold tuning).
- Ranking transactions by suspiciousness for *human review* in a sandbox.

**Not intended (prohibited)**
- Automatically blocking, declining, or reversing real payments.
- Any claim that a scored transaction *is* fraud, or that a customer committed fraud.
- Decision-making about real individuals, credit, accounts, or law enforcement.
- Presenting the ≤0.17 %-fraud dataset's numbers as production performance.
- Any use of the uncalibrated score as a literal "probability of fraud".

## 3. Training data

Source: Kaggle *Credit Card Fraud Detection* (MLG-ULB, ULB Machine Learning
Group), 284,807 European cardholder transactions over 48 hours (September 2013).
Features `V1…V28` are anonymised PCA components; `Time` (seconds) and `Amount`
(EUR) are the only interpretable columns. `Class` is the binary target
(492 frauds in the raw file, i.e. 0.173 %). 1,081 duplicate rows were removed
before splitting, leaving 283,726 rows / 473 frauds.

Nothing else was scraped, merged, or synthesised. If the real file is absent,
`scripts/train.py --demo` trains on **clearly-labelled synthetic data**; such
artifacts carry `is_demo=true` and the UI displays a permanent warning.

## 4. Evaluation — measured results

**Validation set (threshold 0.5, fair comparison):**

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.787 | 0.979 | 0.054 | 0.904 | 0.102 | 1,494 | 9 |
| **Random Forest** | **0.875** | 0.961 | 0.918 | 0.830 | 0.872 | 7 | 16 |
| Balanced Random Forest | 0.778 | 0.970 | 0.051 | 0.904 | 0.097 | 1,580 | 9 |

**Held-out test set, selected model @ threshold 0.2997:**

| Metric | Value |
|---|---|
| Precision | 0.9342 |
| Recall | 0.7474 |
| F1 | 0.8304 |
| PR-AUC (average precision) | 0.8110 |
| ROC-AUC | 0.9437 |
| Confusion matrix | TN 56,646 · FP 5 · FN 24 · TP 71 |

At threshold 0.5 the same model yields precision 0.9710 / recall 0.7053
(FP 2, FN 28). The lower threshold deliberately trades 3 extra false alarms for
4 extra detected frauds — a documented, validation-tuned trade-off.

## 5. Threshold selection

The threshold is not 0.5 by default. On the **validation** split, a
precision–recall sweep selects the cut-off that maximises F1; ties are broken
toward higher recall because a missed fraud (false negative) is assumed costlier
than a false alarm (false positive). The chosen value (0.2997) is stored in
`model_metadata.json` and applied identically at inference time. The `medium`
risk band (`0.5 × threshold ≤ p < threshold`) marks borderline transactions for
review rather than automatic flagging.

## 6. Known limitations & potential bias

- **External validity:** 48 hours of 2013 European cardholder data. Behaviour
  has changed; performance elsewhere is unknown.
- **Uncalibrated scores:** no Platt/isotonic calibration was applied or verified.
  Scores support ranking, not probability statements.
- **Anonymised features:** `V1…V28` cannot be interpreted as merchant, location,
  device, or cardholder attributes. Explanations are statistical associations,
  never causes.
- **Fairness cannot be audited:** the dataset contains no demographic or
  protected attributes, so disparate impact across groups cannot be measured.
- **Selection bias:** the dataset was collected by one processor over two days;
  fraud types not present in it will be missed.
- **Class-imbalance artefacts:** oversampling/class weighting can inflate
  apparent recall on this particular dataset.
- **Threshold cost model is assumed, not business-derived:** the F1-with-recall
  tie-break is a reasonable proxy only.

## 7. Ethical considerations

- Never present a prediction as confirmed fraud; the UI states the disclaimer on
  every result, and prediction responses include a machine-readable warning.
- No card numbers, names, addresses, or account identifiers are accepted,
  stored, or logged. The API rejects unexpected fields.
- Predictions are explainable at a basic level (linear contributions or
  tree-importance-weighted deviations) but the limits of that explanation are
  stated in the response payload.

## 8. How to reproduce

```bash
python scripts/train.py     # writes artifacts/ and reports/figures/
python -m pytest backend/tests -q
```

All random seeds are fixed (`RANDOM_SEED = 42`), so metrics are reproducible.
Raw data is not committed to Git (licence + size); see `README.md` §5.
