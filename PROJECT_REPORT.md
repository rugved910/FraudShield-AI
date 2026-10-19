# FraudShield AI — Intelligent Financial Fraud Detection System Using Machine Learning

### Project Report

---

## 1. Title Page

| Field | Details |
|---|---|
| **Project title** | FraudShield AI — Intelligent Financial Fraud Detection System Using Machine Learning |
| **Student name(s)** | ______________ |
| **Roll number(s)** | ______________ |
| **Class / Semester** | ______________ |
| **Department** | ______________ |
| **College / Institute** | ______________ |
| **Academic year** | ______ – ______ |
| **Guide / Faculty** | ______________ |
| **Subject** | Artificial Intelligence and Machine Learning |

*(Replace the underscores with your details before submission.)*

---

## 2. Abstract

Credit-card fraud is a rare but costly event that is difficult to detect with
conventional rules. This project builds FraudShield AI, an end-to-end machine
learning system that estimates whether a transaction resembles previously
observed fraud. Using the public Kaggle *Credit Card Fraud Detection* dataset
(284,807 European cardholder transactions, 0.173 % fraudulent), the system
applies strict data validation, removes duplicate records, performs a
stratified 60/20/20 train–validation–test split before any fitting, and trains
three imbalance-aware classifiers: Logistic Regression with balanced class
weights, Random Forest with balanced class weights, and a Balanced Random
Forest from imbalanced-learn. Because accuracy is uninformative at a 0.17 %
fraud rate, models are compared using the area under the precision–recall curve
(PR-AUC) on the validation split. The Random Forest achieved the best PR-AUC
(0.875) and was selected; its decision threshold was tuned on validation data
(0.2997, maximising F1 with ties broken toward recall) and applied exactly once
to the held-out test set, where it reached precision 0.934, recall 0.747,
F1 0.830, PR-AUC 0.811 and ROC-AUC 0.944. The trained pipeline is served through
a FastAPI backend exposing single, batch and CSV prediction endpoints plus
analytics endpoints, and visualised in a five-page React dashboard that reads
only real evaluation artifacts. The report documents the data, methodology,
results, limitations and ethical constraints, and explicitly distinguishes a
model *prediction* from *confirmed fraud*.

---

## 3. Introduction

Digital payment volume continues to grow, and with it the opportunity for
fraud. Fraud analysts face a screening problem: they can review only a small
fraction of transactions, so any automated aid must rank the population so that
the few genuine frauds surface near the top of the queue while avoiding an
unmanageable number of false alarms.

Machine learning is well suited to this ranking task because fraud patterns are
subtle, multivariate and shifting, and because labelled historical data is
available. However, real fraud datasets are extremely imbalanced, and naïve
training produces models that appear accurate while detecting almost nothing.
This project therefore treats imbalance handling, leakage-free evaluation,
metric choice and decision-threshold tuning as first-class engineering
problems, not afterthoughts.

The deliverable is a complete, runnable system — data pipeline, trained models,
REST API, analyst dashboard and automated tests — not a notebook prototype.

---

## 4. Problem Statement

Given a single credit-card transaction described by 30 numeric features
(28 anonymised PCA components, transaction time and euro amount), predict
whether the transaction is legitimate or potentially fraudulent, producing:

1. a binary decision at a documented, validated threshold,
2. a model risk score,
3. a low/medium/high risk category, and
4. a basic, honest explanation of the model's reasoning,

such that the resulting system maximises detection of the rare fraud class
without generating an impractical number of false alarms, while never
presenting a prediction as confirmed fraud.

---

## 5. Objectives

1. Acquire and strictly validate the public Kaggle fraud dataset, failing loudly
   on missing, corrupt or malformed data instead of silently substituting
   synthetic records.
2. Perform exploratory data analysis and record dataset statistics.
3. Prevent data leakage by splitting data before any fitting and removing
   duplicates first.
4. Train and fairly compare at least three suitable classification approaches
   with fixed random seeds.
5. Evaluate models with precision, recall, F1, ROC-AUC, PR-AUC, confusion
   matrices and FP/FN counts; select the model using validation data only.
6. Tune the classification threshold on validation data and evaluate once on a
   held-out test set.
7. Persist reproducible artifacts: the fitted pipeline, metadata, metrics,
   thresholds, feature importance and plots.
8. Serve predictions through a validated, documented FastAPI service.
9. Provide an interactive dashboard presenting only real evaluation results.
10. Document the work for academic submission, including a model card and viva
    preparation notes.

---

## 6. Existing System and Proposed System

**Existing (rule-based) systems.** Most production fraud engines begin with
hand-written rules and thresholds (amount limits, velocity checks, geographic
impossibility, blocklists). Rules are transparent and easy to audit, but they
suffer from: high maintenance cost as fraudsters adapt; poor generalisation to
new patterns; high false-positive rates that create customer friction; and
inability to combine many weak signals. Manual review alone does not scale to
millions of daily transactions.

**Proposed system.** FraudShield AI learns the decision boundary from labelled
historical data. It (a) standardises the two interpretable features while leaving
the pre-scaled PCA components untouched, (b) models imbalance through class
weighting and balanced resampling restricted to training folds, (c) ranks
transactions with a validated score, (d) converts the score into a decision
using a threshold tuned on validation data to reflect the asymmetric cost of
missed fraud, and (e) surfaces ranked results and explanations to a human
reviewer. It is a decision-support prototype, not an autonomous blocking system.

---

## 7. Literature Review

The following works are real and publicly verifiable; they informed the design
choices in this project.

1. **Dal Pozzolo, A., Caelen, O., Le Borgne, Y.-A., Waterschoot, S., &
   Bontempi, G. (2014).** *Learned lessons in credit card fraud detection from a
   practitioner perspective.* Expert Systems with Applications, 41(10),
   4915–4928. — Documents the real-world imbalance and cost asymmetry of card
   fraud and motivates evaluation by precision/recall rather than accuracy; the
   sampling and thresholding discussion directly informs this project's design.

2. **Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., & Bontempi, G.
   (2015).** *Credit card fraud detection: a realistic modeling and a novel
   learning strategy.* IEEE Transactions on Neural Networks and Learning
   Systems, 29(8), 3784–3797. — Analyses concept drift and non-stationarity in
   transaction streams, relevant to the temporal limitations noted in §16.

3. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).**
   *SMOTE: Synthetic Minority Over-sampling Technique.* Journal of Artificial
   Intelligence Research, 16, 321–357. — Introduces a standard minority
   oversampling technique; illustrates why resampling must be confined to
   training data.

4. **He, H., & Garcia, E. A. (2009).** *Learning from imbalanced data.* IEEE
   Transactions on Knowledge and Data Engineering, 21(9), 1263–1284. — Survey of
   imbalance strategies including undersampling and cost-sensitive learning,
   underpinning the Balanced Random Forest approach.

5. **Chen, C., Liaw, A., & Breiman, L. (2004).** *Using random forest to learn
   imbalanced data.* Technical Report, University of California, Berkeley. —
   Studies of random forests on imbalanced problems; motivates the class
   weighting and per-tree balancing used here.

6. **Breiman, L. (2001).** *Random Forests.* Machine Learning, 45(1), 5–32. —
   The original ensemble method; basis for the selected model and for
   mean-decrease-in-impurity feature importance.

7. **Davis, J., & Goadrich, M. (2006).** *The relationship between Precision-
   Recall and ROC curves.* Proceedings of ICML 2006, 233–240. — Explains why
   precision–recall curves are more informative than ROC under severe class
   imbalance, the central metric argument of this report.

8. **Saito, T., & Rehmsmeier, M. (2015).** *The precision-recall plot is more
   informative than the ROC plot when evaluating binary classifiers on
   imbalanced datasets.* PLoS ONE, 10(3), e0118432. — Empirical confirmation of
   the same point; motivated selecting the model by PR-AUC.

9. **Mitchell, M., Wu, S., Zaldivar, A., et al. (2019).** *Model Cards for Model
   Reporting.* Proceedings of FAT* 2019, 220–229. — Template for
   `MODEL_CARD.md`, including intended use, limitations and ethical
   considerations.

10. **Lundberg, S. M., & Lee, S.-I. (2017).** *A Unified Approach to
    Interpreting Model Predictions.* Advances in Neural Information Processing
    Systems 30. — SHAP, the standard attribution method; discussed as future
    work since this project uses cheaper built-in explanations.

11. **Pedregosa, F., et al. (2011).** *Scikit-learn: Machine Learning in
    Python.* Journal of Machine Learning Research, 12, 2825–2830. — The library
    providing the pipelines, metrics and models.

12. **Lemaître, G., Nogueira, F., & Aridas, C. K. (2017).** *Imbalanced-learn:
    A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine
    Learning.* Journal of Machine Learning Research, 18(17), 1–5. — Provides
    `BalancedRandomForestClassifier`.

---

## 8. Dataset Description

| Property | Value |
|---|---|
| Source | Kaggle — *Credit Card Fraud Detection*, MLG-ULB (Université Libre de Bruxelles) |
| URL | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| Records (raw) | 284,807 |
| Fraud (raw) | 492 (0.173 %) |
| Duplicate rows | 1,081 (removed before splitting) |
| Records used | 283,726 |
| Fraud used | 473 (0.167 %) |
| Missing values | 0 |
| Features | `Time` (seconds since first transaction), `V1…V28` (anonymised PCA components), `Amount` (EUR) |
| Target | `Class` — 0 = legitimate, 1 = fraud |
| Period | Two days, September 2013 |

**Validation performed.** The loader checks file existence and size, parses the
CSV with explicit corruption handling, verifies all 31 expected columns exist,
confirms all required columns are numeric with no NaN/infinite values, confirms
the target is strictly binary and contains fraud cases, rejects negative `Time`
or `Amount`, and counts duplicates. Failures raise actionable errors; the
pipeline never substitutes synthetic data for the real file (demo mode must be
requested explicitly and is labelled everywhere).

**Known limitations.** Only 48 hours of a single processor's European traffic;
features are anonymised, blocking domain interpretation; no demographic data
exists, so fairness audits are impossible; the dataset has been widely used, so
excellent scores do not imply production readiness.

---

## 9. Methodology

**Step 1 — Data validation.** As in §8. Duplicate rows are dropped *before*
splitting so that no transaction can appear in both a training and an evaluation
split (a subtle but real form of leakage).

**Step 2 — Exploratory analysis.** Class distribution, log-scale amount
distribution, amount by class, fraud by hour of day (derived from `Time`) and a
correlation heatmap are generated and stored as figures.

**Step 3 — Splitting.** The deduplicated data is split with stratification into
60 % train (170,235), 20 % validation (56,745) and 20 % test (56,746) using seed
42. Stratification preserves the ~0.17 % fraud ratio in every split, which is
essential for stable evaluation of a rare class. The split is created **before**
any model fitting or preprocessing.

**Step 4 — Preprocessing.** A scikit-learn `ColumnTransformer` standardises
`Time` and `Amount` with `StandardScaler` **fitted on the training fold only**
and passes `V1…V28` through unchanged, because they are already PCA-scaled. The
transformer is embedded inside each model pipeline so that inference automatically
applies identical preprocessing.

**Step 5 — Imbalance handling.** Logistic Regression and Random Forest use
`class_weight="balanced"`. Balanced Random Forest draws a bootstrap sample per
tree in which the majority class is undersampled to the minority count, so each
tree sees roughly balanced classes. All balancing occurs strictly inside the
training fold.

**Step 6 — Model selection.** Each model is evaluated on the validation split at
threshold 0.5 using precision, recall, F1, ROC-AUC, PR-AUC, confusion-matrix
cells and FP/FN counts. The model with the highest validation PR-AUC is selected.
Accuracy is deliberately not used.

**Step 7 — Threshold tuning.** Using the validation precision–recall curve for
the selected model, the threshold maximising F1 is chosen; ties within 10⁻⁶ are
broken toward the higher-recall (lower) threshold, because a false negative is
assumed costlier than a false positive. The tuned value is 0.2997.

**Step 8 — Final evaluation.** The selected pipeline and frozen threshold are
applied **once** to the untouched test split. No test-driven decisions are made
anywhere in the pipeline.

**Step 9 — Artifact persistence.** Pipeline, metadata, comparison metrics,
curves, threshold sweep, feature importance, dataset statistics and sample
transactions are written to `artifacts/`; figures to `reports/figures/`.

---

## 10. System Architecture

```
                data/raw/creditcard.csv  (Kaggle, 150 MB, git-ignored)
                            │
                            ▼
        [ scripts/train.py ]  validation → EDA figures → stratified split
                            │
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
        Logistic      Random Forest   Balanced RF
        Regression    (selected)      (imbalanced-learn)
              └─────────────┼──────────────┘
                            ▼
        artifacts/  model_pipeline.joblib · model_metadata.json
                    model_comparison.json · evaluation_metrics.json
                    feature_importance.json · dataset_stats.json
                    sample_transactions.json
                            │
                            ▼
        [ backend FastAPI :8000 ]  loads pipeline once at startup
              /api/predict · /api/predict/batch · /api/predict/csv
              /api/model-info · /api/analytics/* · /api/samples · /health
                            │  (JSON over HTTP; Vite dev proxy; CORS localhost:5173)
                            ▼
        [ frontend React + TypeScript :5173 ]
              Dashboard · Transaction Analyzer · Batch Analysis
              Model Performance · About Project
```

**Separation of concerns:** data processing (`scripts/`, `backend/app/core`),
training (`scripts/train.py`), inference (`backend/app/services`), HTTP layer
(`backend/app/api`, `backend/app/schemas`), presentation (`frontend/src`), and
verification (`backend/tests`). The API never loads the training dataset.

---

## 11. Algorithm Explanations

**Logistic Regression.** A linear model computing
`z = w·x + b` and `p = σ(z) = 1/(1+e^{-z})`. Coefficients are log-odds per unit
of the (standardised) feature, which makes the model directly interpretable.
With `class_weight="balanced"`, each class's loss contribution is weighted
inversely to its frequency, making minority errors far more expensive. It serves
as a fast baseline and produced the highest ROC-AUC (0.979) but poor precision at
threshold 0.5.

**Random Forest.** An ensemble of decorrelated decision trees, each trained on a
bootstrap sample with random feature subsets at each split; predictions are the
average class probability across trees. Averaging reduces variance and captures
non-linear interactions. `class_weight="balanced"` re-weights impurities per
node. Feature importance is measured by mean decrease in impurity (MDI). This
model was selected.

**Balanced Random Forest.** An imbalanced-learn variant in which each tree is
trained on a sample containing all minority-class examples and an equal number
of randomly drawn majority examples, so individual trees see balanced classes
without altering the validation or test distributions. It maximises recall but
sacrificed precision on this dataset (PR-AUC 0.778).

**Threshold decision rule.** The model outputs `p`; the transaction is flagged
when `p ≥ t`, where `t = 0.2997` was tuned on validation data. Risk bands are
`low` for `p < 0.5t`, `medium` for `0.5t ≤ p < t`, and `high` for `p ≥ t`.

---

## 12. Implementation Details

| Component | Implementation |
|---|---|
| Language | Python 3.10 (backend/ML), TypeScript 5.6 (frontend) |
| ML | scikit-learn 1.7, imbalanced-learn 0.14, joblib |
| Data | pandas 2.3, NumPy 1.26 |
| Plots | matplotlib 3.10, seaborn 0.13 |
| API | FastAPI 0.141, Pydantic 2.13, uvicorn 0.53 |
| UI | React 18, Vite 5, Tailwind CSS 3, Recharts 2, lucide-react |
| Tests | pytest 9, httpx TestClient, `tsc -b` + `vite build` |

**Training script** (`scripts/train.py`, ~600 lines) is organised into data
loading, EDA, splitting, model construction, evaluation, threshold tuning,
plotting and artifact persistence, with `argparse` supporting `--demo`.

**Backend.** `create_app()` builds the FastAPI application with a lifespan hook
that loads artifacts once; `ModelService` wraps the pipeline, exposes
`predict_one`, `predict_batch` and `explain`, and reports load failures instead
of crashing so `/health` can diagnose a missing model. Prediction endpoints
reject missing, unexpected, non-numeric and out-of-range fields via Pydantic,
return HTTP 503 when the model is absent, and enforce a 1,000-row batch limit
and a 5 MB CSV limit. CORS is restricted to the local Vite origin.

**CSV service.** Validates extension, size and row count; requires the 30
feature columns; ignores extras such as `Class` (labels are never used as
features); skips individual malformed rows and reports each reason with its row
number rather than failing the whole upload.

**Frontend.** Five routed pages share a dark workstation layout. The Dashboard,
Performance and About pages render data fetched from the analytics endpoints;
the Analyzer validates all 30 inputs client-side and renders the returned
explanation; the Batch page validates the file before upload, shows a preview,
per-row predictions, aggregate statistics and a client-side CSV export.

**Persistence.** No database is used. Results exist only in the browser session;
the server stores nothing about submitted transactions and accepts no
cardholder identifiers.

---

## 13. Evaluation Metrics

| Metric | Definition | Role here |
|---|---|---|
| Precision | TP / (TP + FP) | Cost of false alarms |
| Recall (sensitivity) | TP / (TP + FN) | Fraud detection rate |
| F1 | Harmonic mean of precision and recall | Single balanced summary; threshold-tuning objective |
| ROC-AUC | Area under TPR-vs-FPR curve | Overall ranking quality; optimistic under imbalance |
| PR-AUC (average precision) | Area under precision-vs-recall curve | **Primary model-selection metric** |
| Confusion matrix | TN, FP, FN, TP | Absolute error counts in business terms |
| FP / FN counts | Derived from the matrix | Directly interpretable review/workload and loss figures |

All metrics are computed at an explicitly stated threshold. Accuracy is not used.
Training, validation and test results are reported separately and labelled.

---

## 14. Experimental Results

Environment: Windows, Python 3.10.6, scikit-learn 1.7.0, imbalanced-learn 0.14.2,
8-core CPU. Seed 42 throughout; one run reproduced exactly on repeat.

**Validation split (56,745 transactions; threshold 0.5):**

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP | FN | Train (s) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.0538 | 0.9043 | 0.1016 | 0.9788 | 0.7868 | 1,494 | 9 | 2.0 |
| **Random Forest** | 0.9176 | 0.8298 | 0.8715 | 0.9606 | **0.8752** | 7 | 16 | 157.7 |
| Balanced Random Forest | 0.0511 | 0.9043 | 0.0966 | 0.9696 | 0.7781 | 1,580 | 9 | 13.2 |

*Random Forest was selected by validation PR-AUC. Threshold tuned on validation:
0.2997 → validation precision 0.9000, recall 0.8617, F1 0.8804.*

**Held-out test split (56,746 transactions; 95 frauds) at threshold 0.2997:**

| Metric | Value |
|---|---|
| Precision | 0.9342 |
| Recall | 0.7474 |
| F1 | 0.8304 |
| PR-AUC | 0.8110 |
| ROC-AUC | 0.9437 |
| TN / FP / FN / TP | 56,646 / 5 / 24 / 71 |
| Risk distribution | low 56,659 · medium 11 · high 76 |

**Reference — same model at the default threshold 0.5:** precision 0.9710,
recall 0.7053, F1 0.8171 (FP 2, FN 28). The tuned threshold converts 3 extra
false alarms into 4 additional detected frauds.

**Interpretation.** The Random Forest clearly dominates on PR-AUC and F1;
the two recall-optimised models achieve high recall only with thousands of false
positives. No claim of superiority beyond these measured numbers is made; in
particular, performance on this 2-day 2013 dataset does not predict performance
on live traffic. The model misses 24 of 95 test frauds — a limitation stated
plainly in the interface and this report.

---

## 15. Screenshots

Insert screenshots captured from the running application before submission:

| Figure | Description | Path |
|---|---|---|
| 1 | Dashboard with real dataset statistics and confusion matrix | `docs/screenshots/dashboard.png` |
| 2 | Transaction Analyzer result for a real fraud sample | `docs/screenshots/analyzer.png` |
| 3 | Batch analysis with flagged rows and export | `docs/screenshots/batch.png` |
| 4 | Model performance comparison and curves | `docs/screenshots/performance.png` |
| 5 | Swagger API documentation | `docs/screenshots/swagger.png` |
| 6 | Class distribution figure | `reports/figures/01_class_distribution.png` |
| 7 | Precision–recall curves (validation) | `reports/figures/06_precision_recall_curves.png` |
| 8 | Threshold sweep | `reports/figures/10_threshold_sweep.png` |

---

## 16. Advantages and Limitations

**Advantages**
- Complete, reproducible pipeline from raw CSV to a served prediction, with
  fixed seeds and saved artifacts.
- Leakage-safe methodology: duplicates removed before splitting, preprocessing
  fitted on training data only, resampling confined to training folds, and a
  test split touched exactly once.
- Correct metric choice for extreme imbalance (PR-AUC and recall prioritised
  over accuracy) with an explicitly tuned, documented decision threshold.
- Strict input validation on every endpoint; corrupt input files and malformed
  rows produce actionable errors rather than silent wrong answers.
- Explanations are honest about their limits; predictions are never presented as
  confirmed fraud, and the system cannot block payments.
- 56 automated tests plus a type-checked production frontend build.

**Limitations**
- 48 hours of 2013 European cardholder data limits external validity; concept
  drift is not modelled.
- Anonymised PCA features prevent domain interpretation; explanations are
  statistical associations, not causes.
- Model scores are uncalibrated and should not be read as true probabilities.
- The model misses roughly a quarter of test-set frauds; precision above 0.9 is
  measured on a small number of test frauds (95).
- No demographic attributes exist, so fairness across groups cannot be audited.
- CSV batch analysis is capped at 1,000 rows; no streaming ingestion.
- Demo mode (`--demo`) trains on synthetic data whose metrics have no real-world
  meaning, and is labelled as such throughout.

---

## 17. Future Scope

1. **Probability calibration** (Platt scaling / isotonic regression) so scores
   become interpretable probabilities, validated on a held-out calibration split.
2. **Cost-sensitive thresholding** using an explicit fraud-loss matrix supplied
   by a business, replacing the F1 proxy.
3. **Temporal validation and drift monitoring** — train on earlier days, test on
   later ones, and monitor score distributions for drift.
4. **SHAP-based explanations** as an optional dependency to provide theoretically
   grounded attributions.
5. **Graph features** (shared devices, merchants, sessions) when such data is
   available, to capture organised fraud rings.
6. **Human-in-the-loop feedback** so analyst decisions can be logged and used for
   periodic retraining.
7. **Model monitoring dashboard** tracking live precision/recall estimates.
8. **Scaling** to streaming ingestion (Kafka) and GPU-accelerated training.

---

## 18. Conclusion

This project delivered a complete, working machine-learning fraud-detection
system. Data was validated rigorously, 1,081 duplicate records were removed
before a stratified 60/20/20 split, and three imbalance-aware classifiers were
trained and compared on the validation split using PR-AUC, the appropriate
metric at a 0.167 % fraud rate. A Random Forest was selected on measured
validation performance (PR-AUC 0.875), its decision threshold was tuned on
validation data (0.2997), and it was evaluated exactly once on the held-out test
set, achieving precision 0.934, recall 0.747, F1 0.830, PR-AUC 0.811 and
ROC-AUC 0.944 with a confusion matrix of 56,646 true negatives, 5 false
positives, 24 false negatives and 71 true positives. The trained pipeline is
served through a documented FastAPI service and presented in an interactive
React dashboard that displays only real evaluation artifacts. The system's
limitations — uncalibrated scores, anonymised features, a small two-day dataset
and a 25 % miss rate — are documented rather than hidden, and the application
never represents a prediction as confirmed fraud. The project therefore meets
its objective: an honest, reproducible, demonstrable machine-learning solution
to a genuinely difficult detection problem.

---

## 19. References

1. Dal Pozzolo, A., Caelen, O., Le Borgne, Y.-A., Waterschoot, S., & Bontempi,
   G. (2014). Learned lessons in credit card fraud detection from a practitioner
   perspective. *Expert Systems with Applications*, 41(10), 4915–4928.
2. Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., & Bontempi, G. (2015).
   Credit card fraud detection: a realistic modeling and a novel learning
   strategy. *IEEE Transactions on Neural Networks and Learning Systems*,
   29(8), 3784–3797.
3. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE:
   Synthetic Minority Over-sampling Technique. *Journal of Artificial
   Intelligence Research*, 16, 321–357.
4. He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE
   Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284.
5. Chen, C., Liaw, A., & Breiman, L. (2004). Using random forest to learn
   imbalanced data. *Technical Report*, University of California, Berkeley.
6. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
7. Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall
   and ROC curves. *Proceedings of ICML 2006*, 233–240.
8. Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more
   informative than the ROC plot when evaluating binary classifiers on imbalanced
   datasets. *PLoS ONE*, 10(3), e0118432.
9. Mitchell, M., Wu, S., Zaldivar, A., et al. (2019). Model Cards for Model
   Reporting. *Proceedings of FAT\* 2019*, 220–229.
10. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting
    Model Predictions. *Advances in Neural Information Processing Systems*, 30.
11. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn:
    Machine Learning in Python. *Journal of Machine Learning Research*, 12,
    2825–2830.
12. Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A
    Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine
    Learning. *Journal of Machine Learning Research*, 18(17), 1–5.
13. Machine Learning Group — ULB (2018). *Credit Card Fraud Detection* (dataset).
    Kaggle. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
14. Ramírez, S. (2018). *FastAPI* (software). https://fastapi.tiangolo.com
15. Meta Open Source (2024). *React* (software). https://react.dev
