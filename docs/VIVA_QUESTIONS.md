# Viva / Interview Preparation — FraudShield AI

Technically accurate answers for a project defence. Numbers refer to this
repository's actual training run.

---

## A. Dataset & problem framing

**Q1. What is the dataset, and what does each column mean?**
`mlg-ulb/creditcardfraud` from Kaggle: 284,807 transactions by European
cardholders over ~48 hours (Sept 2013), 492 frauds (0.173 %). `V1…V28` are
principal components produced by the dataset authors to anonymise the raw
features; `Time` is seconds since the first transaction; `Amount` is euros;
`Class` is 0 = legitimate, 1 = fraud. Only `Time` and `Amount` are human-readable.

**Q2. Why is this a hard problem?**
Extreme class imbalance (0.17 %), an evolving adversary, cost asymmetry between
missed fraud and false alarms, and anonymised features that block
interpretation. A model that predicts "legitimate" always achieves 99.83 %
accuracy and is worthless.

**Q3. Why is accuracy a bad metric here?**
Because it is dominated by the majority class. Always-legitimate ⇒ 99.83 %
accuracy with zero frauds caught. We therefore report precision, recall, F1,
ROC-AUC, PR-AUC and confusion matrices, and *select models on validation PR-AUC*.

## B. Metrics

**Q4. Precision vs recall — define them for this project.**
Precision = of the transactions we flagged, the fraction that were really fraud
(TP / (TP + FP)); it measures false-alarm cost. Recall = of all real frauds,
the fraction we caught (TP / (TP + FN)); it measures missed-fraud cost. At the
tuned threshold our model achieves precision 0.934 and recall 0.747:
~93 % of flags are true frauds, and three quarters of frauds are caught.

**Q5. What are false positives and false negatives in practice?**
FP = a legitimate transaction sent for review — customer friction and analyst
cost. FN = a fraud transaction scored as legitimate — direct financial loss and
the costliest error, which is why threshold tuning breaks F1 ties toward recall.

**Q6. ROC-AUC vs PR-AUC — why prefer PR-AUC?**
ROC-AUC uses the false-positive rate, whose denominator (the huge legitimate
count) makes it look excellent even for weak models — note Balanced Random
Forest's ROC-AUC 0.970 but PR-AUC 0.778. PR-AUC focuses on the rare positive
class and does not reward easy true negatives, so it discriminates far better
under heavy imbalance. We therefore use PR-AUC for selection.

**Q7. What does PR-AUC ≈ 0.81 mean?**
Area under the precision–recall curve (average precision). A random classifier
would score ≈0.0017 (the fraud prevalence). 0.81 means that across all
thresholds, precision stays high while recall increases substantially — a
strong ranking of the rare class on this dataset.

## C. Data leakage & methodology

**Q8. What is data leakage, and how did you prevent it?**
Leakage is any information in training that would not be available at prediction
time. We prevented it by (a) removing 1,081 duplicate rows *before* splitting,
so no transaction appears in two splits; (b) splitting 60/20/20 stratified
*before* fitting; (c) fitting `StandardScaler` only on the training fold inside
the sklearn pipeline; (d) never resampling the validation or test sets; and
(e) never touching the test set until the model and threshold were frozen.

**Q9. How did you ensure the test set stayed honest?**
Model choice used validation PR-AUC only; threshold tuning used the validation
precision–recall curve only; the test set was scored exactly once, after both
decisions were fixed. `scripts/train.py` contains no test-set-driven branching.

**Q10. Why a stratified split instead of a temporal one?**
Stratification guarantees the 0.17 % fraud ratio in every split, which matters
for stable evaluation of a rare class. A temporal split (train on day 1, test on
day 2) is arguably more realistic and is listed as future work; because the
classes are randomly interleaved in time here, stratification is the standard,
defensible choice.

**Q11. Why not scale the V features?**
They are already principal components from a standardised PCA performed by the
dataset authors — unit-variance, zero-mean. Re-scaling would add no information.
`Time` and `Amount` have very different scales, so they *are* standardised.

## D. Models & imbalance

**Q12. Why these three models?**
Logistic Regression is a fast, interpretable linear baseline; Random Forest
captures non-linear interactions; Balanced Random Forest attacks imbalance
structurally. Comparing them shows that imbalance handling strategy matters more
than raw algorithm complexity here — the class-weighted Random Forest beat the
explicitly balanced one on PR-AUC.

**Q13. How does `class_weight="balanced"` help?**
It scales each class's contribution to the loss inversely to its frequency, so
mistakes on the 0.17 % minority cost ~590× more than majority mistakes. Recall
rises sharply; but note the default 0.5 threshold then yields many false
positives (Logistic Regression: 1,494 FPs on validation), so threshold tuning is
still needed.

**Q14. How does Balanced Random Forest differ?**
Each tree trains on a bootstrap sample in which the majority class is
undersampled to the minority count, so individual trees see ~1:1 classes.
Aggregation across 200 trees recovers precision. Here it produced high recall
(0.904) but weak precision at 0.5 (0.051) and the lowest PR-AUC (0.778) — an
honest negative result we report rather than hide.

**Q15. Did you oversample? Where, and why does that matter?**
Only inside training via Balanced Random Forest. Oversampling before splitting
would create duplicate minority rows in the test set, inflating scores — a
classic leakage error we avoided.

## E. Threshold

**Q16. Why not use 0.5 as the threshold?**
0.5 is arbitrary; it is the default only because it minimises expected error
under balanced 0/1 loss. Fraud detection has asymmetric costs, so the threshold
is a business decision. We tune it on validation data, maximising F1 and
breaking ties toward recall. The result (0.2997) trades 3 extra false alarms for
4 additional detected frauds versus 0.5.

**Q17. Could you choose the threshold from the test set?**
No — that is leakage. Test-set thresholding would overstate performance and
invalidate the held-out evaluation. The threshold is frozen from validation
data before the test set is ever scored.

**Q18. What does the "medium risk" band mean?**
Probability between `0.5 × threshold` and `threshold`. These are borderline
transactions (11 in our test set) flagged for human review rather than
auto-handling. The ratio is configurable in `backend/app/core/config.py`.

## F. Explainability & deployment

**Q19. How do you explain a prediction?**
For the linear model, `coefficient × scaled feature value` gives a signed
contribution. For the selected tree ensemble, we weight each feature's global MDI
importance by how far its scaled value deviates from the training mean, and
report the top three contributions with direction. We state explicitly that
these are statistical associations and that `V` features cannot be interpreted
as real attributes.

**Q20. What does the API guarantee about prediction latency and dataset access?**
The saved pipeline (`artifacts/model_pipeline.joblib`) is loaded once at
startup; inference needs no access to the 150 MB CSV, so endpoints are
stateless and fast. Requests never retrain anything.

**Q21. Is your fraud probability a real probability?**
No. It is an uncalibrated score (no Platt/isotonic calibration was applied or
verified). Every API response says so. Ranking thresholds are valid; the literal
number should not be read as a true probability.

**Q22. What are the system's limitations?**
Two days of 2013 European data; anonymised features prevent real-world
interpretation; uncalibrated scores; no fairness audit possible; the demo
(`--demo`) model has no real-world meaning. Predictions must never block real
payments and never represent confirmed fraud.

**Q23. How is the frontend connected to the model?**
React (port 5173) calls relative `/api/*` URLs; the Vite dev server proxies them
to FastAPI (port 8000), and CORS is additionally restricted to
`http://localhost:5173`. The UI displays only values fetched from the analytics
endpoints, which read the saved evaluation artifacts — no fabricated metrics.

**Q24. What tests exist?**
56 pytest tests cover dataset validation (missing columns, NaN, negative
amounts, corrupt files, duplicates), leakage prevention (scaler fitted on train
fold, V-features untouched), artifact loading and failure modes, single/batch/CSV
prediction, request validation (missing/extra/out-of-range fields, batch limits),
health and analytics endpoints, and CSV edge cases. The frontend build is
type-checked via `tsc -b` and produced by `vite build`.
