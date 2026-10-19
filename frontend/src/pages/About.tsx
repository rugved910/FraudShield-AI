import { Database, GitBranch, Layers, Scale, Target, TrendingUp } from "lucide-react";

function Section({
  icon: Icon,
  title,
  children,
}: {
  icon: typeof Target;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="panel">
      <h2 className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-200">
        <Icon className="h-4 w-4 text-signal-blue" aria-hidden />
        {title}
      </h2>
      <div className="space-y-2 text-sm leading-relaxed text-slate-400">{children}</div>
    </section>
  );
}

export default function About() {
  return (
    <div className="max-w-4xl space-y-4">
      <header>
        <h1 className="text-lg font-bold text-slate-100">About FraudShield AI</h1>
        <p className="text-sm text-slate-500">
          Academic mini-project: intelligent financial fraud detection with machine learning.
        </p>
      </header>

      <Section icon={Target} title="Problem statement">
        <p>
          Payment fraud is rare — roughly 0.17% of transactions in the studied dataset — yet the
          cost of a missed fraud (false negative) is high. Security teams need models that rank
          suspicious transactions for human review while keeping false alarms manageable. This
          project builds and evaluates such a system end-to-end: data validation, leakage-safe
          preprocessing, three imbalance-aware classifiers, threshold tuning, a FastAPI inference
          service, and an analyst dashboard.
        </p>
      </Section>

      <Section icon={Scale} title="Why fraud detection is difficult">
        <ul className="list-inside list-disc space-y-1">
          <li>
            <strong>Extreme class imbalance:</strong> 492 frauds vs 284,315 legitimate
            transactions. A trivial "always legitimate" classifier gets 99.83% accuracy while
            catching zero fraud.
          </li>
          <li>
            <strong>Evolving patterns:</strong> fraudsters change behaviour, so models drift over
            time and need retraining.
          </li>
          <li>
            <strong>Cost asymmetry:</strong> a missed fraud costs far more than a false alarm, so
            the decision threshold must reflect business cost, not 0.5 by default.
          </li>
          <li>
            <strong>Privacy:</strong> the published features are anonymised PCA components —
            analysts cannot inspect raw behaviour, only statistical signals.
          </li>
        </ul>
      </Section>

      <Section icon={Database} title="Dataset and its limitations">
        <p>
          The Kaggle <em>Credit Card Fraud Detection</em> dataset (MLG-ULB, 2018) contains 284,807
          European cardholder transactions over two days in September 2013, of which 492 (0.173%)
          are fraudulent. Features V1–V28 are PCA-transformed and anonymised; only Time (seconds
          since the first transaction) and Amount (EUR) are interpretable.
        </p>
        <ul className="list-inside list-disc space-y-1">
          <li>Only two days of data — results may not generalise to other periods or regions.</li>
          <li>
            No merchant, geographic, or cardholder context is available, limiting explainability.
          </li>
          <li>
            The dataset is widely used for research; strong scores on it do not prove
            production-grade performance.
          </li>
          <li>
            Probability calibration has not been verified — model scores are rankings, not true
            probabilities.
          </li>
        </ul>
      </Section>

      <Section icon={Layers} title="Algorithms used">
        <ul className="list-inside list-disc space-y-1">
          <li>
            <strong>Logistic Regression (class_weight="balanced")</strong> — fast, interpretable
            linear baseline; each feature's standardized coefficient shows its direction of
            association with fraud.
          </li>
          <li>
            <strong>Random Forest (class_weight="balanced")</strong> — non-linear ensemble of 200
            decision trees; class weighting penalises mistakes on the rare fraud class.
          </li>
          <li>
            <strong>Balanced Random Forest (imbalanced-learn)</strong> — each tree is trained on a
            bootstrapped balanced subset (all fraud + equal number of legitimate), attacking the
            imbalance directly. Resampling happens strictly inside the training loop.
          </li>
        </ul>
      </Section>

      <Section icon={GitBranch} title="Preprocessing & leakage prevention">
        <ul className="list-inside list-disc space-y-1">
          <li>1,081 duplicate rows removed before splitting (no transaction in two splits).</li>
          <li>Stratified 60/20/20 train/validation/test split (seed 42) preserving fraud ratio.</li>
          <li>
            A StandardScaler for Time and Amount is fitted on the training split only, inside the
            sklearn pipeline.
          </li>
          <li>V1–V28 are already PCA-scaled and pass through unchanged.</li>
          <li>
            Validation and test sets are never resampled, never used for hyper-parameter or
            threshold tuning.
          </li>
        </ul>
      </Section>

      <Section icon={TrendingUp} title="Evaluation metrics & model selection">
        <p>
          Accuracy is meaningless here, so models are compared by <strong>PR-AUC</strong> (area
          under the precision–recall curve), which measures how well each model ranks the rare
          class. Precision, recall, F1, ROC-AUC, confusion matrices and FP/FN counts are reported
          for every model on the validation split. The winning model's threshold is tuned on the
          validation set to maximise F1 (ties broken toward higher recall, since a missed fraud is
          costlier than a false alarm). The final numbers come from a single, untouched test-split
          evaluation.
        </p>
      </Section>

      <Section icon={Layers} title="System architecture">
        <pre className="overflow-x-auto rounded bg-navy-850 p-3 font-mono text-[11px] leading-relaxed text-slate-400">
{`[scripts/train.py]
  data/raw/creditcard.csv ─▶ validation ─▶ EDA figures
        │
        ▼
  stratified split ─▶ sklearn pipelines (LR / RF / BRF)
        │                 │
        ▼                 ▼
  artifacts/           reports/figures/
  (pipeline.joblib,    (12 PNG plots)
   metrics JSON…)

[backend FastAPI :8000]
  loads pipeline.joblib once at startup
  POST /api/predict, /api/predict/batch, /api/predict/csv
  GET  /api/model-info, /api/analytics/*, /api/samples

[frontend React + Vite :5173]
  Dashboard · Analyzer · Batch · Performance · About`}
        </pre>
      </Section>

      <Section icon={Scale} title="Limitations & ethics">
        <ul className="list-inside list-disc space-y-1">
          <li>
            Predictions are <strong>not confirmed fraud</strong> — they are model scores for
            human review. The system never blocks payments.
          </li>
          <li>
            Scores are uncalibrated: a "90%" score does not mean a 90% chance of fraud.
          </li>
          <li>
            Explanations show statistical association only; V-features cannot be mapped to real
            attributes and no causal claim is made.
          </li>
          <li>
            The training data reflects 2013 European cardholder behaviour; fairness across
            demographics cannot be audited because the dataset is anonymised.
          </li>
          <li>
            If retrained on the synthetic demo data (--demo flag), none of the metrics reflect
            real-world performance.
          </li>
        </ul>
      </Section>

      <Section icon={TrendingUp} title="Future scope">
        <ul className="list-inside list-disc space-y-1">
          <li>Probability calibration (Platt / isotonic) so scores become true probabilities.</li>
          <li>Cost-sensitive threshold selection using an explicit fraud-loss matrix.</li>
          <li>Streaming evaluation and drift detection for continual learning.</li>
          <li>Graph-based features (device, merchant, session networks) when available.</li>
          <li>Human-in-the-loop feedback channel to label reviewed transactions.</li>
        </ul>
      </Section>
    </div>
  );
}
