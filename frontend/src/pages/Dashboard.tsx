import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AlertTriangle, ShieldCheck } from "lucide-react";
import { api, ApiError } from "../services/api";
import type { ConfusionMatrixResponse, ModelComparison, ModelInfo, Overview } from "../types";
import { ErrorBanner, Loading, MetricBar, Panel, StatCard } from "../components/ui";

const RISK_COLORS: Record<string, string> = {
  low: "#34d399",
  medium: "#fbbf24",
  high: "#f87171",
};

const fmt = (n: number | null | undefined) =>
  n === null || n === undefined ? "—" : n.toLocaleString("en-US");

export default function Dashboard() {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [comparison, setComparison] = useState<ModelComparison | null>(null);
  const [cm, setCm] = useState<ConfusionMatrixResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.overview(), api.comparison(), api.confusionMatrix(), api.modelInfo()])
      .then(([o, c, m, mi]) => {
        setOverview(o);
        setComparison(c);
        setCm(m);
        setModelInfo(mi);
      })
      .catch((e: ApiError) => setError(e.detail))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Loading label="Loading dashboard…" />;
  if (error) return <ErrorBanner message={error} />;
  if (!overview || !comparison || !cm) return <ErrorBanner message="No data returned." />;

  const ds = overview.dataset;
  const test = comparison.test_metrics_selected_model;
  const modelRows = Object.entries(comparison.models).map(([name, m]) => ({
    name: name.replaceAll("_", " "),
    "PR-AUC": m.validation.pr_auc,
    "ROC-AUC": m.validation.roc_auc,
    F1: m.validation.f1,
  }));
  const riskData = Object.entries(comparison.test_risk_distribution ?? {}).map(
    ([name, value]) => ({ name: `${name} risk`, value, key: name })
  );

  return (
    <div className="space-y-4">
      <header className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-lg font-bold text-slate-100">Risk Monitoring Dashboard</h1>
          <p className="text-sm text-slate-500">
            Model {comparison.selected_model.replaceAll("_", " ")} · v{overview.model_version} ·
            threshold {overview.decision_threshold.toFixed(3)} · dataset{" "}
            {ds.data_source.startsWith("demo") ? "SYNTHETIC DEMO" : "Kaggle MLG-ULB"}
          </p>
        </div>
        <span className="rounded border border-navy-600 bg-navy-850 px-2 py-1 font-mono text-[11px] text-slate-400">
          Held-out test set: {fmt(modelInfo?.split_sizes?.test)} transactions
        </span>
      </header>

      {overview.is_demo && (
        <div className="flex items-start gap-2 rounded-md border border-signal-amber/40 bg-signal-amber/10 px-4 py-3 text-sm text-signal-amber">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          This model was trained on clearly-labelled synthetic demo data. All metrics describe
          the synthetic dataset only and say nothing about real-world fraud detection.
        </div>
      )}

      {/* Headline numbers — all sourced from dataset_stats.json / evaluation artifacts */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Transactions analysed"
          value={fmt(ds.total_transactions)}
          sub={`Kaggle dataset, ${ds.duplicate_rows_removed ?? 0} duplicate rows removed`}
          tone="blue"
        />
        <StatCard
          label="Actual fraud cases in dataset"
          value={fmt(ds.fraud_cases)}
          sub={`${ds.fraud_percentage.toFixed(3)}% of all transactions`}
          tone="red"
        />
        <StatCard
          label="Legitimate transactions"
          value={fmt(ds.legitimate_cases)}
          sub="Class = 0"
          tone="green"
        />
        <StatCard
          label="Fraud detection rate (test recall)"
          value={`${(test.recall * 100).toFixed(1)}%`}
          sub={`${test.true_positives} of ${test.true_positives + test.false_negatives} test frauds flagged`}
          tone="amber"
        />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {/* Model quality */}
        <Panel title="Selected model — held-out test performance">
          <div className="space-y-3">
            <MetricBar label="Precision" value={test.precision} />
            <MetricBar label="Recall (fraud detection rate)" value={test.recall} />
            <MetricBar label="F1-score" value={test.f1} />
            <MetricBar label="PR-AUC (average precision)" value={test.pr_auc} />
            <MetricBar label="ROC-AUC" value={test.roc_auc} />
          </div>
          <p className="mt-3 text-xs leading-relaxed text-slate-500">
            Measured once on the held-out test split at the validation-tuned threshold. The test
            split was never used for model selection or threshold tuning.
          </p>
        </Panel>

        {/* Confusion matrix */}
        <Panel title={`Confusion matrix — test set (threshold ${cm.threshold.toFixed(3)})`}>
          <div className="grid grid-cols-[auto_1fr_1fr] gap-1 text-sm">
            <div />
            <div className="pb-1 text-center text-xs text-slate-500">Pred: Legit</div>
            <div className="pb-1 text-center text-xs text-slate-500">Pred: Flagged</div>
            <div className="pr-2 text-xs text-slate-400">Actual: Legit</div>
            <div className="rounded bg-navy-850 p-3 text-center font-mono text-signal-green">
              {fmt(cm.true_negatives)}
            </div>
            <div className="rounded bg-navy-850 p-3 text-center font-mono text-signal-amber">
              {fmt(cm.false_positives)}
            </div>
            <div className="pr-2 text-xs text-slate-400">Actual: Fraud</div>
            <div className="rounded bg-navy-850 p-3 text-center font-mono text-signal-red">
              {fmt(cm.false_negatives)}
            </div>
            <div className="rounded bg-navy-850 p-3 text-center font-mono text-signal-green">
              {fmt(cm.true_positives)}
            </div>
          </div>
          <p className="mt-3 text-xs text-slate-500">{cm.note}</p>
        </Panel>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {/* Risk distribution on the test set */}
        <Panel title="Transaction risk distribution (test set, real counts)">
          {riskData.length === 0 ? (
            <p className="text-sm text-slate-500">Risk distribution not recorded in artifacts.</p>
          ) : (
            <div className="flex items-center gap-4">
              <ResponsiveContainer width="50%" height={200}>
                <PieChart>
                  <Pie data={riskData} dataKey="value" nameKey="name" innerRadius={40} outerRadius={70}>
                    {riskData.map((d) => (
                      <Cell key={d.key} fill={RISK_COLORS[d.key] ?? "#64748b"} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ background: "#111a30", border: "1px solid #1e2c4d", borderRadius: 6 }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <ul className="space-y-2 text-sm">
                {riskData.map((d) => (
                  <li key={d.key} className="flex items-center gap-2">
                    <span
                      className="inline-block h-2.5 w-2.5 rounded-full"
                      style={{ background: RISK_COLORS[d.key] ?? "#64748b" }}
                    />
                    <span className="capitalize text-slate-300">{d.key}</span>
                    <span className="font-mono text-slate-500">{fmt(d.value)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Panel>

        {/* Model comparison (validation) */}
        <Panel title="Model comparison — validation PR-AUC (selection metric)">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={modelRows} margin={{ left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2c4d" />
              <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} interval={0} />
              <YAxis domain={[0, 1]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: "#111a30", border: "1px solid #1e2c4d", borderRadius: 6 }}
              />
              <Bar dataKey="PR-AUC" fill="#38bdf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <p className="mt-2 text-xs text-slate-500">
            Winner: <strong className="text-slate-300">{comparison.selected_model.replaceAll("_", " ")}</strong>{" "}
            — {comparison.selection_reason}
          </p>
        </Panel>
      </div>

      <p className="flex items-start gap-2 text-xs leading-relaxed text-slate-500">
        <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-signal-blue" />
        {overview.interpretation} All figures on this page are computed from the saved training
        artifacts of the real Kaggle dataset — nothing is simulated.
      </p>
    </div>
  );
}
