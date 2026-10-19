import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, ApiError } from "../services/api";
import type {
  CurvesResponse,
  FeatureImportance,
  ModelComparison,
  ModelInfo,
} from "../types";
import { ErrorBanner, Loading, Panel } from "../components/ui";

const MODEL_COLORS: Record<string, string> = {
  logistic_regression: "#38bdf8",
  random_forest: "#fbbf24",
  balanced_random_forest: "#f87171",
};

const fmt = (n: number | null | undefined, digits = 4) =>
  n === null || n === undefined ? "—" : n.toFixed(digits);

export default function Performance() {
  const [comparison, setComparison] = useState<ModelComparison | null>(null);
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [valCurves, setValCurves] = useState<CurvesResponse | null>(null);
  const [testCurves, setTestCurves] = useState<CurvesResponse | null>(null);
  const [fi, setFi] = useState<FeatureImportance | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<"validation" | "test">("validation");

  useEffect(() => {
    Promise.all([
      api.comparison(),
      api.modelInfo(),
      api.curves("validation"),
      api.curves("test"),
      api.featureImportance(),
    ])
      .then(([c, i, vc, tc, f]) => {
        setComparison(c);
        setInfo(i);
        setValCurves(vc);
        setTestCurves(tc);
        setFi(f);
      })
      .catch((e: ApiError) => setError(e.detail))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Loading label="Loading evaluation artifacts…" />;
  if (error) return <ErrorBanner message={error} />;
  if (!comparison) return <ErrorBanner message="No comparison data available." />;

  const activeCurves = tab === "validation" ? valCurves : testCurves;
  const curveEntries = Object.entries(activeCurves?.curves ?? {});

  // Build tidy arrays for recharts
  const buildPr = (c: { recall?: number[]; precision?: number[] }) =>
    (c.recall ?? []).map((r, i) => ({ recall: r, precision: c.precision?.[i] ?? 0 }));
  const buildRoc = (c: { fpr?: number[]; tpr?: number[] }) =>
    (c.fpr ?? []).map((f, i) => ({ fpr: f, tpr: c.tpr?.[i] ?? 0 }));

  const fiTop = fi ? fi.features.slice(0, 15).reverse() : [];
  const maxFi = fi ? Math.max(...fi.features.map((f) => f.importance)) : 1;

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-lg font-bold text-slate-100">Model Performance</h1>
        <p className="text-sm text-slate-500">
          Selection: <strong className="text-slate-300">{comparison.selected_model.replaceAll("_", " ")}</strong>{" "}
          by {comparison.selection_metric.replaceAll("_", " ")} · trained v{info?.model_version} ·
          seed {info?.random_seed}
        </p>
      </header>

      {/* Threshold explanation */}
      <Panel title="Threshold selection — how the decision cut-off was chosen">
        <p className="text-sm leading-relaxed text-slate-300">{comparison.selection_reason}</p>
        <div className="mt-3 grid grid-cols-1 gap-3 text-sm sm:grid-cols-3">
          <div className="rounded bg-navy-850 p-3">
            <p className="text-xs text-slate-500">Chosen threshold</p>
            <p className="font-mono text-xl text-signal-blue">
              {fmt(comparison.decision_threshold, 4)}
            </p>
          </div>
          <div className="rounded bg-navy-850 p-3">
            <p className="text-xs text-slate-500">Tuning method</p>
            <p className="mt-0.5 text-xs leading-snug text-slate-300">
              {info?.threshold_tuning?.method ?? "—"}
            </p>
          </div>
          <div className="rounded bg-navy-850 p-3">
            <p className="text-xs text-slate-500">Tuned on</p>
            <p className="mt-0.5 text-xs text-slate-300">
              {info?.threshold_tuning?.tuned_on ?? "—"} split — never the test split
            </p>
          </div>
        </div>
      </Panel>

      {/* Metrics table */}
      <Panel title="Model comparison — validation metrics (threshold 0.5, fair comparison)">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-navy-700 text-slate-500">
              <tr>
                <th className="py-2 pr-4">Model</th>
                <th className="py-2 pr-4">Precision</th>
                <th className="py-2 pr-4">Recall</th>
                <th className="py-2 pr-4">F1</th>
                <th className="py-2 pr-4">ROC-AUC</th>
                <th className="py-2 pr-4">PR-AUC</th>
                <th className="py-2 pr-4">FP</th>
                <th className="py-2 pr-4">FN</th>
                <th className="py-2 pr-4">Train (s)</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(comparison.models).map(([name, m]) => (
                <tr
                  key={name}
                  className={`border-b border-navy-850 ${
                    name === comparison.selected_model ? "bg-navy-850/60" : ""
                  }`}
                >
                  <td className="py-2 pr-4 font-medium text-slate-200">
                    {name.replaceAll("_", " ")}
                    {name === comparison.selected_model && (
                      <span className="ml-2 rounded bg-signal-blue/15 px-1.5 py-0.5 text-[10px] text-signal-blue">
                        selected
                      </span>
                    )}
                  </td>
                  <td className="py-2 pr-4 font-mono text-slate-300">{fmt(m.validation.precision, 3)}</td>
                  <td className="py-2 pr-4 font-mono text-slate-300">{fmt(m.validation.recall, 3)}</td>
                  <td className="py-2 pr-4 font-mono text-slate-300">{fmt(m.validation.f1, 3)}</td>
                  <td className="py-2 pr-4 font-mono text-slate-300">{fmt(m.validation.roc_auc, 3)}</td>
                  <td className="py-2 pr-4 font-mono text-signal-blue">{fmt(m.validation.pr_auc, 3)}</td>
                  <td className="py-2 pr-4 font-mono text-signal-amber">{m.validation.false_positives}</td>
                  <td className="py-2 pr-4 font-mono text-signal-red">{m.validation.false_negatives}</td>
                  <td className="py-2 pr-4 font-mono text-slate-500">{m.train_seconds ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {/* Test metrics of the winner */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Panel title={`Held-out TEST metrics — ${comparison.selected_model.replaceAll("_", " ")} @ tuned threshold`}>
          <table className="w-full text-left text-xs">
            <tbody>
              {(
                [
                  ["Threshold", fmt(comparison.test_metrics_selected_model.threshold, 4)],
                  ["Precision", fmt(comparison.test_metrics_selected_model.precision)],
                  ["Recall", fmt(comparison.test_metrics_selected_model.recall)],
                  ["F1-score", fmt(comparison.test_metrics_selected_model.f1)],
                  ["PR-AUC", fmt(comparison.test_metrics_selected_model.pr_auc)],
                  ["ROC-AUC", fmt(comparison.test_metrics_selected_model.roc_auc)],
                  [
                    "Confusion matrix",
                    `TN ${comparison.test_metrics_selected_model.true_negatives} · FP ${comparison.test_metrics_selected_model.false_positives} · FN ${comparison.test_metrics_selected_model.false_negatives} · TP ${comparison.test_metrics_selected_model.true_positives}`,
                  ],
                ] as [string, string][]
              ).map(([k, v]) => (
                <tr key={k} className="border-b border-navy-850">
                  <td className="py-2 pr-4 text-slate-500">{k}</td>
                  <td className="py-2 font-mono text-slate-200">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-xs leading-relaxed text-slate-500">
            These test numbers were computed <strong className="text-slate-400">once</strong>, after
            the model and threshold were frozen using validation data only. Test metrics at the
            default 0.5 threshold are also recorded in the artifacts for reference. Training,
            validation and test sets are disjoint — no transaction appears in more than one split.
          </p>
        </Panel>

        <Panel title="Dataset imbalance statistics">
          <ul className="space-y-2 text-sm">
            <li className="flex justify-between border-b border-navy-850 pb-2">
              <span className="text-slate-400">Transactions (after dedup)</span>
              <span className="font-mono text-slate-200">{info?.split_sizes
                ? (info.split_sizes.train + info.split_sizes.validation + info.split_sizes.test).toLocaleString()
                : "—"}</span>
            </li>
            <li className="flex justify-between border-b border-navy-850 pb-2">
              <span className="text-slate-400">Fraud rate in source dataset</span>
              <span className="font-mono text-signal-red">~0.17% (492 / 284,807)</span>
            </li>
            <li className="flex justify-between border-b border-navy-850 pb-2">
              <span className="text-slate-400">Train / validation / test</span>
              <span className="font-mono text-slate-200">
                {info?.split_sizes
                  ? `${info.split_sizes.train.toLocaleString()} / ${info.split_sizes.validation.toLocaleString()} / ${info.split_sizes.test.toLocaleString()}`
                  : "—"}
              </span>
            </li>
            <li className="flex justify-between border-b border-navy-850 pb-2">
              <span className="text-slate-400">Split strategy</span>
              <span className="text-slate-300">Stratified (fraud ratio preserved)</span>
            </li>
            <li className="flex justify-between">
              <span className="text-slate-400">Resampling</span>
              <span className="text-slate-300">
                Balanced RF undersamples majority <em>inside training folds only</em>
              </span>
            </li>
          </ul>
          <p className="mt-3 text-xs leading-relaxed text-slate-500">
            With 0.17% fraud, a model that always predicts "legitimate" scores 99.83% accuracy —
            which is why accuracy is never used to select or compare models here. PR-AUC and fraud
            recall are the honest measures.
          </p>
        </Panel>
      </div>

      {/* Curves */}
      <Panel
        title="Ranking curves"
        right={
          <div className="flex gap-1 rounded-md border border-navy-700 p-0.5">
            {(["validation", "test"] as const).map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setTab(t)}
                className={`rounded px-2.5 py-1 text-xs capitalize ${
                  tab === t ? "bg-navy-700 text-signal-blue" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        }
      >
        {!activeCurves ? (
          <p className="text-sm text-slate-500">Curve data unavailable in artifacts.</p>
        ) : (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* PR curves */}
            <div>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                Precision–Recall ({tab})
              </h3>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart margin={{ left: -20, right: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e2c4d" />
                  <XAxis
                    dataKey="recall"
                    type="number"
                    domain={[0, 1]}
                    tick={{ fill: "#94a3b8", fontSize: 10 }}
                    label={{ value: "Recall", position: "insideBottom", fill: "#64748b", fontSize: 11 }}
                  />
                  <YAxis
                    type="number"
                    domain={[0, 1]}
                    tick={{ fill: "#94a3b8", fontSize: 10 }}
                  />
                  <Tooltip
                    contentStyle={{ background: "#111a30", border: "1px solid #1e2c4d", borderRadius: 6 }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {tab === "validation" ? (
                    curveEntries.map(([name, c]) => (
                      <Line
                        key={name}
                        data={buildPr(c)}
                        dataKey="precision"
                        name={`${name} (AP=${(c.ap ?? 0).toFixed(3)})`}
                        dot={false}
                        stroke={MODEL_COLORS[name] ?? "#38bdf8"}
                        strokeWidth={2}
                      />
                    ))
                  ) : (
                    <Line
                      data={buildPr(Object.values(activeCurves.curves)[0])}
                      dataKey="precision"
                      name="test set"
                      dot={false}
                      stroke="#38bdf8"
                      strokeWidth={2}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* ROC curves */}
            <div>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                ROC ({tab})
              </h3>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart margin={{ left: -20, right: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e2c4d" />
                  <XAxis
                    dataKey="fpr"
                    type="number"
                    domain={[0, 1]}
                    tick={{ fill: "#94a3b8", fontSize: 10 }}
                    label={{ value: "False Positive Rate", position: "insideBottom", fill: "#64748b", fontSize: 11 }}
                  />
                  <YAxis type="number" domain={[0, 1]} tick={{ fill: "#94a3b8", fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{ background: "#111a30", border: "1px solid #1e2c4d", borderRadius: 6 }}
                  />
                  <ReferenceLine
                    segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                    stroke="#475569"
                    strokeDasharray="4 4"
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {tab === "validation" ? (
                    curveEntries.map(([name, c]) => (
                      <Line
                        key={name}
                        data={buildRoc(c)}
                        dataKey="tpr"
                        name={`${name} (AUC=${(c.auc ?? 0).toFixed(3)})`}
                        dot={false}
                        stroke={MODEL_COLORS[name] ?? "#38bdf8"}
                        strokeWidth={2}
                      />
                    ))
                  ) : (
                    <Line
                      data={buildRoc(Object.values(activeCurves.curves)[0])}
                      dataKey="tpr"
                      name="test set"
                      dot={false}
                      stroke="#38bdf8"
                      strokeWidth={2}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
        <p className="mt-2 text-xs text-slate-500">
          The dashed diagonal in the ROC chart is random guessing. Curves closer to the top-left
          (ROC) or top-right (PR) are better; with heavy imbalance the PR curve is the more
          informative of the two.
        </p>
      </Panel>

      {/* Feature importance */}
      {fi && (
        <Panel title={`Feature importance — ${fi.model.replaceAll("_", " ")} (${fi.importance_type})`}>
          <div className="space-y-1.5">
            {fiTop.map((f) => (
              <div key={f.feature} className="flex items-center gap-3 text-xs">
                <span className="w-12 shrink-0 text-right font-mono text-slate-400">{f.feature}</span>
                <div className="h-4 flex-1 rounded bg-navy-850">
                  <div
                    className="h-full rounded bg-signal-blue/80"
                    style={{ width: `${(f.importance / maxFi) * 100}%` }}
                  />
                </div>
                <span className="w-16 shrink-0 font-mono text-slate-500">
                  {f.importance.toFixed(4)}
                </span>
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs leading-relaxed text-slate-500">{fi.note}</p>
        </Panel>
      )}
    </div>
  );
}
