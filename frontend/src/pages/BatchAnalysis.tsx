import { useEffect, useRef, useState } from "react";
import { AlertTriangle, Download, FileUp, Loader2 } from "lucide-react";
import { api, ApiError } from "../services/api";
import type { CsvResponse, ModelInfo } from "../types";
import { EmptyState, ErrorBanner, Panel, RiskBadge } from "../components/ui";

const MAX_FILE_MB = 5;

function toCsv(res: CsvResponse): string {
  const header = [
    "transaction_id",
    "predicted_class",
    "predicted_label",
    "fraud_probability",
    "risk_category",
    "threshold",
    "model_version",
  ];
  const lines = [header.join(",")];
  for (const r of res.results) {
    lines.push(
      [
        r.transaction_id,
        r.predicted_class,
        `"${r.predicted_label}"`,
        r.fraud_probability,
        r.risk_category,
        r.threshold,
        r.model_version,
      ].join(",")
    );
  }
  return lines.join("\n");
}

export default function BatchAnalysis() {
  const [res, setRes] = useState<CsvResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.modelInfo().then(setModelInfo).catch(() => setModelInfo(null));
  }, []);

  const handleFile = async (file: File | undefined) => {
    if (!file) return;
    setError(null);
    setRes(null);
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setError(`'${file.name}' is not a .csv file. Please upload a CSV export.`);
      return;
    }
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setError(`File is ${(file.size / 1e6).toFixed(1)} MB — the limit is ${MAX_FILE_MB} MB (max 1000 rows).`);
      return;
    }
    setBusy(true);
    try {
      setRes(await api.predictCsv(file));
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : "Upload failed");
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const download = () => {
    if (!res) return;
    const blob = new Blob([toCsv(res)], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `fraudshield_predictions_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const previewCols = res ? Object.keys(res.preview[0] ?? {}).slice(0, 6) : [];

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-lg font-bold text-slate-100">Batch Analysis</h1>
        <p className="text-sm text-slate-500">
          Upload a CSV with the 30 feature columns ({modelInfo?.feature_schema.features.slice(0, 3).join(", ")}, …).
          A Class column is not required and is never used as a feature.
        </p>
      </header>

      <div
        className={`rounded-lg border-2 border-dashed px-4 py-10 text-center transition-colors ${
          dragOver ? "border-signal-blue bg-navy-850" : "border-navy-600 bg-navy-900"
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFile(e.dataTransfer.files[0]);
        }}
      >
        <input
          ref={inputRef}
          id="csv-input"
          type="file"
          accept=".csv"
          className="sr-only"
          onChange={(e) => handleFile(e.target.files?.[0])}
          aria-label="Choose CSV file"
        />
        <FileUp className="mx-auto h-8 w-8 text-slate-500" aria-hidden />
        <p className="mt-2 text-sm text-slate-300">
          Drag a CSV here, or{" "}
          <button
            type="button"
            className="font-medium text-signal-blue hover:underline"
            onClick={() => inputRef.current?.click()}
          >
            browse files
          </button>
        </p>
        <p className="mt-1 text-xs text-slate-500">
          .csv only · ≤ {MAX_FILE_MB} MB · ≤ 1000 rows · columns: Time, V1…V28, Amount
        </p>
      </div>

      {busy && (
        <div className="flex items-center gap-3 text-sm text-slate-400" role="status">
          <Loader2 className="h-4 w-4 animate-spin text-signal-blue" />
          Scoring transactions…
        </div>
      )}
      {error && <ErrorBanner message={error} />}

      {res && (
        <>
          {/* Summary */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <div className="panel">
              <p className="text-xs uppercase tracking-wide text-slate-500">Rows scored</p>
              <p className="mt-1 font-mono text-2xl text-slate-100">{res.summary.total}</p>
            </div>
            <div className="panel">
              <p className="text-xs uppercase tracking-wide text-slate-500">Flagged as fraud</p>
              <p className="mt-1 font-mono text-2xl text-signal-red">{res.summary.predicted_fraud}</p>
            </div>
            <div className="panel">
              <p className="text-xs uppercase tracking-wide text-slate-500">Legitimate</p>
              <p className="mt-1 font-mono text-2xl text-signal-green">{res.summary.predicted_legitimate}</p>
            </div>
            <div className="panel">
              <p className="text-xs uppercase tracking-wide text-slate-500">Flagged rate</p>
              <p className="mt-1 font-mono text-2xl text-signal-amber">
                {res.summary.fraud_rate_percentage}%
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="flex flex-wrap items-center gap-3">
              {Object.entries(res.summary.risk_distribution).map(([k, v]) => (
                <span key={k} className="flex items-center gap-1.5">
                  <RiskBadge risk={k} />
                  <span className="font-mono text-xs text-slate-500">{v}</span>
                </span>
              ))}
            </div>
            <button type="button" className="btn-secondary ml-auto" onClick={download}>
              <Download className="h-4 w-4" />
              Download predictions CSV
            </button>
          </div>

          {res.invalid_rows.length > 0 && (
            <div className="flex items-start gap-2 rounded-md border border-signal-amber/40 bg-signal-amber/10 px-4 py-3 text-sm text-signal-amber">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <div>
                <strong>{res.invalid_rows.length} row(s) skipped</strong> (valid rows were still
                scored):
                <ul className="mt-1 list-inside list-disc text-xs">
                  {res.invalid_rows.slice(0, 5).map((r) => (
                    <li key={r.row}>
                      Row {r.row}: {r.reason}
                    </li>
                  ))}
                  {res.invalid_rows.length > 5 && <li>… and {res.invalid_rows.length - 5} more</li>}
                </ul>
              </div>
            </div>
          )}
          {res.ignored_columns.length > 0 && (
            <p className="text-xs text-slate-500">
              Ignored non-feature columns: {res.ignored_columns.join(", ")} — labels and other
              metadata are never used as prediction inputs.
            </p>
          )}

          {/* Preview */}
          <Panel title={`Imported rows — preview (first ${Math.min(10, res.preview.length)})`}>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="border-b border-navy-700 text-slate-500">
                  <tr>
                    <th className="py-2 pr-3">#</th>
                    {previewCols.map((c) => (
                      <th key={c} className="py-2 pr-3 font-mono">
                        {c}
                      </th>
                    ))}
                    <th className="py-2 pr-3">…</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {res.preview.map((row, i) => (
                    <tr key={i} className="border-b border-navy-850">
                      <td className="py-1.5 pr-3 text-slate-500">{i + 1}</td>
                      {previewCols.map((c) => (
                        <td key={c} className="py-1.5 pr-3 text-slate-300">
                          {Number(row[c]).toFixed(2)}
                        </td>
                      ))}
                      <td className="py-1.5 text-slate-500">V7…V28, Amount</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

          {/* Per-transaction results */}
          <Panel title={`Per-transaction predictions (${res.results.length})`}>
            <div className="max-h-96 overflow-y-auto">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 border-b border-navy-700 bg-navy-900 text-slate-500">
                  <tr>
                    <th className="py-2 pr-3">#</th>
                    <th className="py-2 pr-3">Transaction</th>
                    <th className="py-2 pr-3">Prediction</th>
                    <th className="py-2 pr-3">Probability</th>
                    <th className="py-2 pr-3">Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {res.results.map((r, i) => (
                    <tr key={r.transaction_id} className="border-b border-navy-850">
                      <td className="py-1.5 pr-3 font-mono text-slate-500">{i + 1}</td>
                      <td className="py-1.5 pr-3 font-mono text-slate-500">{r.transaction_id}</td>
                      <td
                        className={`py-1.5 pr-3 font-medium ${
                          r.predicted_class === 1 ? "text-signal-red" : "text-signal-green"
                        }`}
                      >
                        {r.predicted_label}
                      </td>
                      <td className="py-1.5 pr-3 font-mono text-slate-300">
                        {(r.fraud_probability * 100).toFixed(2)}%
                      </td>
                      <td className="py-1.5 pr-3">
                        <RiskBadge risk={r.risk_category} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}

      {!res && !error && !busy && (
        <EmptyState message="No file uploaded yet. Predictions appear here once a CSV is scored." />
      )}
    </div>
  );
}
