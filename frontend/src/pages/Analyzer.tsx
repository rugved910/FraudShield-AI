import { useCallback, useEffect, useState } from "react";
import { FlaskConical, Loader2, ScanSearch } from "lucide-react";
import { api, ApiError } from "../services/api";
import type { ModelInfo, SamplesResponse, SinglePrediction, TransactionFeatures } from "../types";
import { ErrorBanner, Panel, RiskBadge } from "../components/ui";

const V_FIELDS = Array.from({ length: 28 }, (_, i) => `V${i + 1}`);

function emptyForm(): Record<string, string> {
  const f: Record<string, string> = { Time: "", Amount: "" };
  for (const v of V_FIELDS) f[v] = "";
  return f;
}

export default function Analyzer() {
  const [form, setForm] = useState<Record<string, string>>(emptyForm());
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [result, setResult] = useState<SinglePrediction | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [samples, setSamples] = useState<SamplesResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [showV, setShowV] = useState(false);

  useEffect(() => {
    api.samples().then(setSamples).catch(() => setSamples(null));
    api.modelInfo().then(setModelInfo).catch(() => setModelInfo(null));
  }, []);

  const validate = useCallback(() => {
    const e: Record<string, string> = {};
    for (const [k, v] of Object.entries(form)) {
      if (v.trim() === "") {
        e[k] = "Required";
        continue;
      }
      const n = Number(v);
      if (!Number.isFinite(n)) {
        e[k] = "Must be a finite number";
        continue;
      }
      if (k === "Amount" && n < 0) e[k] = "Amount cannot be negative";
      if (k === "Time" && n < 0) e[k] = "Time cannot be negative";
      if (k.startsWith("V") && Math.abs(n) > 500) e[k] = "Out of expected PCA range (±500)";
    }
    setErrors(e);
    return Object.keys(e).length === 0;
  }, [form]);

  const submit = async (ev: React.FormEvent) => {
    ev.preventDefault();
    setApiError(null);
    setResult(null);
    if (!validate()) return;
    setBusy(true);
    try {
      const payload = Object.fromEntries(
        Object.entries(form).map(([k, v]) => [k, Number(v)])
      ) as TransactionFeatures;
      setResult(await api.predict(payload));
    } catch (err) {
      setApiError(err instanceof ApiError ? err.detail : "Unexpected error");
    } finally {
      setBusy(false);
    }
  };

  const loadSample = (idx: number) => {
    const s = samples?.samples[idx];
    if (!s) return;
    setForm(
      Object.fromEntries(Object.entries(s.features).map(([k, v]) => [k, String(v)]))
    );
    setErrors({});
    setResult(null);
    setApiError(null);
  };

  const field = (name: string, step = "any") => (
    <div key={name}>
      <label htmlFor={`f-${name}`} className="input-label">
        {name}
      </label>
      <input
        id={`f-${name}`}
        name={name}
        className="input-field"
        value={form[name]}
        onChange={(e) => {
          setForm({ ...form, [name]: e.target.value });
          if (errors[name]) setErrors({ ...errors, [name]: "" });
        }}
        inputMode="decimal"
        step={step}
        aria-invalid={Boolean(errors[name])}
        aria-describedby={errors[name] ? `err-${name}` : undefined}
      />
      {errors[name] && (
        <p id={`err-${name}`} className="mt-0.5 text-[11px] text-signal-red">
          {errors[name]}
        </p>
      )}
    </div>
  );

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-lg font-bold text-slate-100">Transaction Analyzer</h1>
        <p className="text-sm text-slate-500">
          Score a single transaction with the trained model
          {modelInfo ? ` (${modelInfo.model_name}, v${modelInfo.model_version})` : ""}.
        </p>
      </header>

      <div className="rounded-md border border-navy-700 bg-navy-850 px-4 py-3 text-xs leading-relaxed text-slate-400">
        <strong className="text-slate-300">Educational input note:</strong> the V1–V28 columns are
        anonymised PCA components published with the Kaggle dataset. They cannot be interpreted as
        merchant, location, or cardholder attributes, and this form can never reveal hidden
        personal details. Amount is in EUR; Time is seconds since the first transaction of the
        dataset's 48-hour window.
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[3fr_2fr]">
        <Panel title="Input features">
          {samples && samples.samples.length > 0 && (
            <div className="mb-4 flex flex-wrap items-center gap-2">
              <span className="text-xs text-slate-500">Load a real held-out sample:</span>
              {samples.samples.map((s, i) => (
                <button
                  key={i}
                  type="button"
                  className="btn-secondary !px-2.5 !py-1 !text-xs"
                  onClick={() => loadSample(i)}
                >
                  <FlaskConical className="h-3 w-3" />
                  #{i + 1} ({s.true_label === 1 ? "actual fraud" : "legitimate"})
                </button>
              ))}
            </div>
          )}
          <form onSubmit={submit} noValidate>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
              {field("Time")}
              {field("Amount")}
            </div>

            <button
              type="button"
              className="mt-3 text-xs font-medium text-signal-blue hover:underline"
              onClick={() => setShowV(!showV)}
              aria-expanded={showV}
            >
              {showV ? "Hide" : "Show"} anonymised PCA features V1–V28 ▾
            </button>

            {showV && (
              <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
                {V_FIELDS.map((v) => field(v))}
              </div>
            )}

            <div className="mt-4 flex items-center gap-3">
              <button type="submit" className="btn-primary" disabled={busy}>
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <ScanSearch className="h-4 w-4" />}
                Predict
              </button>
              <button
                type="button"
                className="btn-secondary"
                onClick={() => {
                  setForm(emptyForm());
                  setErrors({});
                  setResult(null);
                  setApiError(null);
                }}
              >
                Clear
              </button>
            </div>
          </form>
        </Panel>

        <div className="space-y-4">
          {apiError && <ErrorBanner message={apiError} />}

          {result ? (
            <Panel title="Prediction result">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span
                    className={`text-lg font-semibold ${
                      result.predicted_class === 1 ? "text-signal-red" : "text-signal-green"
                    }`}
                  >
                    {result.predicted_label}
                  </span>
                  <RiskBadge risk={result.risk_category} />
                </div>

                <div>
                  <div className="mb-1 flex justify-between text-xs">
                    <span className="text-slate-400">Fraud probability (raw model score)</span>
                    <span className="font-mono text-slate-200">
                      {(result.fraud_probability * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="relative h-3 w-full rounded bg-navy-800">
                    <div
                      className={`h-full rounded ${
                        result.risk_category === "high"
                          ? "bg-signal-red"
                          : result.risk_category === "medium"
                          ? "bg-signal-amber"
                          : "bg-signal-green"
                      }`}
                      style={{ width: `${Math.min(100, result.fraud_probability * 100)}%` }}
                    />
                    <div
                      className="absolute top-[-3px] h-[18px] w-0.5 bg-slate-300"
                      style={{ left: `${Math.min(100, result.threshold * 100)}%` }}
                      title={`Decision threshold ${result.threshold.toFixed(3)}`}
                    />
                  </div>
                  <p className="mt-1 text-[11px] text-slate-500">
                    White marker = decision threshold {result.threshold.toFixed(4)} ·
                    model v{result.model_version} · tx {result.transaction_id}
                  </p>
                </div>

                <div>
                  <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Why the model scored it this way
                  </h3>
                  <ul className="space-y-2">
                    {result.explanation
                      .filter((e) => e.feature !== "_note")
                      .map((e) => (
                        <li key={e.feature} className="rounded bg-navy-850 px-3 py-2 text-xs">
                          <div className="flex justify-between font-mono">
                            <span className="text-slate-200">{e.feature}</span>
                            <span
                              className={
                                e.contribution >= 0 ? "text-signal-red" : "text-signal-green"
                              }
                            >
                              {e.contribution >= 0 ? "+" : ""}
                              {e.contribution.toFixed(4)} → {e.direction}
                            </span>
                          </div>
                          <p className="mt-1 text-slate-500">
                            value {e.value} · {e.interpretation}
                          </p>
                        </li>
                      ))}
                  </ul>
                  <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
                    {result.explanation[result.explanation.length - 1]?.interpretation}
                  </p>
                </div>

                <p className="text-[11px] leading-relaxed text-slate-500">
                  <strong className="text-slate-400">Probability note:</strong>{" "}
                  {result.probability_note}
                </p>
                <p className="rounded border border-navy-700 bg-navy-850 p-2 text-[11px] leading-relaxed text-slate-500">
                  {result.disclaimer}
                </p>
              </div>
            </Panel>
          ) : (
            !apiError && (
              <div className="rounded-md border border-dashed border-navy-600 px-4 py-10 text-center text-sm text-slate-500">
                Submit a transaction to see its prediction, risk category and explanation here.
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
}
