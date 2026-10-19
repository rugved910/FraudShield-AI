// Thin API client. Uses relative URLs — the Vite dev server proxies /api to
// the FastAPI backend (see vite.config.ts), so no CORS setup is needed in dev.
// In production, serve the built frontend behind the same origin as the API.

import type {
  ApiInfo,
  BatchResponse,
  ConfusionMatrixResponse,
  CsvResponse,
  CurvesResponse,
  FeatureImportance,
  Health,
  ModelComparison,
  ModelInfo,
  Overview,
  SamplesResponse,
  SinglePrediction,
  TransactionFeatures,
} from "../types";

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError(0, "Cannot reach the backend. Is the API running on port 8000?");
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (Array.isArray(body?.detail)) {
        detail = body.detail
          .map((d: { loc?: string[]; msg?: string }) => `${d.loc?.slice(1).join(".")}: ${d.msg}`)
          .join("; ");
      } else if (typeof body?.detail === "object" && body.detail !== null) {
        // FastAPI validation errors keyed by field
        detail = Object.entries(body.detail as Record<string, string[]>)
          .map(([k, v]) => `${k}: ${v.join(", ")}`)
          .join("; ");
      } else if (typeof body?.error === "string") detail = `${body.error}: ${body.detail ?? ""}`;
    } catch {
      /* keep default message */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  info: () => request<ApiInfo>("/"),
  health: () => request<Health>("/health"),
  modelInfo: () => request<ModelInfo>("/api/model-info"),
  predict: (tx: TransactionFeatures) =>
    request<SinglePrediction>("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(tx),
    }),
  predictBatch: (transactions: TransactionFeatures[]) =>
    request<BatchResponse>("/api/predict/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transactions }),
    }),
  predictCsv: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<CsvResponse>("/api/predict/csv", { method: "POST", body: form });
  },
  samples: () => request<SamplesResponse>("/api/samples"),
  overview: () => request<Overview>("/api/analytics/overview"),
  comparison: () => request<ModelComparison>("/api/analytics/model-comparison"),
  confusionMatrix: () => request<ConfusionMatrixResponse>("/api/analytics/confusion-matrix"),
  featureImportance: () => request<FeatureImportance>("/api/analytics/feature-importance"),
  curves: (split: "validation" | "test") =>
    request<CurvesResponse>(`/api/analytics/curves?split=${split}`),
};
