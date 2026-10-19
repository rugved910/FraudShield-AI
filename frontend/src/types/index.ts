// Shared type definitions mirroring the FastAPI response schemas.

export interface ApiInfo {
  name: string;
  version: string;
  description: string;
  documentation: string;
  endpoints: string[];
  disclaimer: string;
}

export interface Health {
  status: string;
  model_ready: boolean;
  model_version: string | null;
  model_name: string | null;
  data_source: string | null;
  is_demo: boolean;
  loaded_at: string | null;
  detail: string | null;
}

export interface TransactionFeatures {
  Time: number;
  Amount: number;
  [key: string]: number; // V1..V28
}

export interface ExplanationItem {
  feature: string;
  value: number;
  contribution: number;
  direction: string;
  interpretation: string;
}

export interface PredictionResult {
  transaction_id: string;
  predicted_class: number;
  predicted_label: string;
  fraud_probability: number;
  risk_category: "low" | "medium" | "high";
  threshold: number;
  model_version: string;
  probability_note: string;
}

export interface SinglePrediction extends PredictionResult {
  explanation: ExplanationItem[];
  disclaimer: string;
}

export interface BatchSummary {
  total: number;
  predicted_fraud: number;
  predicted_legitimate: number;
  fraud_rate_percentage: number;
  risk_distribution: Record<string, number>;
}

export interface BatchResponse {
  results: PredictionResult[];
  summary: BatchSummary;
  model_version: string;
  threshold: number;
}

export interface InvalidRow {
  row: number;
  reason: string;
}

export interface CsvResponse extends BatchResponse {
  preview: Record<string, number>[];
  invalid_rows: InvalidRow[];
  ignored_columns: string[];
  filename: string;
}

export interface ModelMetrics {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  pr_auc: number;
  true_negatives: number;
  false_positives: number;
  false_negatives: number;
  true_positives: number;
}

export interface ModelComparison {
  selection_metric: string;
  selection_reason: string;
  selected_model: string;
  decision_threshold: number;
  models: Record<string, { validation: ModelMetrics; validation_tuned?: ModelMetrics; train_seconds?: number }>;
  test_metrics_selected_model: ModelMetrics;
  test_metrics_selected_model_at_threshold_0_5?: ModelMetrics;
  test_risk_distribution?: Record<string, number>;
  total_training_seconds?: number;
}

export interface DatasetStats {
  data_source: string;
  total_transactions: number;
  fraud_cases: number;
  legitimate_cases: number;
  fraud_percentage: number;
  duplicate_rows_removed?: number;
  missing_values?: number;
  amount_stats?: Record<string, number>;
  time_span_hours?: number;
}

export interface Overview {
  dataset: DatasetStats;
  selected_model: string | null;
  model_version: string;
  decision_threshold: number;
  is_demo: boolean;
  test_metrics: Partial<ModelMetrics>;
  interpretation: string;
}

export interface ConfusionMatrixResponse {
  model: string | null;
  split: string;
  threshold: number;
  true_negatives: number;
  false_positives: number;
  false_negatives: number;
  true_positives: number;
  matrix: number[][];
  labels: string[];
  note: string;
}

export interface FeatureImportance {
  model: string;
  importance_type: string;
  note: string;
  features: { feature: string; importance: number }[];
}

export interface CurvePoint {
  fpr?: number[];
  tpr?: number[];
  recall?: number[];
  precision?: number[];
  auc?: number;
  ap?: number;
}

export interface CurvesResponse {
  split: string;
  selected_model: string | null;
  curves: Record<string, CurvePoint>;
}

export interface SampleTransaction {
  true_label: number;
  features: Record<string, number>;
}

export interface SamplesResponse {
  note: string;
  data_source: string;
  samples: SampleTransaction[];
}

export interface ModelInfo {
  model_version: string;
  model_name: string;
  trained_at_utc: string;
  data_source: string;
  is_demo: boolean;
  decision_threshold: number;
  threshold_tuning: {
    method: string;
    tuned_on: string;
    validation_metrics_at_tuned_threshold: ModelMetrics;
  } | null;
  risk_categories: Record<string, string> | null;
  preprocessing: Record<string, string> | null;
  split_sizes: Record<string, number> | null;
  random_seed: number | null;
  selection_metric: string | null;
  selection_reason: string | null;
  feature_schema: { features: string[]; notes: string[] };
  probability_note: string;
}
