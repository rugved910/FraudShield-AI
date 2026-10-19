"""FraudShield AI — reproducible training pipeline.

Usage
-----
    python scripts/train.py              # train on the real Kaggle dataset
    python scripts/train.py --demo       # train on clearly-labelled synthetic data

What this script does
---------------------
1. Loads and validates the dataset (fails loudly if it is missing/corrupt).
2. Performs exploratory data analysis and saves figures to reports/figures/.
3. Splits the data **before** any fitting: 60 % train / 20 % validation / 20 % test
   (stratified, so the fraud ratio is preserved in every split).
4. Trains three imbalance-aware classifiers:
      - Logistic Regression (class_weight="balanced")
      - Random Forest        (class_weight="balanced")
      - Balanced Random Forest (imblearn — bootstraps a balanced subset per tree)
5. Compares models on the VALIDATION set (PR-AUC as the primary metric) and
   selects the winner.  The classification threshold is then tuned on the
   validation set only (maximising F1, ties broken toward higher recall).
6. Evaluates the selected model + threshold exactly ONCE on the held-out test set.
7. Saves the fitted pipeline, metrics, curves and plots under artifacts/.

Leakage prevention
------------------
* Duplicate rows are removed before splitting so one transaction cannot land
  in two splits.
* The StandardScaler for Time/Amount is fitted inside the sklearn pipeline on
  the training fold only.
* Oversampling (Balanced Random Forest) happens inside the training loop only;
  validation and test data are never resampled.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from imblearn.ensemble import BalancedRandomForestClassifier

from backend.app.core.config import (
    AMOUNT_COLUMN,
    ARTIFACTS_DIR,
    DEMO_DATA_SOURCE_LABEL,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    PROCESSED_DIR,
    RISK_LOW_MAX_RATIO,
    TARGET_COLUMN,
    TIME_COLUMN,
    V_FEATURES,
)
from backend.app.core.data_validation import (
    DatasetError,
    generate_demo_dataframe,
    load_dataset,
)

RANDOM_SEED = 42
MODEL_VERSION = "1.0.0"
N_ESTIMATORS = 200

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
sns.set_theme(style="darkgrid", palette="deep")
FIGSIZE = (8, 5)
DPI = 130


def _savefig(name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=DPI)
    plt.close()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


# ---------------------------------------------------------------------------
# Phase 1 — exploratory data analysis
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame, report: dict, data_source: str) -> dict:
    """Create analysis figures and return dataset statistics for the API."""
    n_total = len(df)
    n_fraud = int((df[TARGET_COLUMN] == 1).sum())
    n_legit = n_total - n_fraud

    # 1. Class imbalance
    plt.figure(figsize=FIGSIZE)
    ax = sns.countplot(x=TARGET_COLUMN, data=df, hue=TARGET_COLUMN, legend=False,
                       palette={0: "#38bdf8", 1: "#f87171"})
    ax.set_title("Class Distribution — Legitimate vs Fraud")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Legitimate (0)", "Fraud (1)"])
    for p, count in zip(ax.patches, [n_legit, n_fraud]):
        ax.annotate(f"{count:,}\n({count / n_total * 100:.3f}%)",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    _savefig("01_class_distribution.png")

    # 2. Transaction amount distribution (log scale)
    plt.figure(figsize=FIGSIZE)
    sns.histplot(df[AMOUNT_COLUMN], bins=80, log_scale=True, color="#38bdf8")
    plt.title("Transaction Amount Distribution (log scale)")
    _savefig("02_amount_distribution.png")

    # 3. Amount by class
    plt.figure(figsize=FIGSIZE)
    sample = df.sample(min(40000, n_total), random_state=RANDOM_SEED)
    sns.boxplot(x=TARGET_COLUMN, y=AMOUNT_COLUMN, data=sample, showfliers=False,
                hue=TARGET_COLUMN, legend=False,
                palette={0: "#38bdf8", 1: "#f87171"})
    plt.ylim(0, sample[AMOUNT_COLUMN].quantile(0.99))
    plt.title("Transaction Amount by Class (outliers hidden)")
    _savefig("03_amount_by_class.png")

    # 4. Fraud over time of day (Time = seconds since first transaction)
    plt.figure(figsize=FIGSIZE)
    hours = (df.loc[df[TARGET_COLUMN] == 1, TIME_COLUMN] / 3600) % 24
    sns.histplot(hours, bins=24, color="#f87171")
    plt.title("Fraud Cases by Hour of Day (derived from Time)")
    plt.xlabel("Hour")
    _savefig("04_fraud_by_hour.png")

    # 5. Correlation heatmap of anonymised features vs target
    plt.figure(figsize=(11, 9))
    corr = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                xticklabels=False, yticklabels=False)
    plt.title("Feature Correlation Heatmap (V1–V28, Time, Amount vs Class)")
    _savefig("05_correlation_heatmap.png")

    stats = {
        "data_source": data_source,
        "total_transactions": n_total,
        "fraud_cases": n_fraud,
        "legitimate_cases": n_legit,
        "fraud_percentage": round(n_fraud / n_total * 100, 4),
        "duplicate_rows_removed": report.get("duplicate_rows", 0),
        "missing_values": report.get("missing_values", 0),
        "amount_stats": {
            "mean": round(float(df[AMOUNT_COLUMN].mean()), 2),
            "median": round(float(df[AMOUNT_COLUMN].median()), 2),
            "max": round(float(df[AMOUNT_COLUMN].max()), 2),
            "fraud_mean": round(float(df.loc[df[TARGET_COLUMN] == 1, AMOUNT_COLUMN].mean()), 2),
            "legit_mean": round(float(df.loc[df[TARGET_COLUMN] == 0, AMOUNT_COLUMN].mean()), 2),
        },
        "time_span_hours": round(float(df[TIME_COLUMN].max() / 3600), 1),
    }
    dump_json(ARTIFACTS_DIR / "dataset_stats.json", stats)
    return stats


# ---------------------------------------------------------------------------
# Phase 2 — splitting & preprocessing
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Scale Time and Amount; pass the already-PCA-scaled V features through.

    V1..V28 are principal components produced by the dataset authors, so they
    are already centred/scaled and must not be re-scaled.
    """
    return ColumnTransformer(
        transformers=[
            ("time_amount_scaler", StandardScaler(), [TIME_COLUMN, AMOUNT_COLUMN]),
            ("passthrough", "passthrough", V_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipelines() -> dict[str, Pipeline]:
    """Return the three candidate model pipelines."""
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=N_ESTIMATORS,
                        class_weight="balanced",
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "balanced_random_forest": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    BalancedRandomForestClassifier(
                        n_estimators=N_ESTIMATORS,
                        sampling_strategy="all",
                        replacement=False,
                        bootstrap=False,
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }


# ---------------------------------------------------------------------------
# Phase 3 — evaluation helpers
# ---------------------------------------------------------------------------
def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                           threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def curve_points(fpr, tpr, downsample: int = 200) -> dict:
    idx = np.unique(np.linspace(0, len(fpr) - 1, min(downsample, len(fpr))).astype(int))
    return {"fpr": fpr[idx].round(5).tolist(), "tpr": tpr[idx].round(5).tolist()}


def pr_curve_points(recall, precision, downsample: int = 200) -> dict:
    idx = np.unique(np.linspace(0, len(recall) - 1, min(downsample, len(recall))).astype(int))
    return {
        "recall": recall[idx].round(5).tolist(),
        "precision": precision[idx].round(5).tolist(),
    }


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, list]:
    """Pick the probability threshold that maximises F1 on the validation set.

    Ties are broken toward the *lower* threshold (higher recall), because a
    missed fraud (false negative) is costlier than a false alarm.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns len(thr) = len(prec) - 1
    f1s = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    best = int(np.argmax(f1s))
    best_thr = float(np.clip(thr[best], 1e-4, 1 - 1e-4))
    # tie-break: among thresholds with near-identical F1 (within 1e-6), prefer
    # the one with the highest recall.
    candidates = np.flatnonzero(f1s >= f1s[best] - 1e-6)
    best_recall_idx = candidates[int(np.argmax(rec[candidates]))]
    best_thr = float(np.clip(thr[best_recall_idx], 1e-4, 1 - 1e-4))

    idx = np.unique(np.linspace(0, len(thr) - 1, min(150, len(thr))).astype(int))
    sweep = [
        {"threshold": round(float(thr[i]), 5),
         "precision": round(float(prec[i]), 5),
         "recall": round(float(rec[i]), 5),
         "f1": round(float(f1s[i]), 5)}
        for i in idx
    ]
    return best_thr, sweep


# ---------------------------------------------------------------------------
# Phase 4 — plots
# ---------------------------------------------------------------------------
def plot_curves(all_val_curves: dict, selected: str, test_curves: dict | None) -> None:
    colours = {"logistic_regression": "#38bdf8",
               "random_forest": "#fbbf24",
               "balanced_random_forest": "#f87171"}

    plt.figure(figsize=FIGSIZE)
    for name, c in all_val_curves.items():
        plt.plot(c["pr"]["recall"], c["pr"]["precision"],
                 label=f"{name} (AP={c['ap']:.3f})", color=colours.get(name))
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="no-skill")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves (validation set)")
    plt.legend(fontsize=8)
    _savefig("06_precision_recall_curves.png")

    plt.figure(figsize=FIGSIZE)
    for name, c in all_val_curves.items():
        plt.plot(c["roc"]["fpr"], c["roc"]["tpr"],
                 label=f"{name} (AUC={c['auc']:.3f})", color=colours.get(name))
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (validation set)")
    plt.legend(fontsize=8)
    _savefig("07_roc_curves.png")

    if test_curves is not None:
        plt.figure(figsize=FIGSIZE)
        plt.plot(test_curves["pr"]["recall"], test_curves["pr"]["precision"],
                 color=colours.get(selected, "#38bdf8"))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision–Recall Curve — {selected} (test set)")
        _savefig("08_test_pr_curve.png")

        plt.figure(figsize=FIGSIZE)
        plt.plot(test_curves["roc"]["fpr"], test_curves["roc"]["tpr"],
                 color=colours.get(selected, "#38bdf8"))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — {selected} (test set)")
        _savefig("09_test_roc_curve.png")


def plot_threshold_sweep(sweep: list, chosen: float) -> None:
    df = pd.DataFrame(sweep)
    plt.figure(figsize=FIGSIZE)
    plt.plot(df["threshold"], df["precision"], label="Precision")
    plt.plot(df["threshold"], df["recall"], label="Recall")
    plt.plot(df["threshold"], df["f1"], label="F1")
    plt.axvline(chosen, color="red", ls="--", lw=1, label=f"chosen threshold={chosen:.3f}")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.title("Threshold Sweep on Validation Set (selected model)")
    plt.legend(fontsize=8)
    _savefig("10_threshold_sweep.png")


def plot_confusion(cm: dict, title: str, filename: str) -> None:
    matrix = np.array([[cm["true_negatives"], cm["false_positives"]],
                       [cm["false_negatives"], cm["true_positives"]]])
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Fraud"],
                yticklabels=["Legitimate", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    _savefig(filename)


def plot_feature_importance(fi: dict, model_name: str) -> None:
    top = fi["features"][:15][::-1]
    plt.figure(figsize=FIGSIZE)
    sns.barplot(x=[f["importance"] for f in top],
                y=[f["feature"] for f in top], color="#38bdf8")
    plt.title(f"Feature Importance — {model_name}\n({fi['importance_type']})")
    plt.xlabel("Importance")
    _savefig("12_feature_importance.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Train FraudShield AI models.")
    parser.add_argument("--demo", action="store_true",
                        help="Train on clearly-labelled SYNTHETIC demo data "
                             "(only when the real dataset is unavailable).")
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load -----------------------------------------------------------
    if args.demo:
        print("[DEMO MODE] Generating synthetic data — results are NOT real.")
        df, report = generate_demo_dataframe()
        data_source = DEMO_DATA_SOURCE_LABEL
    else:
        try:
            df, report = load_dataset()
        except DatasetError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            print("\nTip: run 'python scripts/train.py --demo' for a synthetic demo.",
                  file=sys.stderr)
            return 1
        data_source = report["data_source"]

    print(f"Loaded {report['rows_after_dedup']:,} rows "
          f"({report['fraud_cases']:,} fraud, {report['fraud_percentage']:.3f}%) "
          f"from {data_source}")
    if report.get("duplicate_rows"):
        print(f"Removed {report['duplicate_rows']:,} duplicate rows before splitting.")

    # ---- 2. EDA ------------------------------------------------------------
    run_eda(df, report, data_source)
    print("EDA figures written to reports/figures/")

    # ---- 3. Stratified splits (BEFORE any fitting) ------------------------
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval,
        random_state=RANDOM_SEED,
    )
    print(f"Split sizes — train: {len(X_train):,}  validation: {len(X_val):,}  "
          f"test: {len(X_test):,} (fraud ratio preserved via stratification)")

    # Persist the split indices for reproducibility / leakage tests.
    dump_json(PROCESSED_DIR / "split_manifest.json", {
        "seed": RANDOM_SEED,
        "test_size": 0.20,
        "validation_fraction_of_trainval": 0.25,
        "n_train": int(len(X_train)),
        "n_validation": int(len(X_val)),
        "n_test": int(len(X_test)),
        "fraud_rate_train": float(y_train.mean()),
        "fraud_rate_validation": float(y_val.mean()),
        "fraud_rate_test": float(y_test.mean()),
    })

    # ---- 4. Train & validate ----------------------------------------------
    pipelines = build_pipelines()
    comparison: dict[str, Any] = {}
    val_curves: dict[str, Any] = {}
    t0 = time.time()

    for name, pipe in pipelines.items():
        print(f"Training {name} ...", flush=True)
        start = time.time()
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_val)[:, 1]
        metrics = classification_metrics(y_val.to_numpy(), y_prob, threshold=0.5)
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        rec, prec, _ = precision_recall_curve(y_val, y_prob)
        comparison[name] = {"validation": metrics, "train_seconds": round(time.time() - start, 1)}
        val_curves[name] = {
            "roc": curve_points(fpr, tpr),
            "pr": pr_curve_points(rec, prec),
            "auc": metrics["roc_auc"],
            "ap": metrics["pr_auc"],
        }
        print(f"  -> val PR-AUC {metrics['pr_auc']:.4f}  ROC-AUC {metrics['roc_auc']:.4f}  "
              f"F1 {metrics['f1']:.4f}  ({comparison[name]['train_seconds']}s)")

    # ---- 5. Model selection (validation PR-AUC) ---------------------------
    # PR-AUC is the primary metric because accuracy is meaningless at a
    # 0.17 % fraud rate, and PR-AUC directly reflects performance on the
    # rare class.
    selected = max(comparison, key=lambda n: comparison[n]["validation"]["pr_auc"])
    print(f"\nSelected model by validation PR-AUC: {selected}")

    # ---- 6. Threshold tuning on validation ONLY ---------------------------
    best_pipe = pipelines[selected]
    y_prob_val = best_pipe.predict_proba(X_val)[:, 1]
    best_threshold, sweep = tune_threshold(y_val.to_numpy(), y_prob_val)
    tuned_val_metrics = classification_metrics(y_val.to_numpy(), y_prob_val, best_threshold)
    comparison[selected]["validation_tuned"] = tuned_val_metrics
    print(f"Tuned threshold: {best_threshold:.4f} -> val F1 {tuned_val_metrics['f1']:.4f}, "
          f"recall {tuned_val_metrics['recall']:.4f}, precision {tuned_val_metrics['precision']:.4f}")

    # ---- 7. ONE test-set evaluation ---------------------------------------
    y_prob_test = best_pipe.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test.to_numpy(), y_prob_test, best_threshold)
    test_metrics_at_05 = classification_metrics(y_test.to_numpy(), y_prob_test, 0.5)
    fpr_t, tpr_t, _ = roc_curve(y_test, y_prob_test)
    rec_t, prec_t, _ = precision_recall_curve(y_test, y_prob_test)
    test_curves = {"roc": curve_points(fpr_t, tpr_t), "pr": pr_curve_points(rec_t, prec_t)}
    print(f"TEST ({selected} @ {best_threshold:.4f}): "
          f"precision {test_metrics['precision']:.4f}  recall {test_metrics['recall']:.4f}  "
          f"F1 {test_metrics['f1']:.4f}  PR-AUC {test_metrics['pr_auc']:.4f}  "
          f"ROC-AUC {test_metrics['roc_auc']:.4f}")

    # Risk-category distribution of the test set at the tuned threshold
    # (real counts — the dashboard shows these instead of invented numbers).
    risk_dist = {"low": 0, "medium": 0, "high": 0}
    for p in y_prob_test:
        if p >= best_threshold:
            risk_dist["high"] += 1
        elif p >= best_threshold * RISK_LOW_MAX_RATIO:
            risk_dist["medium"] += 1
        else:
            risk_dist["low"] += 1

    # ---- 8. Feature importance --------------------------------------------
    model_obj = best_pipe.named_steps["model"]
    if hasattr(model_obj, "feature_importances_"):
        importance_type = "mean_decrease_in_impurity"
        importances = model_obj.feature_importances_
    else:
        importance_type = "standardized_log_odds_coefficients"
        importances = np.abs(model_obj.coef_[0])
    order = np.argsort(importances)[::-1]
    if importance_type == "mean_decrease_in_impurity":
        importance_note = (
            "MDI = how much the forest relies on a feature across trees "
            "(mean decrease in impurity). V1..V28 are anonymised PCA components — "
            "they cannot be interpreted as real-world attributes such as merchant "
            "or location. Importance shows statistical reliance, not causation."
        )
    else:
        importance_note = (
            "Values are absolute standardized logistic-regression coefficients "
            "(log-odds contribution per 1 standard deviation of the feature). "
            "V1..V28 are anonymised PCA components — they cannot be interpreted as "
            "real-world attributes. Coefficients show association, not causation."
        )
    feature_importance = {
        "model": selected,
        "importance_type": importance_type,
        "note": importance_note,
        "features": [
            {"feature": FEATURE_COLUMNS[i], "importance": float(importances[i])}
            for i in order
        ],
    }

    # ---- 9. Plots ----------------------------------------------------------
    plot_curves(val_curves, selected, test_curves)
    plot_threshold_sweep(sweep, best_threshold)
    plot_confusion(test_metrics, f"Confusion Matrix — {selected} (test set, threshold={best_threshold:.3f})",
                   "11_confusion_matrix_test.png")
    plot_feature_importance(feature_importance, selected)
    print("Evaluation figures written to reports/figures/")

    # ---- 10. Persist artifacts --------------------------------------------
    trained_at = datetime.now(timezone.utc).isoformat()
    joblib.dump(best_pipe, ARTIFACTS_DIR / "model_pipeline.joblib")
    dump_json(ARTIFACTS_DIR / "model_metadata.json", {
        "model_version": MODEL_VERSION,
        "model_name": selected,
        "trained_at_utc": trained_at,
        "data_source": data_source,
        "is_demo": data_source == DEMO_DATA_SOURCE_LABEL,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "decision_threshold": best_threshold,
        "threshold_tuning": {
            "method": "maximise F1 on validation set (ties broken toward higher recall)",
            "tuned_on": "validation",
            "validation_metrics_at_tuned_threshold": tuned_val_metrics,
        },
        "preprocessing": {
            "time_amount": "StandardScaler fitted on training data only",
            "v_features": "V1..V28 are pre-scaled PCA components; passed through unchanged",
        },
        "risk_categories": {
            "low": f"probability < {RISK_LOW_MAX_RATIO} * threshold",
            "medium": f"{RISK_LOW_MAX_RATIO} * threshold <= probability < threshold",
            "high": "probability >= threshold",
        },
        "split_sizes": {"train": int(len(X_train)), "validation": int(len(X_val)),
                        "test": int(len(X_test))},
        "random_seed": RANDOM_SEED,
    })
    dump_json(ARTIFACTS_DIR / "model_comparison.json", {
        "selection_metric": "validation_pr_auc",
        "selection_reason": (
            "Accuracy is misleading with ~0.17 % fraud. PR-AUC measures quality "
            "of the rare-class ranking and is the primary selection criterion; "
            "recall is prioritised over precision at the tie-break stage."
        ),
        "selected_model": selected,
        "decision_threshold": best_threshold,
        "models": comparison,
        "test_metrics_selected_model": test_metrics,
        "test_metrics_selected_model_at_threshold_0_5": test_metrics_at_05,
        "test_risk_distribution": risk_dist,
        "total_training_seconds": round(time.time() - t0, 1),
    })
    dump_json(ARTIFACTS_DIR / "evaluation_metrics.json", {
        "model_version": MODEL_VERSION,
        "data_source": data_source,
        "selected_model": selected,
        "decision_threshold": best_threshold,
        "validation_curves": val_curves,
        "test_curves": test_curves,
        "threshold_sweep": sweep,
        "note": (
            "All model comparison metrics were measured on the validation split. "
            "Test-set metrics exist only for the selected model at the tuned "
            "threshold (plus a 0.5 reference row); the test set was never used "
            "for model selection, hyper-parameter tuning or threshold tuning."
        ),
    })
    dump_json(ARTIFACTS_DIR / "feature_importance.json", feature_importance)

    # A handful of REAL held-out test rows for the demo loader in the UI.
    rng = np.random.default_rng(RANDOM_SEED)
    fraud_idx = X_test.index[y_test == 1]
    legit_idx = X_test.index[y_test == 0]
    picks = list(rng.choice(fraud_idx, size=min(4, len(fraud_idx)), replace=False)) + \
            list(rng.choice(legit_idx, size=min(4, len(legit_idx)), replace=False))
    samples = []
    for i in picks:
        row = X_test.loc[i]
        samples.append({
            "true_label": int(y_test.loc[i]),
            "features": {c: float(row[c]) for c in FEATURE_COLUMNS},
        })
    dump_json(ARTIFACTS_DIR / "sample_transactions.json", {
        "note": (
            "REAL rows drawn from the held-out test split of the "
            f"{data_source} dataset (never seen during training). "
            "true_label is the dataset's Class column and is shown for "
            "educational comparison only."
        ),
        "data_source": data_source,
        "samples": samples,
    })

    print(f"\nAll artifacts saved to {ARTIFACTS_DIR}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
