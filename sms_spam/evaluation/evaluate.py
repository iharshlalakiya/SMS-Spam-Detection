"""
sms_spam/evaluation/evaluate.py — Evaluation Pipeline Step
===========================================================
Contains the single evaluation step used by main.py:

    step_evaluate  →  predict → metrics → plots → save artefacts

Imported and called by main.py.
"""

import sys
import yaml
import pickle
import json
import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no popup windows

import matplotlib.pyplot as plt
from pathlib import Path

from sms_spam.models.svm          import SpamDetector
from sms_spam.features.extraction import TFIDFExtractor
from sms_spam.evaluation.metrics  import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from sms_spam.mlflow import MlflowTracker
from sms_spam.logs.logger import get_logger

log = get_logger(__name__)


def step_evaluate(
    detector: SpamDetector,
    tfidf: TFIDFExtractor,
    X_test_tfidf,
    y_test,
    models_dir: Path,
    results_dir: Path,
) -> dict:
    """
    Evaluate the trained SVM, save plots/reports, and persist artefacts.

    Parameters
    ----------
    detector : SpamDetector
        Trained SVM classifier.
    tfidf : TFIDFExtractor
        Fitted TF-IDF vectorizer.
    X_test_tfidf : sparse matrix
        Test features.
    y_test : pd.Series
        Ground-truth test labels.
    models_dir : Path
        Directory to save svm.pkl and tfidf_vectorizer.pkl.
    results_dir : Path
        Directory to save confusion matrix, ROC curve, and metrics JSON.

    Returns
    -------
    dict
        Metrics dictionary (accuracy, precision, recall, f1_score, roc_auc).
    """
    # ── Predict ──────────────────────────────────────────────────────────────────
    log.info("Running predictions on test set (%d samples)", len(y_test))
    y_pred  = detector.predict(X_test_tfidf)
    y_proba = detector.predict_proba(X_test_tfidf)

    # ── Metrics ──────────────────────────────────────────────────────────────────
    metrics = calculate_metrics(y_test.values, y_pred, y_proba[:, 1])
    log.info("Metrics — Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f  AUC=%.4f",
             metrics.get('accuracy', 0), metrics.get('precision', 0),
             metrics.get('recall', 0),   metrics.get('f1_score', 0),
             metrics.get('roc_auc', 0))

    # ── Plots ───────────────────────────────────────────────────────────────────
    cm_path  = results_dir / "confusion_matrices" / "svm_cm.png"
    roc_path = results_dir / "plots"              / "svm_roc.png"
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    roc_path.parent.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(y_test.values, y_pred,        "SVM", str(cm_path))
    log.info("Confusion matrix saved → %s", cm_path)
    plot_roc_curve(y_test.values, y_proba[:, 1], "SVM", str(roc_path))
    log.info("ROC curve saved → %s", roc_path)

    # ── Save JSON report ─────────────────────────────────────────────────────────
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / "svm_results.json"
    with open(json_path, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    log.info("Metrics JSON saved → %s", json_path)

    # ── Save model artefacts ──────────────────────────────────────────────────────
    models_dir.mkdir(parents=True, exist_ok=True)
    detector.save(str(models_dir / "svm.pkl"))
    joblib.dump(tfidf, str(models_dir / "tfidf_vectorizer.pkl"))
    log.info("Model artefacts saved → %s", models_dir)

    return metrics, cm_path, roc_path, json_path


def run_dvc_stage(features_path: Path, models_dir: Path, results_dir: Path, params: dict = None) -> None:
    params = params or {}
    # ── Load features ─────────────────────────────────────────────────────────
    print(f"Loading features from {features_path}")
    with open(features_path, "rb") as f:
        data = pickle.load(f)

    X_test_tfidf = data["X_test_tfidf"]
    y_test       = data["y_test"]

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = models_dir / "svm.pkl"
    print(f"Loading model from {model_path}")
    detector = SpamDetector.from_file(str(model_path))

    # ── Predict ───────────────────────────────────────────────────────────────
    print(f"Predicting on {len(y_test)} test samples")
    y_pred  = detector.predict(X_test_tfidf)
    y_proba = detector.predict_proba(X_test_tfidf)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = calculate_metrics(y_test.values, y_pred, y_proba[:, 1])
    for k, v in metrics.items():
        print(f"{k:<15} {v:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    cm_path  = results_dir / "confusion_matrices" / "svm_cm.png"
    roc_path = results_dir / "plots" / "svm_roc.png"
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    roc_path.parent.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(y_test.values, y_pred,        "SVM", str(cm_path))
    plot_roc_curve(y_test.values, y_proba[:, 1], "SVM", str(roc_path))
    print(f"Plots saved -> {cm_path} | {roc_path}")

    # ── JSON metrics (DVC metrics file) ───────────────────────────────────────
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / "svm_results.json"
    with open(json_path, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print(f"Metrics JSON saved -> {json_path}")

    # ── MLflow tracking ───────────────────────────────────────────────────────
    mlflow_cfg = params.get("mlflow", {})
    tracker = MlflowTracker(
        experiment_name = mlflow_cfg.get("experiment_name", "SMS-Spam-Detection"),
        tracking_uri    = mlflow_cfg.get("tracking_uri", "mlruns"),
        model_type      = "SVM",
    )
    with tracker:
        tracker.log_params({k: v for k, v in params.items() if k != "mlflow"})
        tracker.log_metrics(metrics)
        tracker.log_artifacts(cm_path, roc_path, json_path)
        tracker.log_model(
            detector   = detector,
            tfidf      = joblib.load(str(models_dir / "tfidf_vectorizer.pkl")),
            models_dir = models_dir,
            register   = mlflow_cfg.get("register_model", False),
            model_name = mlflow_cfg.get("model_name", "SmsSpamDetector"),
        )

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'Metric':<15}  {'Value':>10}")
    print("-" * 28)
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        if key in metrics:
            label = key.replace("_", " ").title()
            print(f"{label:<15}  {metrics[key]:>10.4f}  ({metrics[key]*100:.2f}%)")


if __name__ == "__main__":
    # DVC Execution
    ROOT = Path(__file__).resolve().parents[2]
    with open(ROOT / "params.yaml") as f:
        params = yaml.safe_load(f)

    run_dvc_stage(
        features_path = ROOT / params["data"]["features_path"],
        models_dir    = ROOT / params["paths"]["models_dir"],
        results_dir   = ROOT / params["paths"]["results_dir"],
        params        = params,
    )
    print("\n[stage_evaluate] DONE  Evaluation complete")
