"""
sms_spam/train/compare_models.py — Multi-Model MLflow Comparison
=================================================================
Trains multiple classifiers on the same data and logs each one
as a separate MLflow run under the same experiment so you can
compare them side-by-side in the MLflow UI.

Models compared
---------------
    SVM                 (baseline — production model)
    Naive Bayes         (fast, probabilistic)
    Logistic Regression (linear, interpretable)
    Random Forest       (ensemble, non-linear)

Usage
-----
    python -m sms_spam.train.compare_models
    python -m sms_spam.train.compare_models --data-path data/raw/spam.csv

Output (MLflow UI)
------------------
    Experiment: SMS-Spam-Detection
    ├── svm-20260420-*          F1=0.9181  AUC=0.9837
    ├── naivebayes-20260420-*   F1=0.9643  AUC=0.9891
    ├── logreg-20260420-*       F1=0.9312  AUC=0.9812
    └── randomforest-20260420-* F1=0.9501  AUC=0.9923
"""

from __future__ import annotations

import sys
import os
import argparse
import time
import yaml
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mlflow
import mlflow.sklearn

from sklearn.svm                import SVC
from sklearn.naive_bayes        import MultinomialNB
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier
from sklearn.metrics            import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from sms_spam.train.train       import step_preprocess, step_split, step_features
from sms_spam.mlflow.mlflow_tracker import _get_tracking_uri, _set_remote_credentials, _get_git_commit
from sms_spam.logs.logger       import get_logger

import platform

log = get_logger("sms_spam.compare_models")


# ── Model catalogue ────────────────────────────────────────────────────────────
# Add / remove models here — each gets its own MLflow run automatically.

def _build_classifiers(params: dict) -> dict:
    svm_p = params.get("svm", {})
    return {
        "SVM": {
            "model": SVC(
                C           = svm_p.get("C", 1.0),
                kernel      = svm_p.get("kernel", "linear"),
                max_iter    = svm_p.get("max_iter", 1000),
                probability = True,
            ),
            "params": {
                "svm.C":        svm_p.get("C", 1.0),
                "svm.kernel":   svm_p.get("kernel", "linear"),
                "svm.max_iter": svm_p.get("max_iter", 1000),
            },
        },
        "NaiveBayes": {
            "model":  MultinomialNB(alpha=0.1),
            "params": {"nb.alpha": 0.1},
        },
        "LogisticRegression": {
            "model": LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
            "params": {"lr.C": 1.0, "lr.max_iter": 1000, "lr.solver": "lbfgs"},
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                n_estimators=100, max_depth=None, random_state=42
            ),
            "params": {"rf.n_estimators": 100, "rf.max_depth": "None", "rf.random_state": 42},
        },
    }


# ── Core comparison logic ──────────────────────────────────────────────────────

def _evaluate(model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred),            6),
        "precision": round(precision_score(y_test, y_pred),           6),
        "recall":    round(recall_score(y_test, y_pred),              6),
        "f1_score":  round(f1_score(y_test, y_pred),                  6),
        "roc_auc":   round(roc_auc_score(y_test, y_proba),            6),
    }


def run_comparison(data_path: Path, params: dict, tracking_uri: str) -> None:
    """
    Train all classifiers and log each as a separate MLflow run.

    All runs are grouped under the same 'SMS-Spam-Detection' experiment
    so you can compare them side-by-side in the MLflow UI.
    """
    experiment_name = params.get("mlflow", {}).get(
        "experiment_name", "SMS-Spam-Detection"
    )

    # Configure tracking
    _set_remote_credentials()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Prepare data once (shared across all classifiers)
    print("\n[*] Preparing shared dataset...")
    df = step_preprocess(data_path)
    X_train, X_test, y_train, y_test = step_split(df)
    tfidf, X_train_tfidf, X_test_tfidf = step_features(X_train, X_test)
    print(f"   Train: {len(X_train)}  |  Test: {len(X_test)}  |  Features: {X_train_tfidf.shape[1]}")

    classifiers = _build_classifiers(params)
    results = {}
    git_sha = _get_git_commit()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    print(f"\n[~] Running {len(classifiers)} models under experiment '{experiment_name}'...\n")

    for name, cfg in classifiers.items():
        model  = cfg["model"]
        hparams = cfg["params"]
        run_name = f"{name.lower()}-{timestamp}"

        print(f"   >>  {name} ...", end="", flush=True)
        t0 = time.time()

        try:
            model.fit(X_train_tfidf, y_train)
            elapsed = time.time() - t0
            metrics = _evaluate(model, X_test_tfidf, y_test)
            results[name] = metrics

            # ── One MLflow run per classifier ──────────────────────────────────
            with mlflow.start_run(
                run_name = run_name,
                tags     = {
                    "project":        "SMS-Spam-Detection",
                    "model_type":     name,
                    "run_type":       "comparison",
                    "git_commit":     git_sha,
                    "python_version": platform.python_version(),
                    "os":             platform.system(),
                },
            ):
                # Common data params
                mlflow.log_params({
                    "data.test_size":     params.get("data", {}).get("test_size", 0.2),
                    "data.random_state":  params.get("data", {}).get("random_state", 42),
                    "tfidf.max_features": params.get("tfidf", {}).get("max_features", 5000),
                })
                # Model-specific params
                mlflow.log_params(hparams)
                # Metrics
                mlflow.log_metrics(metrics)
                mlflow.log_metric("training_time_s", round(elapsed, 4))
                # Log as sklearn model (not registered — comparison runs only)
                mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"  F1={metrics['f1_score']:.4f}  AUC={metrics['roc_auc']:.4f}  ({elapsed:.1f}s)")
            log.info("%s — F1=%.4f  AUC=%.4f  time=%.1fs", name, metrics['f1_score'], metrics['roc_auc'], elapsed)

        except Exception as exc:
            print(f"  FAILED: {exc}")
            log.exception("Model %s failed", name)

    # ── Summary table ──────────────────────────────────────────────────────────
    if results:
        print(f"\n{'─'*70}")
        print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
        print(f"{'─'*70}")
        best_f1    = max(m["f1_score"] for m in results.values())
        for name, m in sorted(results.items(), key=lambda x: x[1]["f1_score"], reverse=True):
            star = " *" if m["f1_score"] == best_f1 else ""
            print(
                f"  {name+star:<22} {m['accuracy']:>9.4f} {m['precision']:>10.4f}"
                f" {m['recall']:>8.4f} {m['f1_score']:>8.4f} {m['roc_auc']:>8.4f}"
            )
        print(f"{'─'*70}")
        print(f"\n[i] View all runs: run `mlflow ui` then open http://127.0.0.1:5000")
        print(f"   Select multiple runs -> click 'Compare' for side-by-side charts\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Compare multiple classifiers with MLflow tracking.")
    p.add_argument("--data-path", default=str(ROOT / "data" / "raw" / "spam.csv"),
                   help="Path to spam.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    params_path = ROOT / "params.yaml"
    with open(params_path) as f:
        params = yaml.safe_load(f)

    tracking_uri = _get_tracking_uri(params.get("mlflow", {}).get("tracking_uri", "mlruns"))

    run_comparison(
        data_path    = Path(args.data_path),
        params       = params,
        tracking_uri = tracking_uri,
    )
