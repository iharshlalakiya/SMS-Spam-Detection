#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SMS Spam Detection — Main Pipeline
====================================
Single entry point that runs the complete pipeline:

    Step 1 │ Check / download dataset
    Step 2 │ Load & preprocess data
    Step 3 │ Split into train / test sets
    Step 4 │ Extract TF-IDF features
    Step 5 │ Train SVM classifier
    Step 6 │ Evaluate & save results

Usage
-----
    python main.py
    python main.py --skip-download   # if spam.csv already exists
    python main.py --data-path data/raw/spam.csv
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Fix Windows console encoding ──────────────────────────────────────────────
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

# ── Project root on sys.path so `sms_spam` is importable ──────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib

from sms_spam.logs.logger import get_logger
log = get_logger("sms_spam.main")

# ── Training pipeline steps (sms_spam/train.py) ───────────────────────────────
from sms_spam.train import (
    step_preprocess,
    step_split,
    step_features,
    step_train,
)

# ── Evaluation pipeline step (sms_spam/evaluation/evaluate.py) ─────────────
from sms_spam.evaluation.evaluate import step_evaluate


# ── Evaluation helpers (sms_spam/evaluation/metrics.py) ──────────────────────
from sms_spam.evaluation.metrics import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from sms_spam.models.svm import SpamDetector
from sms_spam.features.extraction import TFIDFExtractor


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def banner(step, total, title):
    bar  = "═" * 68
    print(f"\n╔{bar}╗")
    print(f"║  Step {step}/{total}  │  {title:<56}║")
    print(f"╚{bar}╝")


def ok(msg):   print(f"   ✅  {msg}")
def err(msg):  print(f"   ❌  {msg}")
def info(msg): print(f"   ℹ️   {msg}")


# File → function call tracer
_FILE_MAP = {
    # ── sms_spam/train.py ────────────────────────────────────────────────
    "step_preprocess"      : "sms_spam/train.py",
    "step_split"           : "sms_spam/train.py",
    "step_features"        : "sms_spam/train.py",
    "step_train"           : "sms_spam/train.py",
    # ── sms_spam/evaluation/evaluate.py ────────────────────────────────
    "step_evaluate"        : "sms_spam/evaluation/evaluate.py",
    # ── internals called inside sms_spam/train.py ─────────────────────────
    "load_data"            : "sms_spam/data/preprocessing.py",
    "preprocess_pipeline"  : "sms_spam/data/preprocessing.py",
    "train_test_split"     : "sklearn/model_selection",
    "TFIDFExtractor"       : "sms_spam/features/extraction.py",
    "fit_transform"        : "sms_spam/features/extraction.py  →  TFIDFExtractor",
    "transform"            : "sms_spam/features/extraction.py  →  TFIDFExtractor",
    "SpamDetector"         : "sms_spam/models/svm.py",
    "detector.train"       : "sms_spam/models/svm.py           →  SpamDetector",
    # ── internals called inside sms_spam/evaluation/evaluate.py ──────────
    "detector.predict"     : "sms_spam/models/svm.py           →  SpamDetector",
    "detector.predict_proba": "sms_spam/models/svm.py          →  SpamDetector",
    "detector.save"        : "sms_spam/models/svm.py           →  SpamDetector",
    "get_training_diagnostics": "sms_spam/models/svm.py        →  SpamDetector",
    "calculate_metrics"    : "sms_spam/evaluation/metrics.py",
    "plot_confusion_matrix": "sms_spam/evaluation/metrics.py",
    "plot_roc_curve"       : "sms_spam/evaluation/metrics.py",
    "joblib.dump"          : "joblib  (third-party)",
}


def trace(fn_label: str):
    """Print the file and function name being called."""
    file = _FILE_MAP.get(fn_label, "<unknown>")
    print(f"   📄 {file:<50}  →  {fn_label}()")


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline steps
# ══════════════════════════════════════════════════════════════════════════════

def step_download(data_path: Path, skip: bool) -> bool:
    """Ensure the dataset exists; download it if needed."""
    if data_path.exists():
        ok(f"Dataset already present: {data_path}")
        return True

    if skip:
        err(f"Dataset not found at {data_path} and --skip-download was set.")
        return False

    info("Dataset not found. Attempting download...")

    # Try GitHub mirror (no credentials needed)
    try:
        import urllib.request
        url = (
            "https://raw.githubusercontent.com/"
            "mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
        )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"   ⬇️  Downloading from mirror...")
        urllib.request.urlretrieve(url, data_path)
        if data_path.exists() and data_path.stat().st_size > 0:
            ok(f"Downloaded to {data_path}")
            return True
    except Exception as exc:
        info(f"Mirror failed: {exc}")

    # Fall back to Kaggle API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        info("Kaggle API authenticated. Downloading dataset...")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(
            "uciml/sms-spam-collection-dataset",
            path=str(data_path.parent),
            unzip=True,
        )
        if data_path.exists():
            ok("Downloaded via Kaggle API")
            return True
    except Exception as exc:
        info(f"Kaggle download failed: {exc}")

    err("Could not download the dataset automatically.")
    print("\n   Please download spam.csv manually from:")
    print("   https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")
    print(f"\n   Place it at: {data_path}\n")
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Run the full SMS spam detection pipeline.")
    p.add_argument("--data-path",      default=str(ROOT / "data" / "raw" / "spam.csv"),
                   help="Path to spam.csv (default: data/raw/spam.csv)")
    p.add_argument("--models-dir",     default=str(ROOT / "models"),
                   help="Directory to save trained artefacts")
    p.add_argument("--results-dir",    default=str(ROOT / "results"),
                   help="Directory to save evaluation outputs")
    p.add_argument("--skip-download",  action="store_true",
                   help="Skip dataset download step")
    return p.parse_args()


def main():
    args = parse_args()
    data_path   = Path(args.data_path)
    models_dir  = Path(args.models_dir)
    results_dir = Path(args.results_dir)

    total_steps = 6
    pipeline_start = time.time()

    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║          SMS SPAM DETECTION — Full Pipeline                         ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # ── Step 1: Download ──────────────────────────────────────────────────────
    banner(1, total_steps, "Dataset")
    log.info("Step 1 — Checking dataset at %s", data_path)
    if not step_download(data_path, args.skip_download):
        log.error("Dataset not available. Aborting.")
        sys.exit(1)

    # ── Step 2: Preprocess  [sms_spam/train.py → step_preprocess()] ──────────
    banner(2, total_steps, "Load & Preprocess")
    trace("step_preprocess")
    log.info("Step 2 — Loading and preprocessing data")
    try:
        df = step_preprocess(data_path)
        ok(f"Loaded {len(df)} messages  │  spam={df['label'].sum()}  ham={(df['label']==0).sum()}")
        log.info("Loaded %d messages  (spam=%d, ham=%d)", len(df), df['label'].sum(), (df['label']==0).sum())
    except Exception as exc:
        err(f"Preprocessing failed: {exc}")
        log.exception("Preprocessing failed")
        import traceback; traceback.print_exc(); sys.exit(1)

    # ── Step 3: Split  [sms_spam/train.py → step_split()] ────────────────────
    banner(3, total_steps, "Train / Test Split  (80 / 20)")
    trace("step_split")
    log.info("Step 3 — Splitting dataset (80/20 stratified)")
    X_train, X_test, y_train, y_test = step_split(df)
    ok(f"Train: {len(X_train)}  │  Test: {len(X_test)}")
    log.info("Split complete — train=%d  test=%d", len(X_train), len(X_test))

    # ── Step 4: Features  [sms_spam/train.py → step_features()] ──────────────
    banner(4, total_steps, "TF-IDF Feature Extraction")
    trace("step_features")
    log.info("Step 4 — Extracting TF-IDF features")
    try:
        tfidf, X_train_tfidf, X_test_tfidf = step_features(X_train, X_test)
        ok(f"Feature dimensions: {X_train_tfidf.shape[1]}")
        log.info("TF-IDF feature dimensions: %d", X_train_tfidf.shape[1])
    except Exception as exc:
        err(f"Feature extraction failed: {exc}")
        log.exception("Feature extraction failed")
        import traceback; traceback.print_exc(); sys.exit(1)

    # ── Step 5: Train  [sms_spam/train.py → step_train()] ────────────────────
    banner(5, total_steps, "Train SVM Classifier")
    trace("step_train")
    log.info("Step 5 — Training SVM classifier")
    try:
        detector = step_train(X_train_tfidf, y_train)
        diag = detector.get_training_diagnostics()
        if diag:
            for k, v in diag.items():
                info(f"{k.replace('_', ' ').title()}: {v}")
                log.debug("Diagnostics — %s: %s", k, v)
        ok(f"Training complete in {detector.training_time:.2f}s")
        log.info("Training complete in %.2fs", detector.training_time)
    except Exception as exc:
        err(f"Training failed: {exc}")
        log.exception("Training failed")
        import traceback; traceback.print_exc(); sys.exit(1)

    # ── Step 6: Evaluate  [sms_spam/evaluation/evaluate.py → step_evaluate()] ──
    banner(6, total_steps, "Evaluate & Save Results")
    trace("step_evaluate")
    log.info("Step 6 — Evaluating model")
    try:
        metrics, cm_path, roc_path, json_path = step_evaluate(
            detector, tfidf, X_test_tfidf, y_test, models_dir, results_dir
        )
        print()
        print(f"   {'Metric':<15} {'Value':>10}")
        print(f"   {'-'*27}")
        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            if key in metrics:
                label = key.replace("_", " ").title()
                print(f"   {label:<15} {metrics[key]:>10.4f}  ({metrics[key]*100:.2f}%)")
                log.info("%-15s %.4f", label, metrics[key])
        print()
        ok(f"Confusion matrix → {cm_path}")
        ok(f"ROC curve        → {roc_path}")
        ok(f"Metrics JSON     → {json_path}")
        ok(f"Model saved      → {models_dir / 'svm.pkl'}")
        ok(f"Vectorizer saved → {models_dir / 'tfidf_vectorizer.pkl'}")
        log.info("Artefacts saved — model: %s | vectorizer: %s | metrics: %s",
                 models_dir / 'svm.pkl', models_dir / 'tfidf_vectorizer.pkl', json_path)
    except Exception as exc:
        err(f"Evaluation failed: {exc}")
        log.exception("Evaluation failed")
        import traceback; traceback.print_exc(); sys.exit(1)

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - pipeline_start
    f1 = metrics.get("f1_score", 0)

    log.info("Pipeline complete in %.1fs — F1=%.4f  Accuracy=%.4f",
             elapsed, f1, metrics['accuracy'])

    print(f"\n╔══════════════════════════════════════════════════════════════════════╗")
    print(f"║  ✅  Pipeline complete in {elapsed:>6.1f}s                              ║")
    print(f"║  🏆  SVM  F1-Score: {f1:.4f}  Accuracy: {metrics['accuracy']:.4f}              ║")
    print(f"╚══════════════════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()
