"""
monitoring/dvc_exporter.py
Exports DVC pipeline metrics and stage info to Prometheus on port 9400.
Reads:
  - results/metrics/svm_results.json  (model metrics)
  - dvc.lock                           (pipeline stage hashes / sizes)
  - params.yaml                        (hyperparameters)
"""

import time
import json
import os
import yaml
from pathlib import Path
from prometheus_client import start_http_server, Gauge, Info

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(os.environ.get("PROJECT_DIR", "/app"))
METRICS_FILE = BASE_DIR / "results" / "metrics" / "svm_results.json"
DVC_LOCK     = BASE_DIR / "dvc.lock"
PARAMS_FILE  = BASE_DIR / "params.yaml"
SCRAPE_INTERVAL = int(os.environ.get("SCRAPE_INTERVAL", "30"))

# ── Prometheus metrics — model metrics from DVC ────────────────────────────
dvc_accuracy  = Gauge("dvc_model_accuracy",   "DVC tracked model accuracy")
dvc_precision = Gauge("dvc_model_precision",  "DVC tracked model precision")
dvc_recall    = Gauge("dvc_model_recall",     "DVC tracked model recall")
dvc_f1        = Gauge("dvc_model_f1_score",   "DVC tracked model F1 score")
dvc_roc_auc   = Gauge("dvc_model_roc_auc",    "DVC tracked model ROC AUC")

# ── Prometheus metrics — DVC pipeline stage sizes ──────────────────────────
dvc_stage_model_size      = Gauge("dvc_stage_model_size_bytes",      "SVM model file size in bytes")
dvc_stage_vectorizer_size = Gauge("dvc_stage_vectorizer_size_bytes", "TF-IDF vectorizer file size in bytes")
dvc_stage_features_size   = Gauge("dvc_stage_features_size_bytes",   "Features pkl file size in bytes")
dvc_stage_processed_size  = Gauge("dvc_stage_processed_size_bytes",  "Processed data file size in bytes")

# ── Prometheus metrics — hyperparameters from params.yaml ─────────────────
dvc_param_svm_c            = Gauge("dvc_param_svm_c",             "SVM C hyperparameter")
dvc_param_svm_max_iter     = Gauge("dvc_param_svm_max_iter",      "SVM max_iter hyperparameter")
dvc_param_tfidf_features   = Gauge("dvc_param_tfidf_max_features","TF-IDF max_features hyperparameter")
dvc_param_test_size        = Gauge("dvc_param_test_size",         "Train/test split test_size")

# ── Pipeline info ──────────────────────────────────────────────────────────
dvc_pipeline_info = Info("dvc_pipeline", "DVC pipeline stage information")


def read_metrics():
    if not METRICS_FILE.exists():
        return
    try:
        with open(METRICS_FILE) as f:
            m = json.load(f)
        dvc_accuracy.set(m.get("accuracy",  0))
        dvc_precision.set(m.get("precision", 0))
        dvc_recall.set(m.get("recall",      0))
        dvc_f1.set(m.get("f1_score",        0))
        dvc_roc_auc.set(m.get("roc_auc",    0))
        print(f"[dvc-exporter] metrics — accuracy={m.get('accuracy', 0):.4f}  f1={m.get('f1_score', 0):.4f}")
    except Exception as exc:
        print(f"[dvc-exporter] Error reading metrics: {exc}")


def read_dvc_lock():
    if not DVC_LOCK.exists():
        return
    try:
        with open(DVC_LOCK) as f:
            lock = yaml.safe_load(f)

        stages = lock.get("stages", {})

        # model size from train stage
        train_outs = stages.get("train", {}).get("outs", [])
        for out in train_outs:
            if "svm.pkl" in out.get("path", ""):
                dvc_stage_model_size.set(out.get("size", 0))

        # vectorizer size from featurize stage
        feat_outs = stages.get("featurize", {}).get("outs", [])
        for out in feat_outs:
            if "tfidf_vectorizer" in out.get("path", ""):
                dvc_stage_vectorizer_size.set(out.get("size", 0))
            if "features.pkl" in out.get("path", ""):
                dvc_stage_features_size.set(out.get("size", 0))

        # processed data size from preprocess stage
        pre_outs = stages.get("preprocess", {}).get("outs", [])
        for out in pre_outs:
            if "processed.pkl" in out.get("path", ""):
                dvc_stage_processed_size.set(out.get("size", 0))

        # pipeline info tag
        stage_names = list(stages.keys())
        dvc_pipeline_info.info({
            "stages":       ",".join(stage_names),
            "stage_count":  str(len(stage_names)),
            "schema":       lock.get("schema", "unknown"),
        })

        print(f"[dvc-exporter] pipeline — stages={stage_names}")
    except Exception as exc:
        print(f"[dvc-exporter] Error reading dvc.lock: {exc}")


def read_params():
    if not PARAMS_FILE.exists():
        return
    try:
        with open(PARAMS_FILE) as f:
            params = yaml.safe_load(f)

        dvc_param_svm_c.set(params.get("svm", {}).get("C", 0))
        dvc_param_svm_max_iter.set(params.get("svm", {}).get("max_iter", 0))
        dvc_param_tfidf_features.set(params.get("tfidf", {}).get("max_features", 0))
        dvc_param_test_size.set(params.get("data", {}).get("test_size", 0))

        print(f"[dvc-exporter] params — C={params.get('svm', {}).get('C')}  max_features={params.get('tfidf', {}).get('max_features')}")
    except Exception as exc:
        print(f"[dvc-exporter] Error reading params: {exc}")


if __name__ == "__main__":
    print(f"[dvc-exporter] Starting DVC exporter on :9400  (project: {BASE_DIR})")
    start_http_server(9400)
    while True:
        read_metrics()
        read_dvc_lock()
        read_params()
        time.sleep(SCRAPE_INTERVAL)
