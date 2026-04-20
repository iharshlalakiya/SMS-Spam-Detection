"""
monitoring/mlflow_exporter.py
Exports latest MLflow run metrics to Prometheus format.
Runs on port 9200.
"""

import time
import os
from prometheus_client import start_http_server, Gauge, Info
import mlflow
from mlflow.tracking import MlflowClient

# ── Prometheus metrics ─────────────────────────────────────────────────────
model_accuracy   = Gauge("mlflow_model_accuracy",   "Latest model accuracy")
model_precision  = Gauge("mlflow_model_precision",  "Latest model precision")
model_recall     = Gauge("mlflow_model_recall",     "Latest model recall")
model_f1         = Gauge("mlflow_model_f1_score",   "Latest model F1 score")
model_roc_auc    = Gauge("mlflow_model_roc_auc",    "Latest model ROC AUC")
model_train_time = Gauge("mlflow_training_time_s",  "Latest model training time (s)")
total_runs       = Gauge("mlflow_total_runs",        "Total MLflow runs logged")
model_info       = Info("mlflow_latest_run",         "Latest MLflow run metadata")

TRACKING_URI     = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME  = os.environ.get("MLFLOW_EXPERIMENT_NAME", "SMS-Spam-Detection")
SCRAPE_INTERVAL  = int(os.environ.get("SCRAPE_INTERVAL", "30"))


def fetch_and_update():
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        client = MlflowClient()

        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            return

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            return

        run = runs[0]
        m   = run.data.metrics

        model_accuracy.set(m.get("accuracy",       0))
        model_precision.set(m.get("precision",      0))
        model_recall.set(m.get("recall",            0))
        model_f1.set(m.get("f1_score",              0))
        model_roc_auc.set(m.get("roc_auc",          0))
        model_train_time.set(m.get("training_time_s", 0))

        all_runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        total_runs.set(len(all_runs))

        model_info.info({
            "run_id":     run.info.run_id[:8],
            "run_name":   run.data.tags.get("mlflow.runName", "unknown"),
            "model_type": run.data.tags.get("model_type", "unknown"),
            "git_commit": run.data.tags.get("git_commit", "unknown"),
            "status":     run.info.status,
        })

        print(f"[exporter] Updated — accuracy={m.get('accuracy', 0):.4f}  f1={m.get('f1_score', 0):.4f}")

    except Exception as exc:
        print(f"[exporter] Error fetching MLflow metrics: {exc}")


if __name__ == "__main__":
    print(f"[exporter] Starting MLflow exporter on :9200  (tracking: {TRACKING_URI})")
    start_http_server(9200)
    while True:
        fetch_and_update()
        time.sleep(SCRAPE_INTERVAL)
