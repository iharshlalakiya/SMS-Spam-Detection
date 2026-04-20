"""
monitoring/app_metrics.py
Exposes Streamlit app metrics to Prometheus on port 9300.
Import and call start_metrics_server() once at app startup.
"""

import time
import threading
from prometheus_client import start_http_server, Counter, Histogram, Gauge

# ── Metrics ────────────────────────────────────────────────────────────────
predictions_total = Counter(
    "sms_predictions_total",
    "Total number of predictions made",
    ["result"],          # labels: spam / ham
)

prediction_latency = Histogram(
    "sms_prediction_latency_seconds",
    "Time taken to make a prediction",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)

app_errors_total = Counter(
    "sms_app_errors_total",
    "Total number of app errors",
)

model_loaded = Gauge(
    "sms_model_loaded",
    "1 if model is loaded successfully, 0 otherwise",
)

active_sessions = Gauge(
    "sms_active_sessions",
    "Approximate number of active Streamlit sessions",
)

_server_started = False


def start_metrics_server(port: int = 9300) -> None:
    """Start Prometheus metrics HTTP server in a background thread."""
    global _server_started
    if _server_started:
        return
    try:
        thread = threading.Thread(
            target=start_http_server,
            args=(port,),
            daemon=True,
        )
        thread.start()
        _server_started = True
        print(f"[metrics] Prometheus metrics server started on :{port}")
    except Exception as exc:
        print(f"[metrics] Could not start metrics server: {exc}")


def record_prediction(label: int, latency: float) -> None:
    result = "spam" if label == 1 else "ham"
    predictions_total.labels(result=result).inc()
    prediction_latency.observe(latency)


def record_error() -> None:
    app_errors_total.inc()


def set_model_loaded(loaded: bool) -> None:
    model_loaded.set(1 if loaded else 0)
