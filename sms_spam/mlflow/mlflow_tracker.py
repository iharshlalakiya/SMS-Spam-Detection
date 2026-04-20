"""
sms_spam/mlflow_tracker.py — MLflow Experiment Tracking Utility
================================================================
Industry-standard MLflow wrapper for the SMS Spam Detection project.

Features
--------
- Environment variable override for CI/DagsHub (MLFLOW_TRACKING_URI etc.)
- Git commit SHA, Python version, OS tagging for reproducibility
- pip environment snapshot logged as artifact
- Context-manager support with automatic run cleanup
- Graceful no-op when MLflow is unavailable

Usage
-----
    from sms_spam.mlflow_tracker import MlflowTracker

    tracker = MlflowTracker(experiment_name="SMS-Spam-Detection")
    with tracker:
        tracker.log_params(params)
        tracker.log_training_info(detector)
        tracker.log_metrics(metrics)
        tracker.log_artifacts(cm_path, roc_path, json_path)
        tracker.log_model(detector, tfidf, models_dir,
                          register=True, model_name="SmsSpamDetector")
"""

from __future__ import annotations

import os
import sys
import platform
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from sms_spam.logs.logger import get_logger
log = get_logger(__name__)

try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    log.warning("mlflow not installed. Run `pip install mlflow>=2.10`.")


# ── Environment helpers ────────────────────────────────────────────────────────

def _get_git_commit() -> str:
    """Return the current git commit SHA (short), or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _get_tracking_uri(default: str) -> str:
    """
    Resolve MLflow tracking URI with CI override support.

    Priority: MLFLOW_TRACKING_URI env var  >  params.yaml value
    This allows CI (GitHub Actions + DagsHub) to override without changing code.
    """
    return os.environ.get("MLFLOW_TRACKING_URI", default)


def _set_remote_credentials() -> None:
    """
    Set HTTP authentication from env vars if running against a remote server.
    Used for DagsHub / MLflow hosted server in CI.
    """
    username = os.environ.get("MLFLOW_TRACKING_USERNAME")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        log.info("MLflow remote credentials configured from env vars")


def _snapshot_environment(tmp_dir: str) -> Optional[str]:
    """
    Write a pip freeze snapshot to a temp file and return its path.
    Logged as an artifact so any run can be reproduced exactly.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=30
        )
        snap_path = os.path.join(tmp_dir, "requirements_snapshot.txt")
        with open(snap_path, "w") as f:
            f.write(result.stdout)
        return snap_path
    except Exception as exc:
        log.warning("Could not capture pip environment: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════

class MlflowTracker:
    """
    Production-grade MLflow tracking wrapper.

    Supports:
    - Local tracking  (mlruns/) for development
    - Remote tracking (DagsHub/hosted) for CI via env var override
    - Full reproducibility tagging (git, python, os)
    - Graceful no-op when MLflow is unavailable

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name (created if it does not exist).
    tracking_uri : str
        Local dir or remote URI. Overridden by MLFLOW_TRACKING_URI env var in CI.
    run_name : str | None
        Human-readable label. Auto-generated as ``{model}-{timestamp}`` if None.
    model_type : str
        Model type tag (e.g. "SVM", "RandomForest"). Shown in run tags.
    tags : dict | None
        Any extra tags to attach to the run.
    """

    def __init__(
        self,
        experiment_name: str = "SMS-Spam-Detection",
        tracking_uri: str = "mlruns",
        run_name: Optional[str] = None,
        model_type: str = "SVM",
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        self.experiment_name = experiment_name
        # CI env var takes priority over params.yaml value
        self.tracking_uri    = _get_tracking_uri(tracking_uri)
        self.run_name        = run_name or f"{model_type.lower()}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.model_type      = model_type
        self.tags            = tags or {}
        self._run            = None
        self._active         = False

    # ── Context-manager ────────────────────────────────────────────────────────

    def __enter__(self) -> "MlflowTracker":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self._set_status("FAILED")
        self.end()
        return False   # always propagate exceptions

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Configure tracking URI, set credentials (for remote), create experiment,
        and start a new run with full reproducibility tags.
        """
        if not _MLFLOW_AVAILABLE:
            return
        try:
            # Set credentials before connecting (no-op locally)
            _set_remote_credentials()

            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)

            # Build reproducibility tags
            run_tags = {
                "project":        "SMS-Spam-Detection",
                "model_type":     self.model_type,
                "git_commit":     _get_git_commit(),
                "python_version": platform.python_version(),
                "os":             platform.system(),
                "pipeline":       "main.py",
                **self.tags,
            }

            self._run = mlflow.start_run(
                run_name=self.run_name,
                tags=run_tags,
            )
            self._active = True

            is_remote = self.tracking_uri.startswith("http")
            dest = self.tracking_uri if is_remote else f"local mlruns/"
            log.info("MLflow run started — experiment: '%s'  run_id: %s  dest: %s",
                     self.experiment_name, self._run.info.run_id, dest)
            print(f"   📊 MLflow run started  →  {dest}")

        except Exception as exc:
            log.warning("MLflow start_run failed: %s", exc)
            self._active = False

    def end(self) -> None:
        """End active run and print the run ID for reference."""
        if not _MLFLOW_AVAILABLE or not self._active:
            return
        try:
            mlflow.end_run()
            self._active = False
            run_id = self._run.info.run_id if self._run else "unknown"
            log.info("MLflow run ended — run_id: %s", run_id)
            print(f"   📊 MLflow run complete  →  run id: {run_id}")
        except Exception as exc:
            log.warning("MLflow end_run failed: %s", exc)

    # ── Logging helpers ────────────────────────────────────────────────────────

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Flatten and log a nested params dict.

        Example
        -------
        tracker.log_params({
            "svm": {"C": 1.0, "kernel": "linear"},
            "tfidf": {"max_features": 5000},
        })
        → logs as: svm.C=1.0, svm.kernel=linear, tfidf.max_features=5000
        """
        if not _MLFLOW_AVAILABLE or not self._active:
            return
        try:
            flat = _flatten(params)
            mlflow.log_params(flat)
            log.info("MLflow — logged %d params", len(flat))
        except Exception as exc:
            log.warning("MLflow log_params failed: %s", exc)

    def log_training_info(self, detector) -> None:
        """Log training time and any diagnostics from the SpamDetector."""
        if not _MLFLOW_AVAILABLE or not self._active:
            return
        try:
            mlflow.log_metric("training_time_s", round(detector.training_time, 4))
            diag = detector.get_training_diagnostics() or {}
            for k, v in diag.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"train_{k}", float(v))
            log.info("MLflow — logged training diagnostics")
        except Exception as exc:
            log.warning("MLflow log_training_info failed: %s", exc)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log evaluation metrics (accuracy, precision, recall, f1_score, roc_auc)."""
        if not _MLFLOW_AVAILABLE or not self._active:
            return
        try:
            mlflow.log_metrics({k: round(float(v), 6) for k, v in metrics.items()})
            log.info("MLflow — logged metrics: %s",
                     {k: f"{v:.4f}" for k, v in metrics.items()})
        except Exception as exc:
            log.warning("MLflow log_metrics failed: %s", exc)

    def log_artifacts(
        self,
        cm_path:   Optional[Path] = None,
        roc_path:  Optional[Path] = None,
        json_path: Optional[Path] = None,
    ) -> None:
        """Log plots and metrics JSON. Missing files are silently skipped."""
        if not _MLFLOW_AVAILABLE or not self._active:
            return
        paths = {
            "confusion_matrix": cm_path,
            "roc_curve":        roc_path,
            "metrics_json":     json_path,
        }
        for label, path in paths.items():
            if path and Path(path).exists():
                try:
                    mlflow.log_artifact(str(path))
                    log.info("MLflow — logged artifact (%s): %s", label, path)
                except Exception as exc:
                    log.warning("MLflow log_artifact(%s) failed: %s", label, exc)

    def log_environment(self) -> None:
        """
        Capture a pip freeze snapshot and log it as an artifact.
        This ensures 100% reproducibility — any run can be recreated
        by installing from this snapshot.
        """
        if not _MLFLOW_AVAILABLE or not self._active:
            return
        try:
            with tempfile.TemporaryDirectory() as tmp:
                snap = _snapshot_environment(tmp)
                if snap:
                    mlflow.log_artifact(snap, artifact_path="environment")
                    log.info("MLflow — logged pip environment snapshot")
        except Exception as exc:
            log.warning("MLflow log_environment failed: %s", exc)

    def log_model(
        self,
        detector,
        tfidf,
        models_dir: Path,
        register:   bool = False,
        model_name: str  = "SmsSpamDetector",
    ) -> None:
        """
        Log the trained sklearn model to MLflow and optionally register it.

        Parameters
        ----------
        detector   : SpamDetector — trained SVM wrapper
        tfidf      : TFIDFExtractor — fitted vectorizer
        models_dir : Path — directory containing svm.pkl / tfidf_vectorizer.pkl
        register   : bool — register in Model Registry if True
        model_name : str  — registry model name
        """
        if not _MLFLOW_AVAILABLE or not self._active:
            return
        try:
            # Log raw pickle files as artifacts for DVC compatibility
            for fname in ["svm.pkl", "tfidf_vectorizer.pkl"]:
                fpath = Path(models_dir) / fname
                if fpath.exists():
                    mlflow.log_artifact(str(fpath), artifact_path="model_artifacts")

            # Log as native sklearn model (enables `mlflow models serve`)
            registered_model_name = model_name if register else None
            mlflow.sklearn.log_model(
                sk_model              = detector.model,
                artifact_path         = "sklearn_model",
                registered_model_name = registered_model_name,
            )
            log.info("MLflow — model logged%s",
                     f" and registered as '{model_name}'" if register else "")
            if register:
                print(f"   📊 Model registered  →  '{model_name}' in Model Registry")
        except Exception as exc:
            log.warning("MLflow log_model failed: %s", exc)

    # ── Status / tag helpers ───────────────────────────────────────────────────

    def _set_status(self, status: str) -> None:
        if not _MLFLOW_AVAILABLE or not self._active:
            return
        try:
            mlflow.set_tag("pipeline_status", status)
        except Exception:
            pass

    def set_tag(self, key: str, value: str) -> None:
        """Set an arbitrary tag on the active run."""
        if not _MLFLOW_AVAILABLE or not self._active:
            return
        try:
            mlflow.set_tag(key, value)
        except Exception as exc:
            log.warning("MLflow set_tag failed: %s", exc)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def run_id(self) -> Optional[str]:
        return self._run.info.run_id if self._run else None

    @property
    def is_active(self) -> bool:
        return self._active


# ── Internal helpers ──────────────────────────────────────────────────────────

def _flatten(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """
    Recursively flatten a nested dict for MLflow param logging.

    >>> _flatten({"svm": {"C": 1.0, "kernel": "linear"}})
    {'svm.C': '1.0', 'svm.kernel': 'linear'}
    """
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten(v, new_key, sep=sep))
        else:
            items[new_key] = str(v)[:500]   # MLflow param limit: 500 chars
    return items
