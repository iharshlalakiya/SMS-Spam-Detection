"""
sms_spam/mlflow_registry.py — MLflow Model Registry Manager
=============================================================
Utility for managing the MLflow Model Registry lifecycle:

    NONE  →  Staging  →  Production  →  Archived

Usage
-----
    from sms_spam.mlflow_registry import ModelRegistryManager

    registry = ModelRegistryManager()

    # Promote a version to Staging after CI passes
    registry.transition("SmsSpamDetector", version=2, stage="Staging")

    # Promote to Production after QA
    registry.transition("SmsSpamDetector", version=2, stage="Production")

    # Load the current Production model
    model = registry.load_production_model("SmsSpamDetector")

    # Print the full version history
    registry.print_version_history("SmsSpamDetector")

Run as a script
---------------
    # Transition latest Staging → Production
    python -m sms_spam.mlflow_registry --promote SmsSpamDetector

    # List all versions
    python -m sms_spam.mlflow_registry --list SmsSpamDetector
"""

from __future__ import annotations

import os
import argparse
from typing import Optional

from sms_spam.logs.logger import get_logger
log = get_logger(__name__)

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


_VALID_STAGES = {"None", "Staging", "Production", "Archived"}


class ModelRegistryManager:
    """
    Manages MLflow Model Registry lifecycle transitions.

    Stages
    ------
    None  →  Staging  →  Production  →  Archived

    Parameters
    ----------
    tracking_uri : str | None
        MLflow tracking URI. Reads MLFLOW_TRACKING_URI env var if not provided.
    """

    def __init__(self, tracking_uri: Optional[str] = None) -> None:
        if not _MLFLOW_AVAILABLE:
            raise RuntimeError("mlflow is not installed. Run: pip install mlflow>=2.10")

        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
        mlflow.set_tracking_uri(uri)
        self._client = MlflowClient()
        log.info("ModelRegistryManager initialised — tracking URI: %s", uri)

    # ── Lifecycle transitions ──────────────────────────────────────────────────

    def transition(
        self,
        model_name: str,
        version: int,
        stage: str,
        description: Optional[str] = None,
    ) -> None:
        """
        Transition a registered model version to a new lifecycle stage.

        Parameters
        ----------
        model_name  : str  Registered model name (e.g. "SmsSpamDetector")
        version     : int  Version number to transition
        stage       : str  Target stage: "Staging", "Production", or "Archived"
        description : str  Optional comment recorded in the registry

        Example
        -------
        registry.transition("SmsSpamDetector", version=2, stage="Production",
                             description="Improved F1 from 0.918 to 0.940")
        """
        if stage not in _VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Choose from: {_VALID_STAGES}")

        self._client.transition_model_version_stage(
            name         = model_name,
            version      = str(version),
            stage        = stage,
            archive_existing_versions = (stage == "Production"),
            # Auto-archive previous Production when promoting new one
        )
        if description:
            self._client.update_model_version(
                name        = model_name,
                version     = str(version),
                description = description,
            )

        print(f"   ✅ {model_name} v{version}  →  {stage}")
        log.info("Transitioned %s v%s → %s", model_name, version, stage)

    def promote_latest_to_staging(self, model_name: str) -> None:
        """Promote the latest registered version to Staging."""
        latest = self._get_latest_version(model_name, stage="None")
        if latest is None:
            print(f"   ⚠️  No undeployed versions found for '{model_name}'")
            return
        self.transition(model_name, int(latest.version), "Staging")

    def promote_staging_to_production(self, model_name: str) -> None:
        """Promote the current Staging version to Production."""
        staging = self._get_latest_version(model_name, stage="Staging")
        if staging is None:
            print(f"   ⚠️  No Staging version found for '{model_name}'")
            return
        self.transition(
            model_name, int(staging.version), "Production",
            description="Promoted from Staging after QA approval",
        )

    # ── Model loading ──────────────────────────────────────────────────────────

    def load_production_model(self, model_name: str):
        """
        Load and return the current Production model.

        Returns
        -------
        sklearn estimator loaded from the MLflow registry.

        Example
        -------
        clf = registry.load_production_model("SmsSpamDetector")
        predictions = clf.predict(X_test)
        """
        uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(uri)
        log.info("Loaded Production model: %s", uri)
        return model

    def load_staging_model(self, model_name: str):
        """Load the current Staging model for pre-production testing."""
        uri = f"models:/{model_name}/Staging"
        model = mlflow.sklearn.load_model(uri)
        log.info("Loaded Staging model: %s", uri)
        return model

    # ── Inspection helpers ─────────────────────────────────────────────────────

    def print_version_history(self, model_name: str) -> None:
        """Print a formatted table of all registered versions and their stages."""
        versions = self._client.search_model_versions(f"name='{model_name}'")
        if not versions:
            print(f"No registered versions found for '{model_name}'")
            return

        print(f"\n{'─'*65}")
        print(f"  Model Registry: {model_name}")
        print(f"{'─'*65}")
        print(f"  {'Version':<10} {'Stage':<15} {'Run ID':<20} {'Description'}")
        print(f"{'─'*65}")
        for v in sorted(versions, key=lambda x: int(x.version)):
            desc = (v.description or "")[:30]
            run_short = v.run_id[:8] if v.run_id else "N/A"
            print(f"  v{v.version:<9} {v.current_stage:<15} {run_short:<20} {desc}")
        print(f"{'─'*65}\n")

    def get_production_version(self, model_name: str) -> Optional[str]:
        """Return the version number of the current Production model, or None."""
        prod = self._get_latest_version(model_name, stage="Production")
        return prod.version if prod else None

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_latest_version(self, model_name: str, stage: str):
        """Return the most recent model version in a given stage."""
        try:
            versions = self._client.get_latest_versions(model_name, stages=[stage])
            return versions[0] if versions else None
        except Exception as exc:
            log.warning("Could not fetch versions for %s (%s): %s", model_name, stage, exc)
            return None


# ── CLI entry point ────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="MLflow Model Registry Manager for SMS Spam Detection"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list",    metavar="MODEL", help="List all versions of a model")
    group.add_argument("--promote", metavar="MODEL", help="Promote latest Staging → Production")
    group.add_argument("--stage",   metavar="MODEL", help="Transition specific version to stage")

    parser.add_argument("--version",  type=int, help="Version number (used with --stage)")
    parser.add_argument("--to",       help="Target stage (used with --stage)")
    parser.add_argument("--desc",     help="Description / comment for this transition")
    parser.add_argument("--uri",      help="MLflow tracking URI (overrides env var)")

    args = parser.parse_args()
    registry = ModelRegistryManager(tracking_uri=args.uri)

    if args.list:
        registry.print_version_history(args.list)

    elif args.promote:
        registry.promote_staging_to_production(args.promote)

    elif args.stage:
        if not args.version or not args.to:
            parser.error("--stage requires --version and --to")
        registry.transition(args.stage, args.version, args.to, description=args.desc)


if __name__ == "__main__":
    _cli()
