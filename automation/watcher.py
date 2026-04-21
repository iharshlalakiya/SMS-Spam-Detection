"""
Data Watcher — Auto-trigger training pipeline on new data
==========================================================
Watches data/raw/ for any new or changed .csv file.
When a change is detected it runs:
    1. dvc repro          (preprocess → featurize → train → evaluate)
    2. python main.py     (full pipeline with MLflow tracking)

Run locally:
    python automation/watcher.py

Run on Azure VM (via systemd):
    sudo systemctl start sms-watcher
"""

import hashlib
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
WATCH_DIR = ROOT / "data" / "raw"
LOG_FILE  = ROOT / "logs" / "watcher.log"
INTERVAL  = 30   # seconds between checks

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("watcher")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(directory: Path) -> dict[str, str]:
    """Return {filename: md5} for all .csv files in directory."""
    return {
        p.name: md5(p)
        for p in sorted(directory.glob("*.csv"))
        if p.is_file()
    }


def run_pipeline():
    log.info("=" * 60)
    log.info("NEW DATA DETECTED — starting training pipeline")
    log.info("=" * 60)

    # Step 1: dvc repro
    log.info("Running: dvc repro")
    result = subprocess.run(
        ["dvc", "repro"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    log.info(result.stdout[-3000:] if result.stdout else "(no stdout)")
    if result.returncode != 0:
        log.error("dvc repro FAILED:\n%s", result.stderr[-2000:])
        return False

    # Step 2: python main.py (MLflow tracking)
    log.info("Running: python main.py")
    result = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--skip-download"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    log.info(result.stdout[-3000:] if result.stdout else "(no stdout)")
    if result.returncode != 0:
        log.error("main.py FAILED:\n%s", result.stderr[-2000:])
        return False

    log.info("Pipeline completed successfully.")
    return True


def main():
    log.info("Watcher started — monitoring: %s", WATCH_DIR)
    log.info("Check interval: %ds", INTERVAL)
    WATCH_DIR.mkdir(parents=True, exist_ok=True)

    last_snapshot = snapshot(WATCH_DIR)
    log.info("Initial snapshot: %s", list(last_snapshot.keys()))

    while True:
        time.sleep(INTERVAL)
        try:
            current = snapshot(WATCH_DIR)

            new_files     = set(current) - set(last_snapshot)
            changed_files = {f for f in current if f in last_snapshot and current[f] != last_snapshot[f]}

            if new_files:
                log.info("New file(s) detected: %s", new_files)
            if changed_files:
                log.info("Changed file(s) detected: %s", changed_files)

            if new_files or changed_files:
                success = run_pipeline()
                if success:
                    last_snapshot = current   # update only on success
                else:
                    log.warning("Pipeline failed — will retry on next change.")
            else:
                log.debug("No changes detected.")

        except KeyboardInterrupt:
            log.info("Watcher stopped by user.")
            break
        except Exception as e:
            log.error("Watcher error: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
