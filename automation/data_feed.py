"""
Data Feed Service — Continuous SMS data ingestion
==================================================
Simulates a real-time SMS data stream by appending new messages to
data/raw/spam.csv at a configurable interval, then triggering the
training pipeline so the HTML monitor reflects live results.

On Azure VM this runs as the `sms-data-feed` systemd service.

To use a real data source, replace `fetch_new_messages()` with your
API/Kafka/database call that returns a list of (label, message) tuples.

Run locally:
    python automation/data_feed.py
"""

import csv
import hashlib
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
RAW_CSV   = ROOT / "data" / "raw" / "spam.csv"
LOG_FILE  = ROOT / "logs" / "data_feed.log"

# How often to check for / inject new data (seconds)
FEED_INTERVAL = 60
# How many messages to inject per cycle (set 0 to disable simulation)
SIMULATE_BATCH = 5

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
log = logging.getLogger("data_feed")

# ---------------------------------------------------------------------------
# Sample pool — replace / extend with your real data source
# ---------------------------------------------------------------------------
_SAMPLE_POOL = [
    ("spam", "WINNER!! You have been selected for a £1000 prize. Call now!"),
    ("ham",  "Hey, are we still on for lunch tomorrow?"),
    ("spam", "FREE entry in 2 a weekly competition to win FA Cup final tkts!"),
    ("ham",  "I'll be home by 7. Can you start dinner?"),
    ("spam", "Urgent! Your mobile number has won £2000 bonus. Claim now."),
    ("ham",  "Don't forget the meeting at 3pm today."),
    ("spam", "Congratulations! You've won a free holiday. Text YES to 80488."),
    ("ham",  "Can you pick up some milk on your way home?"),
    ("spam", "Your account has been suspended. Verify now: http://bit.ly/xxx"),
    ("ham",  "Running 10 mins late, save me a seat."),
    ("spam", "You have 1 new voicemail. Call 0906 6475 to listen. Cost 25p/min."),
    ("ham",  "Thanks for the birthday wishes!"),
    ("spam", "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11"),
    ("ham",  "What time does the film start?"),
    ("spam", "Claim your FREE ringtone now! Reply TONE to 85233."),
    ("ham",  "I'm at the gym, back in an hour."),
]

_pool_index = 0


def fetch_new_messages() -> list[tuple[str, str]]:
    """
    Return a list of (label, message) tuples.

    Replace this function body with a real data source:
        - REST API call
        - Kafka consumer poll
        - Database query for new rows since last run
        - File drop from an upstream system
    """
    global _pool_index
    batch = []
    for _ in range(SIMULATE_BATCH):
        item = _SAMPLE_POOL[_pool_index % len(_SAMPLE_POOL)]
        batch.append(item)
        _pool_index += 1
    return batch


def csv_row_count() -> int:
    if not RAW_CSV.exists():
        return 0
    with open(RAW_CSV, encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f) - 1  # subtract header


def append_to_csv(messages: list[tuple[str, str]]) -> int:
    """Append messages to spam.csv, return count appended."""
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RAW_CSV.exists()
    appended = 0
    with open(RAW_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["v1", "v2"])
        for label, text in messages:
            writer.writerow([label, text])
            appended += 1
    return appended


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def run_pipeline() -> bool:
    log.info("Triggering pipeline after new data ingestion…")
    result = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--skip-download"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.stdout:
        log.info(result.stdout[-2000:])
    if result.returncode != 0:
        log.error("Pipeline failed:\n%s", result.stderr[-1000:])
        return False
    log.info("Pipeline completed successfully.")
    return True


def main():
    log.info("Data feed service started (interval=%ds, batch=%d)", FEED_INTERVAL, SIMULATE_BATCH)
    last_md5 = file_md5(RAW_CSV) if RAW_CSV.exists() else ""

    while True:
        try:
            messages = fetch_new_messages()

            if messages:
                before = csv_row_count()
                appended = append_to_csv(messages)
                after = csv_row_count()
                log.info("Appended %d messages (%d → %d total rows)", appended, before, after)

                new_md5 = file_md5(RAW_CSV)
                if new_md5 != last_md5:
                    run_pipeline()
                    last_md5 = new_md5
            else:
                log.debug("No new messages this cycle.")

        except KeyboardInterrupt:
            log.info("Data feed stopped.")
            break
        except Exception as exc:
            log.error("Feed error: %s", exc, exc_info=True)

        time.sleep(FEED_INTERVAL)


if __name__ == "__main__":
    main()
