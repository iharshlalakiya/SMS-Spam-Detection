#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sms_spam/pipeline/stage_download.py — DVC Stage 1: Download
============================================================
Ensures the raw dataset (spam.csv) is present.

Called by DVC via:
    python -m sms_spam.pipeline.stage_download

Reads params.yaml:
    data.raw_path
"""

import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sms_spam.logs.logger import get_logger

log = get_logger("pipeline.download")


def download(raw_path: Path) -> None:
    if raw_path.exists() and raw_path.stat().st_size > 0:
        log.info("Dataset already present: %s", raw_path)
        return

    raw_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Mirror 1: GitHub ──────────────────────────────────────────────────────
    url = (
        "https://raw.githubusercontent.com/"
        "mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
    )
    log.info("Downloading dataset from GitHub mirror -> %s", raw_path)
    try:
        urllib.request.urlretrieve(url, raw_path)
        if raw_path.exists() and raw_path.stat().st_size > 0:
            log.info("Download complete — %d bytes", raw_path.stat().st_size)
            return
    except Exception as exc:
        log.warning("Mirror download failed: %s", exc)

    # ── Mirror 2: Kaggle API ──────────────────────────────────────────────────
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            "uciml/sms-spam-collection-dataset",
            path=str(raw_path.parent),
            unzip=True,
        )
        if raw_path.exists():
            log.info("Downloaded via Kaggle API")
            return
    except Exception as exc:
        log.warning("Kaggle download failed: %s", exc)

    log.error("Could not download dataset. Place spam.csv at: %s", raw_path)
    sys.exit(1)


if __name__ == "__main__":
    with open(ROOT / "params.yaml") as f:
        params = yaml.safe_load(f)

    raw_path = ROOT / params["data"]["raw_path"]
    download(raw_path)
    print(f"[stage_download] DONE  Dataset ready at {raw_path}")
