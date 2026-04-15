#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SMS Spam Dataset Downloader
============================
Automatically downloads the SMS Spam Collection dataset from Kaggle
or falls back to a direct URL mirror.

Usage
-----
    python scripts/download_dataset.py
"""

import os
import sys
import zipfile
import pandas as pd
from pathlib import Path

# Resolve project root so relative data paths are correct
ROOT = Path(__file__).resolve().parent.parent

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


def check_kaggle_credentials():
    """Return True if ~/.kaggle/kaggle.json exists."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("=" * 70)
        print("⚠️  Kaggle API credentials not found!")
        print("=" * 70)
        print("\nTo set up API credentials:")
        print("  1. Visit https://www.kaggle.com → Account Settings → API")
        print("  2. Click 'Create New Token' — saves kaggle.json")
        print(f"  3. Place it at: {kaggle_json}")
        print("  4. Run this script again")
        print("=" * 70)
        return False
    return True


def download_from_kaggle(data_dir):
    """Download via the official Kaggle API."""
    print("\n" + "=" * 70)
    print("📦 Downloading from Kaggle API")
    print("=" * 70)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("✓ Authenticated")
        data_dir.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files("uciml/sms-spam-collection-dataset",
                                   path=str(data_dir), unzip=True)
        print("✓ Downloaded")
        return True
    except Exception as exc:
        print(f"❌ Kaggle download failed: {exc}")
        return False


def download_alternative(data_dir):
    """Fall back to a raw GitHub mirror."""
    print("\n" + "=" * 70)
    print("🌐 Trying alternative (GitHub mirror)...")
    print("=" * 70)
    try:
        import urllib.request
        url = (
            "https://raw.githubusercontent.com/"
            "mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        dest = data_dir / "spam.csv"
        print(f"📥 {url}")
        urllib.request.urlretrieve(url, dest)
        if dest.exists() and dest.stat().st_size > 0:
            print(f"✓ Saved to {dest}")
            return True
        print("❌ File empty or missing after download")
        return False
    except Exception as exc:
        print(f"❌ Alternative download failed: {exc}")
        return False


def verify_dataset(data_file):
    """Load and summarise the dataset; return True on success."""
    print("\n" + "=" * 70)
    print("🔍 Verifying dataset...")
    print("=" * 70)
    if not data_file.exists():
        print(f"❌ Not found: {data_file}")
        return False
    try:
        df = pd.read_csv(data_file, encoding="latin-1")
        print(f"✓ {len(df):,} rows  |  {len(df.columns)} columns  |  {data_file.stat().st_size/1024:.1f} KB")
        print(f"  Columns: {list(df.columns)}")
        print(df.head())
        return True
    except Exception as exc:
        print(f"❌ Verification failed: {exc}")
        return False


def main():
    print("\n" + "=" * 70)
    print("🚀 SMS Spam Dataset Downloader")
    print("=" * 70)

    data_dir  = ROOT / "data" / "raw"
    data_file = data_dir / "spam.csv"

    if data_file.exists():
        print(f"\n✓ Dataset already present: {data_file}")
        resp = input("\nRe-download? (y/N): ").strip().lower()
        if resp != "y":
            verify_dataset(data_file)
            return

    success = False

    if check_kaggle_credentials():
        success = download_from_kaggle(data_dir)

    if not success:
        print("\n💡 Trying mirror...")
        success = download_alternative(data_dir)

    if success and verify_dataset(data_file):
        print("\n" + "=" * 70)
        print("✅ Dataset ready!")
        print("=" * 70)
        print("\nNext step:")
        print("  python scripts/train.py")
        return

    print("\n" + "=" * 70)
    print("❌ Automatic download failed — please download manually:")
    print("=" * 70)
    print("\n  https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")
    print(f"\n  Place spam.csv in: {data_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌ Unexpected error: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)
