"""
sms_spam/train.py — Training Pipeline Steps
=============================================
Contains the four steps that prepare data and train the SVM:

    step_preprocess  →  load + clean + preprocess text
    step_split       →  train / test split
    step_features    →  TF-IDF feature extraction
    step_train       →  fit the SVM classifier

These functions are imported and called by main.py.
"""

import sys
import yaml
import pickle
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from sms_spam.data.preprocessing  import load_data, preprocess_pipeline
from sms_spam.features.extraction import TFIDFExtractor
from sms_spam.models.svm          import SpamDetector


def step_preprocess(data_path: Path) -> pd.DataFrame:
    """
    Load the raw CSV and run the full text-preprocessing pipeline.

    Parameters
    ----------
    data_path : Path
        Path to spam.csv

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: label (int), message, processed_text, …
    """
    df = load_data(str(data_path))
    df = df.rename(columns={"v1": "label", "v2": "message"})
    df = df[["label", "message"]]
    df = preprocess_pipeline(df, text_column="message")
    df["label"] = (df["label"] == "spam").astype(int)
    return df


def step_split(df: pd.DataFrame):
    """
    Stratified 80/20 train-test split.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame from step_preprocess().

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df["processed_text"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"],
    )
    return X_train, X_test, y_train, y_test


def step_features(X_train, X_test):
    """
    Fit a TF-IDF vectorizer on X_train and transform both splits.

    Parameters
    ----------
    X_train, X_test : pd.Series
        Raw preprocessed text series.

    Returns
    -------
    tfidf : TFIDFExtractor
        Fitted vectorizer (needed later for saving / inference).
    X_train_tfidf, X_test_tfidf : sparse matrix
        TF-IDF feature matrices.
    """
    tfidf = TFIDFExtractor(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)
    return tfidf, X_train_tfidf, X_test_tfidf


def step_train(X_train_tfidf, y_train) -> SpamDetector:
    """
    Instantiate and fit the SVM spam classifier.

    Parameters
    ----------
    X_train_tfidf : sparse matrix
        TF-IDF features for the training split.
    y_train : array-like
        Binary labels (0 = ham, 1 = spam).

    Returns
    -------
    SpamDetector
        Fitted classifier.
    """
    detector = SpamDetector()
    detector.train(X_train_tfidf, y_train.values, sklearn_verbose=0)
    return detector


def run_dvc_stage(
    features_path: Path,
    models_dir: Path,
    C: float,
    kernel: str,
    max_iter: int,
    probability: bool,
) -> None:
    print(f"Loading features from {features_path}")
    with open(features_path, "rb") as f:
        data = pickle.load(f)

    X_train_tfidf = data["X_train_tfidf"]
    y_train       = data["y_train"]

    print(f"Training SVM (C={C}, kernel={kernel}, max_iter={max_iter})")
    detector = SpamDetector(C=C, kernel=kernel, max_iter=max_iter, probability=probability)
    detector.train(X_train_tfidf, y_train.values, sklearn_verbose=0)
    print(f"Training complete in {detector.training_time:.2f}s")

    diag = detector.get_training_diagnostics()
    if diag:
        for k, v in diag.items():
            print(f"Diagnostics — {k}: {v}")

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "svm.pkl"
    detector.save(str(model_path))
    print(f"Model saved -> {model_path}")


if __name__ == "__main__":
    # DVC Execution
    ROOT = Path(__file__).resolve().parents[2]
    with open(ROOT / "params.yaml") as f:
        params = yaml.safe_load(f)

    run_dvc_stage(
        features_path = ROOT / params["data"]["features_path"],
        models_dir    = ROOT / params["paths"]["models_dir"],
        C             = params["svm"]["C"],
        kernel        = params["svm"]["kernel"],
        max_iter      = params["svm"]["max_iter"],
        probability   = params["svm"]["probability"],
    )
    print(f"[stage_train] DONE  Model saved -> {ROOT / params['paths']['models_dir'] / 'svm.pkl'}")
