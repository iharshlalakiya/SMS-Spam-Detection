"""
Feature Extraction Module for SMS Spam Detection

Provides TF-IDF vectorization for the SVM classifier,
plus a helper for manual hand-crafted features.
"""

import sys
import numpy as np
import pickle
import yaml
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFExtractor:
    """
    TF-IDF Feature Extractor for the SVM spam classifier.

    Wraps ``sklearn.feature_extraction.text.TfidfVectorizer`` with a
    consistent ``fit`` / ``transform`` / ``fit_transform`` interface and
    pickle-based persistence.
    """

    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Parameters
        ----------
        max_features : int
            Maximum vocabulary size kept by TF-IDF.
        ngram_range : tuple[int, int]
            The lower and upper boundary of the n-gram range.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
        )
        self.is_fitted = False

    def fit(self, texts):
        """Fit the vectorizer on training texts."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        print(f"TF-IDF Vectorizer fitted with {len(self.vectorizer.vocabulary_)} features")

    def transform(self, texts):
        """Transform texts to TF-IDF feature matrix."""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        """Fit and transform in one step."""
        features = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        print(f"TF-IDF Vectorizer fitted with {len(self.vectorizer.vocabulary_)} features")
        return features

    def get_feature_names(self):
        """Return the feature names (vocabulary terms)."""
        return self.vectorizer.get_feature_names_out()

    def save(self, filepath):
        """Persist vectorizer to a pickle file."""
        with open(filepath, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"Vectorizer saved to {filepath}")

    def load(self, filepath):
        """Load vectorizer from a pickle file."""
        with open(filepath, "rb") as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True
        print(f"Vectorizer loaded from {filepath}")


def extract_manual_features(texts):
    """
    Extract hand-crafted numeric features from a list of text strings.

    Features per message: length, word count, average word length,
    capital-letter ratio, digit ratio, and special-character ratio.

    Parameters
    ----------
    texts : list[str]
        Raw or preprocessed message strings.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(len(texts), 6)``.
    """
    features = []
    for text in texts:
        words = text.split()
        features.append([
            len(text),
            len(words),
            np.mean([len(w) for w in words]) if words else 0.0,
            sum(1 for c in text if c.isupper()) / len(text) if text else 0.0,
            sum(1 for c in text if c.isdigit()) / len(text) if text else 0.0,
            sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if text else 0.0,
        ])
    return np.array(features)


def run_dvc_stage(
    processed_path: Path,
    features_path: Path,
    vectorizer_path: Path,
    max_features: int,
    ngram_range: list,
) -> None:
    print(f"Loading preprocessed split from {processed_path}")
    with open(processed_path, "rb") as f:
        data = pickle.load(f)

    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]

    print(f"Fitting TF-IDF (max_features={max_features}, ngram_range={ngram_range})")
    tfidf = TFIDFExtractor(
        max_features=max_features,
        ngram_range=tuple(ngram_range),
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)
    print(f"Feature dimensions: {X_train_tfidf.shape[1]}")

    # ── Save features ─────────────────────────────────────────────────────────
    features_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        X_train_tfidf=X_train_tfidf,
        X_test_tfidf=X_test_tfidf,
        y_train=y_train,
        y_test=y_test,
    )
    with open(features_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Features saved -> {features_path}")

    # ── Save fitted vectorizer ────────────────────────────────────────────────
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf, vectorizer_path)
    print(f"Vectorizer saved -> {vectorizer_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        sample_texts = [
            "free prize winner call now",
            "hey how are you doing today",
            "urgent claim your cash prize",
            "meeting tomorrow at office",
        ]
        tfidf = TFIDFExtractor(max_features=100)
        tfidf_features = tfidf.fit_transform(sample_texts)
        print(f"TF-IDF shape: {tfidf_features.shape}")
    else:
        # DVC Execution
        ROOT = Path(__file__).resolve().parents[2]
        with open(ROOT / "params.yaml") as f:
            params = yaml.safe_load(f)

        run_dvc_stage(
            processed_path  = ROOT / params["data"]["processed_path"],
            features_path   = ROOT / params["data"]["features_path"],
            vectorizer_path = ROOT / params["paths"]["models_dir"] / "tfidf_vectorizer.pkl",
            max_features    = params["tfidf"]["max_features"],
            ngram_range     = params["tfidf"]["ngram_range"],
        )
        print("[stage_featurize] DONE  Features saved")
