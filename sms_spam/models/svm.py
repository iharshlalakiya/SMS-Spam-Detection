"""
SVM Spam Detector
=================
Wraps ``sklearn.svm.SVC`` with a consistent train / predict / save / load
interface used throughout the SMS Spam Detection pipeline.
"""

import numpy as np
import pickle
import time
from sklearn.svm import SVC


class SpamDetector:
    """SVM-based spam classifier."""

    def __init__(
        self,
        model_type="svm",   # kept for API compatibility
        C: float = 1.0,
        kernel: str = "linear",
        max_iter: int = -1,
        probability: bool = True,
    ):
        self.model_type = "svm"
        self.model = SVC(
            kernel=kernel,
            C=C,
            max_iter=max_iter,
            probability=probability,
            random_state=42,
        )
        self.training_time = None
        self.is_trained = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _configure_verbose(self, sklearn_verbose=0):
        """Enable native libsvm verbosity if requested."""
        if sklearn_verbose <= 0:
            return
        params = self.model.get_params()
        if "verbose" in params:
            self.model.set_params(verbose=bool(sklearn_verbose))
            print(f"Native sklearn verbosity enabled for SVM (level={sklearn_verbose}).")
        else:
            print("Native sklearn verbosity is not available for SVM.")

    def _fit_with_fallback(self, X_train, y_train):
        """Retry with n_jobs=1 when multiprocessing is blocked (Windows)."""
        try:
            self.model.fit(X_train, y_train)
        except PermissionError:
            params = self.model.get_params() if hasattr(self.model, "get_params") else {}
            if "n_jobs" in params and params["n_jobs"] != 1:
                print("Multiprocessing is unavailable here; retrying with n_jobs=1...")
                self.model.set_params(n_jobs=1)
                self.model.fit(X_train, y_train)
            else:
                raise

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self, X_train, y_train, **kwargs):
        """
        Fit the SVM on training data.

        Parameters
        ----------
        X_train : array-like
            Feature matrix (TF-IDF sparse matrix or dense array).
        y_train : array-like
            Binary labels (0 = ham, 1 = spam).
        sklearn_verbose : int, optional
            Verbosity level passed to the underlying libsvm solver.
        """
        print("Training SVM...")
        start = time.time()
        self._configure_verbose(sklearn_verbose=kwargs.get("sklearn_verbose", 0))
        self._fit_with_fallback(X_train, y_train)
        self.training_time = time.time() - start
        self.is_trained = True
        print(f"Training completed in {self.training_time:.2f} seconds")

    def predict(self, X_test):
        """Return predicted class labels."""
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """Return class-probability estimates (shape: n_samples × 2)."""
        return self.model.predict_proba(X_test)

    def save(self, filepath):
        """Pickle the fitted model to *filepath*."""
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath):
        """Load a previously saved model from *filepath* (instance method)."""
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True

    @classmethod
    def from_file(cls, filepath) -> "SpamDetector":
        """Classmethod: create a SpamDetector and load a saved model."""
        instance = cls()
        instance.load(filepath)
        return instance

    def get_training_diagnostics(self):
        """Return lightweight post-training stats."""
        if not self.is_trained:
            return {}
        if hasattr(self.model, "n_support_"):
            return {"support_vectors": int(np.sum(self.model.n_support_))}
        return {}


def get_model_description():
    """Return a human-readable description of the SVM model."""
    return {
        "name": "Support Vector Machine",
        "description": "Linear SVM that finds the optimal hyperplane to separate spam from ham.",
    }
