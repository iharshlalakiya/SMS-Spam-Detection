"""Smoke tests for sms_spam.models.svm.SpamDetector."""

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sms_spam.models.svm import SpamDetector, get_model_description

# Tiny labelled dataset for fast tests
TEXTS  = ["free prize call now", "win money now", "hi how are you", "see you tomorrow"]
LABELS = np.array([1, 1, 0, 0])


@pytest.fixture()
def fitted_detector():
    vec = TfidfVectorizer(max_features=50)
    X   = vec.fit_transform(TEXTS)
    det = SpamDetector()
    det.train(X, LABELS)
    return det, X


class TestSpamDetector:
    def test_is_trained_after_train(self, fitted_detector):
        det, _ = fitted_detector
        assert det.is_trained is True

    def test_predict_returns_binary(self, fitted_detector):
        det, X = fitted_detector
        preds = det.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_length_matches_input(self, fitted_detector):
        det, X = fitted_detector
        assert len(det.predict(X)) == X.shape[0]

    def test_predict_proba_shape(self, fitted_detector):
        det, X = fitted_detector
        proba = det.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)

    def test_predict_proba_sums_to_one(self, fitted_detector):
        det, X = fitted_detector
        proba = det.predict_proba(X)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(X.shape[0]))

    def test_save_load_roundtrip(self, fitted_detector, tmp_path):
        det, X = fitted_detector
        path = str(tmp_path / "svm.pkl")
        det.save(path)

        det2 = SpamDetector()
        det2.load(path)
        np.testing.assert_array_equal(det.predict(X), det2.predict(X))

    def test_get_training_diagnostics(self, fitted_detector):
        det, _ = fitted_detector
        diag = det.get_training_diagnostics()
        assert isinstance(diag, dict)
        assert "support_vectors" in diag

    def test_training_time_set(self, fitted_detector):
        det, _ = fitted_detector
        assert det.training_time is not None and det.training_time > 0


class TestGetModelDescription:
    def test_returns_dict(self):
        desc = get_model_description()
        assert isinstance(desc, dict)

    def test_has_name_and_description(self):
        desc = get_model_description()
        assert "name" in desc and "description" in desc
