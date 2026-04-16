"""Smoke tests for sms_spam.features.extraction."""

import numpy as np
import pytest
from sms_spam.features.extraction import TFIDFExtractor, extract_manual_features

SAMPLE_TEXTS = [
    "free prize winner call now",
    "hey how are you doing today",
    "urgent claim your cash reward",
    "meeting scheduled for tomorrow morning",
]


class TestTFIDFExtractor:
    def test_fit_transform_shape(self):
        ext = TFIDFExtractor(max_features=50)
        X = ext.fit_transform(SAMPLE_TEXTS)
        assert X.shape[0] == len(SAMPLE_TEXTS)
        assert X.shape[1] <= 50

    def test_transform_same_cols_as_fit(self):
        ext = TFIDFExtractor(max_features=50)
        ext.fit(SAMPLE_TEXTS)
        X_train = ext.transform(SAMPLE_TEXTS)
        X_test  = ext.transform(["win free money"])
        assert X_train.shape[1] == X_test.shape[1]

    def test_transform_before_fit_raises(self):
        ext = TFIDFExtractor()
        with pytest.raises(ValueError):
            ext.transform(SAMPLE_TEXTS)

    def test_get_feature_names_returns_array(self):
        ext = TFIDFExtractor(max_features=20)
        ext.fit(SAMPLE_TEXTS)
        names = ext.get_feature_names()
        assert len(names) > 0

    def test_save_load_roundtrip(self, tmp_path):
        ext = TFIDFExtractor(max_features=20)
        X_before = ext.fit_transform(SAMPLE_TEXTS)
        path = str(tmp_path / "tfidf.pkl")
        ext.save(path)

        ext2 = TFIDFExtractor(max_features=20)
        ext2.load(path)
        X_after = ext2.transform(SAMPLE_TEXTS)
        np.testing.assert_array_almost_equal(X_before.toarray(), X_after.toarray())


class TestExtractManualFeatures:
    def test_output_shape(self):
        feats = extract_manual_features(SAMPLE_TEXTS)
        assert feats.shape == (len(SAMPLE_TEXTS), 6)

    def test_length_feature(self):
        texts = ["hi", "hello world"]
        feats = extract_manual_features(texts)
        assert feats[0, 0] < feats[1, 0]  # shorter text → smaller length

    def test_empty_string(self):
        feats = extract_manual_features([""])
        assert feats.shape == (1, 6)
        assert feats[0, 0] == 0  # length of empty string
