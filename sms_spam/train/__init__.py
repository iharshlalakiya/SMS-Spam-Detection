"""
sms_spam.train — SMS Spam Detection Package

Public API
----------
    SpamDetector       : SVM-based spam classifier
    TFIDFExtractor     : TF-IDF feature extraction wrapper
    preprocess_text    : Single-message preprocessing function
    preprocess_pipeline: Full-DataFrame preprocessing pipeline
    calculate_metrics  : Evaluation metric helper
"""

from sms_spam.data.preprocessing import preprocess_text, preprocess_pipeline
from sms_spam.features.extraction import TFIDFExtractor
from sms_spam.models.svm import SpamDetector
from sms_spam.evaluation.metrics import calculate_metrics, print_metrics
from sms_spam.train.train import step_preprocess, step_split, step_features, step_train

__version__ = "1.0.0"
__author__ = "Harsh"
