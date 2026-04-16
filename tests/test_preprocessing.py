"""Smoke tests for sms_spam.data.preprocessing."""

import pytest
from sms_spam.data.preprocessing import clean_text, preprocess_text, preprocess_pipeline
import pandas as pd


def test_clean_text_lowercases():
    assert clean_text("HELLO WORLD") == "hello world"


def test_clean_text_removes_url():
    result = clean_text("visit http://spam.com now")
    assert "http" not in result
    assert "spam" not in result or "com" not in result


def test_clean_text_removes_email():
    result = clean_text("email me at foo@bar.com")
    assert "@" not in result


def test_clean_text_empty_string():
    assert clean_text("") == ""


def test_clean_text_non_string():
    assert clean_text(None) == ""
    assert clean_text(42) == ""


def test_preprocess_text_returns_string():
    result = preprocess_text("Free prize! Call now 08001234567")
    assert isinstance(result, str)


def test_preprocess_text_removes_stopwords():
    result = preprocess_text("this is a test message")
    # Common stopwords shouldn't dominate the output
    assert "this" not in result.split()


def test_preprocess_pipeline_adds_columns():
    df = pd.DataFrame({"label": ["ham", "spam"], "message": ["Hello there", "Free prize!"]})
    out = preprocess_pipeline(df, text_column="message")
    for col in ["processed_text", "original_length", "processed_length", "word_count"]:
        assert col in out.columns


def test_preprocess_pipeline_length_columns():
    df = pd.DataFrame({"label": ["ham"], "message": ["Hello world"]})
    out = preprocess_pipeline(df, text_column="message")
    assert out["original_length"].iloc[0] == len("Hello world")
