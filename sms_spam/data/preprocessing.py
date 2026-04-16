"""
Data Preprocessing Module for SMS Spam Detection

Provides functions for text cleaning and preprocessing:
- Text cleaning (removing special characters, URLs, etc.)
- Tokenization
- Stop word removal
- Lemmatization
- Complete preprocessing pipeline
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def download_nltk_data():
    """Download required NLTK datasets."""
    for package in ["punkt", "stopwords", "wordnet", "punkt_tab"]:
        try:
            nltk.download(package, quiet=True)
        except Exception:
            pass


# Initialize NLTK resources at import time
download_nltk_data()


def load_data(filepath):
    """
    Load SMS spam dataset from CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least ``label`` and ``message`` columns.
    """
    encodings = ["utf-8", "latin-1", "iso-8859-1"]
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError(f"Could not decode file: {filepath}")

    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]]
        df.columns = ["label", "message"]
    elif "label" not in df.columns:
        df.columns = ["label", "message"] + list(df.columns[2:])
        df = df[["label", "message"]]

    df["label_encoded"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def clean_text(text):
    """
    Clean text by removing unwanted characters and patterns.

    Parameters
    ----------
    text : str
        Raw text to clean.

    Returns
    -------
    str
        Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\b\d{10,}\b", "", text)
    text = re.sub(r"\b\d{3}[-.\\s]?\d{3}[-.\\s]?\d{4}\b", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text):
    """Tokenize text into words."""
    try:
        return word_tokenize(text)
    except Exception:
        return text.split()


def remove_stopwords(tokens):
    """Remove English stopwords from token list."""
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves", "he", "him",
            "his", "himself", "she", "her", "hers", "herself", "it", "its",
            "itself", "they", "them", "their", "theirs", "themselves",
            "what", "which", "who", "whom", "this", "that", "these", "those",
            "am", "is", "are", "was", "were", "be", "been", "being", "have",
            "has", "had", "having", "do", "does", "did", "doing", "a", "an",
            "the", "and", "but", "if", "or", "because", "as", "until",
            "while", "of", "at", "by", "for", "with", "about", "against",
            "between", "into", "through", "during", "before", "after",
            "above", "below", "to", "from", "up", "down", "in", "out", "on",
            "off", "over", "under", "again", "further", "then", "once",
        }
    return [token for token in tokens if token not in stop_words]


def lemmatize_tokens(tokens):
    """Lemmatize tokens to their base form."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def preprocess_text(text):
    """
    Apply complete preprocessing pipeline to a single text string.

    Parameters
    ----------
    text : str
        Raw message text.

    Returns
    -------
    str
        Preprocessed text.
    """
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


def preprocess_pipeline(df, text_column="message"):
    """
    Apply preprocessing to an entire DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the raw text column.
    text_column : str
        Name of the column to preprocess.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional ``processed_text``, ``original_length``,
        ``processed_length``, and ``word_count`` columns.
    """
    print(f"Starting preprocessing pipeline on {len(df)} messages...")
    df = df.copy()
    df["processed_text"] = df[text_column].apply(preprocess_text)
    df["original_length"] = df[text_column].apply(len)
    df["processed_length"] = df["processed_text"].apply(len)
    df["word_count"] = df["processed_text"].apply(lambda x: len(x.split()))
    print("Preprocessing complete!")
    print(f"  Average original length : {df['original_length'].mean():.2f}")
    print(f"  Average processed length: {df['processed_length'].mean():.2f}")
    return df


def demonstrate_preprocessing(text):
    """Demonstrate each preprocessing step with intermediate outputs."""
    print("=" * 60)
    print("PREPROCESSING DEMONSTRATION")
    print("=" * 60)
    print(f"\n1. ORIGINAL TEXT:\n   {repr(text)}")
    cleaned = clean_text(text)
    print(f"\n2. AFTER CLEANING:\n   {repr(cleaned)}")
    tokens = tokenize_text(cleaned)
    print(f"\n3. AFTER TOKENIZATION:\n   {tokens}")
    no_stopwords = remove_stopwords(tokens)
    print(f"\n4. AFTER STOPWORD REMOVAL:\n   {no_stopwords}")
    lemmatized = lemmatize_tokens(no_stopwords)
    print(f"\n5. AFTER LEMMATIZATION:\n   {lemmatized}")
    final = " ".join(lemmatized)
    print(f"\n6. FINAL RESULT:\n   {repr(final)}")
    print("=" * 60)
    return final


if __name__ == "__main__":
    sample = "WINNER!! You have won a FREE prize! Call 08001234567 NOW!"
    demonstrate_preprocessing(sample)
