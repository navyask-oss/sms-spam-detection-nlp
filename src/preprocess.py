import re
import string
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords


def _ensure_nltk_resources() -> None:
    """Download NLTK resources once if they are unavailable."""
    resources = {
        "corpora/stopwords": "stopwords",
    }
    for lookup_key, download_name in resources.items():
        try:
            nltk.data.find(lookup_key)
        except LookupError:
            nltk.download(download_name, quiet=True)


def clean_text(text: str) -> str:
    """Normalize SMS text for vectorization."""
    _ensure_nltk_resources()
    stop_words = set(stopwords.words("english"))

    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)

    tokens = [token for token in text.split() if token not in stop_words]
    return " ".join(tokens).strip()


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load and standardize dataset columns to: label, message."""
    df = pd.read_csv(csv_path)

    if {"label", "message"}.issubset(df.columns):
        normalized = df[["label", "message"]].copy()
    elif {"v1", "v2"}.issubset(df.columns):
        normalized = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
    else:
        raise ValueError("Dataset must include either [label, message] or [v1, v2] columns.")

    normalized["label"] = normalized["label"].astype(str).str.strip().str.lower()
    normalized["message"] = normalized["message"].astype(str)
    normalized = normalized.dropna().drop_duplicates()
    normalized["clean_message"] = normalized["message"].apply(clean_text)
    return normalized
