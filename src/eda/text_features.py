from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


_SWEDISH_STOPWORDS = {
    "och", "det", "att", "i", "en", "som", "av", "för", "med", "på", "den",
    "de", "var", "från", "om", "är", "inte", "ett", "har", "sig", "till",
    "med", "utan", "under", "efter", "genom", "eller", "bland", "man", "hon",
}


def _tokenize(text: str) -> list[str]:
    if text is None:
        return []
    return re.findall(r"[A-Za-zÅÄÖåäö]+(?:[-'][A-Za-zÅÄÖåäö]+)?", text)


def _count_punctuation(text: str) -> float:
    if not text:
        return 0.0
    punct = sum(1 for ch in text if ch in ".,;:!?\"'()[]{}-—–")
    return punct / max(len(text), 1)


def _count_digits(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(ch.isdigit() for ch in text)
    return digits / max(len(text), 1)


def _stopword_ratio(text: str) -> float:
    tokens = [t.lower() for t in _tokenize(text)]
    if not tokens:
        return 0.0
    stop_count = sum(1 for t in tokens if t in _SWEDISH_STOPWORDS)
    return stop_count / len(tokens)


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_length_chars"] = df["text"].fillna("").str.len()
    df["token_count"] = df["text"].fillna("").str.split().str.len()
    df["unique_token_count"] = df["text"].fillna("").apply(lambda x: len(set(_tokenize(x.lower()))))
    df["type_token_ratio"] = df.apply(
        lambda row: (len(set(_tokenize(row["text"].lower()))) / max(len(_tokenize(row["text"])), 1)),
        axis=1,
    )
    df["avg_token_length"] = df["text"].fillna("").apply(
        lambda x: np.mean([len(t) for t in _tokenize(x)]) if _tokenize(x) else 0.0
    )
    df["punctuation_ratio"] = df["text"].fillna("").apply(_count_punctuation)
    df["digit_ratio"] = df["text"].fillna("").apply(_count_digits)
    df["stopword_ratio"] = df["text"].fillna("").apply(_stopword_ratio)
    df["lexical_diversity"] = df["type_token_ratio"]
    df["target_in_text"] = df.apply(
        lambda row: str(row["word"]).lower() in row["text"].lower(),
        axis=1,
    )
    df["target_position"] = df.apply(
        lambda row: row["text"].lower().find(str(row["word"]).lower()) if row["word"] else -1,
        axis=1,
    )
    return df
