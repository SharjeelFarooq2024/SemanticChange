from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.eda.text_features import add_text_features


DEFAULT_NUMERIC_FEATURES = [
    "text_length_chars", "token_count", "unique_token_count", "type_token_ratio",
    "avg_token_length", "punctuation_ratio", "digit_ratio", "stopword_ratio",
    "target_position",
]
DEFAULT_CAT_FEATURES = ["word", "pos", "period_label"]
DEFAULT_ABLATED_CAT_FEATURES = ["pos", "period_label"]

# Backward-compatibility aliases
NUMERIC_FEATURES = DEFAULT_NUMERIC_FEATURES
CAT_FEATURES = DEFAULT_CAT_FEATURES


def get_numeric_features(config: dict | None = None) -> list[str]:
    if config and "classical" in config and "numeric_features" in config["classical"]:
        return list(config["classical"]["numeric_features"])
    return list(DEFAULT_NUMERIC_FEATURES)


def get_categorical_features(config: dict | None = None) -> list[str]:
    if config and "classical" in config and "categorical_features" in config["classical"]:
        return list(config["classical"]["categorical_features"])
    return list(DEFAULT_CAT_FEATURES)


def get_ablated_categorical_features(config: dict | None = None) -> list[str]:
    if config and "classical" in config and "ablated_categorical_features" in config["classical"]:
        return list(config["classical"]["ablated_categorical_features"])
    return list(DEFAULT_ABLATED_CAT_FEATURES)


def build_feature_frame(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Add EDA features while preserving the original usage rows."""
    cat_cols = get_categorical_features(config)
    required = {"text"}.union(cat_cols)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Cannot build baseline features; missing columns: " + ", ".join(missing))
    return add_text_features(df).fillna({col: "" for col in cat_cols})


def build_interpretable_preprocessor(config: dict) -> ColumnTransformer:
    classical = config["classical"]
    num_cols = get_numeric_features(config)
    cat_cols = get_categorical_features(config)
    text = TfidfVectorizer(
        ngram_range=tuple(classical["word_ngram_range"]),
        min_df=classical["min_df"],
        max_features=classical["max_features"],
        sublinear_tf=True,
    )
    return ColumnTransformer(
        transformers=[
            ("text", text, "text"),
            ("numeric", StandardScaler(), num_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.get_feature_names_out())


def build_ablated_preprocessor(config: dict) -> ColumnTransformer:
    """Build a preprocessor that ablates target word identity.

    Excludes the categorical 'word' column so downstream models cannot exploit
    target-word label imbalances, isolating pure contextual and EDA features.
    """
    classical = config["classical"]
    num_cols = get_numeric_features(config)
    ablated_cat = get_ablated_categorical_features(config)
    text = TfidfVectorizer(
        ngram_range=tuple(classical["word_ngram_range"]),
        min_df=classical["min_df"],
        max_features=classical["max_features"],
        sublinear_tf=True,
    )
    return ColumnTransformer(
        transformers=[
            ("text", text, "text"),
            ("numeric", StandardScaler(), num_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ablated_cat),
        ],
        remainder="drop",
    )