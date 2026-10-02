from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value)


def _word_candidates(word: str) -> set[str]:
    w = str(word).lower()
    candidates = {w}
    if len(w) <= 2:
        return candidates
    for n in range(1, min(5, len(w))):
        candidates.add(w[:-n])
    suffixes = [
        "a",
        "an",
        "en",
        "ar",
        "as",
        "er",
        "or",
        "na",
        "orna",
        "arna",
        "ers",
        "s",
    ]
    for suffix in suffixes:
        if w.endswith(suffix):
            candidates.add(w[: -len(suffix)])
    if w.endswith("a"):
        candidates.add(w[:-1] + "an")
    if w.endswith("en"):
        candidates.add(w[:-2])
    if w.endswith("er"):
        candidates.add(w[:-2])
    return {c for c in candidates if c}


def _classify_offset_quality(row: pd.Series) -> str:
    text = _clean_text(row.get("text"))
    word = str(row.get("word", "")).lower()
    start = row.get("start")
    end = row.get("end")
    if start is None or end is None:
        return "suspicious/mismatch"
    try:
        start_i = int(start)
        end_i = int(end)
    except (TypeError, ValueError):
        return "suspicious/mismatch"
    if start_i < 0 or end_i <= start_i or end_i > len(text):
        return "suspicious/mismatch"
    segment = text[start_i:end_i].lower()
    if segment == word:
        return "exact target match"

    tokens = re.findall(r"[a-zA-ZåäöÅÄÖ]+(?:[-'][a-zA-ZåäöÅÄÖ]+)?", text.lower())
    seq = [t for t in tokens if len(t) > 1]
    normalized = {t.lower() for t in seq}
    stem_forms = _word_candidates(word)
    if segment in normalized or word in normalized:
        return "exact target match"

    if any(token in stem_forms for token in normalized):
        return "plausible inflection/derivation"

    shared_stem = False
    for token in normalized:
        if len(token) < 4:
            continue
        if token.startswith(word[:3]) or word.startswith(token[:3]):
            shared_stem = True
        if any(token.startswith(st) and len(st) >= 3 for st in stem_forms):
            shared_stem = True
    if shared_stem:
        return "shared-stem candidate"
    return "suspicious/mismatch"


def compute_offset_quality(usage_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    usage_df = usage_df.copy()
    usage_df["offset_quality"] = usage_df.apply(_classify_offset_quality, axis=1)
    quality_summary = (
        usage_df["offset_quality"]
        .value_counts()
        .rename_axis("offset_quality")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return usage_df, quality_summary


def flatten_subtask_labels(df: pd.DataFrame, target_col: str = "label") -> pd.Series:
    values: List[Any] = []
    if df.empty:
        return pd.Series(dtype="int64")
    for entry in df[target_col].tolist():
        if entry is None:
            continue
        if isinstance(entry, list):
            values.extend(entry)
        elif isinstance(entry, tuple):
            values.extend(list(entry))
        elif isinstance(entry, str):
            try:
                values.extend([int(v) for v in entry.strip("[]").split(",") if v.strip()])
            except ValueError:
                values.append(entry)
        else:
            values.append(entry)
    return pd.Series(values)


def compute_dataset_statistics(
    usage_df: pd.DataFrame,
    subtask1_df: pd.DataFrame,
    subtask2_df: pd.DataFrame,
    definitions_df: pd.DataFrame,
) -> Dict[str, Any]:
    usage_df = usage_df.copy()
    usage_df["text_length_chars"] = usage_df["text"].fillna("").str.len()
    usage_df["token_count"] = usage_df["text"].fillna("").str.split().str.len()

    dataset_summary = {
        "usage_records": int(len(usage_df)),
        "unique_target_words": int(usage_df["word"].nunique(dropna=True)),
        "unique_sentence_ids": int(usage_df["sentence_id"].nunique(dropna=True)),
        "unique_periods": int(usage_df["period_label"].nunique(dropna=True)),
        "min_year": int(usage_df["year_int"].min()) if usage_df["year_int"].notna().any() else None,
        "max_year": int(usage_df["year_int"].max()) if usage_df["year_int"].notna().any() else None,
        "pos_categories": int(usage_df["pos"].nunique(dropna=True)),
        "empty_text": int((usage_df["text"].fillna("") == "").sum()),
        "duplicate_records": int(usage_df.duplicated().sum()),
        "duplicate_sentence_ids": int(usage_df["sentence_id"].duplicated(keep=False).sum()),
        "exact_duplicate_texts": int(usage_df["text"].duplicated(keep=False).sum()),
        "missing_values": usage_df.isna().sum().to_dict(),
        "invalid_years": int(usage_df["year_int"].isna().sum()),
        "invalid_period_labels": int((~usage_df["period_label"].astype(str).str.fullmatch(r"\d{4}-\d{4}", na=False)).sum()),
        "word_count_per_target": usage_df.groupby("word").size().to_dict(),
        "period_count_per_label": usage_df.groupby("period_label").size().to_dict(),
        "subtask1_sense_counts": flatten_subtask_labels(subtask1_df).value_counts().sort_index().to_dict(),
        "subtask2_label_counts": flatten_subtask_labels(subtask2_df).value_counts().sort_index().to_dict(),
        "definitions_count": int(len(definitions_df)),
    }

    quality_usage_df, quality_summary = compute_offset_quality(usage_df)
    dataset_summary["offset_quality"] = quality_summary.set_index("offset_quality")["count"].to_dict()

    word_stats = (
        usage_df.groupby("word")
        .size()
        .rename("usage_count")
        .reset_index()
        .sort_values("usage_count", ascending=False)
    )
    word_stats["min_count"] = word_stats["usage_count"]
    word_stats["max_count"] = word_stats["usage_count"]
    word_stats["mean_count"] = word_stats["usage_count"].mean()
    word_stats["median_count"] = word_stats["usage_count"].median()
    word_stats["std_count"] = word_stats["usage_count"].std(ddof=0)
    word_stats["imbalance_ratio"] = word_stats["usage_count"].max() / max(word_stats["usage_count"].min(), 1)

    period_counts = (
        usage_df.groupby("period_label")
        .agg(
            usage_count=("sentence_id", "count"),
            unique_target_words=("word", "nunique"),
        )
        .reset_index()
        .sort_values("period_label")
    )

    text_stats = usage_df["text_length_chars"].describe().to_dict()
    token_stats = usage_df["token_count"].describe().to_dict()
    quality_summary = quality_summary.rename(columns={"offset_quality": "quality_label"})

    return {
        "summary": dataset_summary,
        "usage_df": quality_usage_df,
        "word_statistics": word_stats,
        "period_statistics": period_counts,
        "text_describe": text_stats,
        "token_describe": token_stats,
        "offset_quality": quality_summary,
    }


def write_dataset_tables(output_dir: Path, stats: Dict[str, Any]) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    pd.DataFrame([stats["summary"]]).to_csv(output_dir / "dataset_statistics.csv", index=False)
    paths["dataset_statistics"] = str(output_dir / "dataset_statistics.csv")
    stats["offset_quality"].to_csv(output_dir / "data_quality.csv", index=False)
    paths["data_quality"] = str(output_dir / "data_quality.csv")
    stats["word_statistics"].to_csv(output_dir / "word_statistics.csv", index=False)
    paths["word_statistics"] = str(output_dir / "word_statistics.csv")
    stats["period_statistics"].to_csv(output_dir / "period_statistics.csv", index=False)
    paths["period_statistics"] = str(output_dir / "period_statistics.csv")
    return paths
