from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts, ignoring blank lines."""
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_usage_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "word",
        "sentence_id",
        "text",
        "pos",
        "period_label",
        "start",
        "end",
        "year",
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA

    df = df.copy()
    df["word"] = df["word"].map(_safe_str)
    df["sentence_id"] = df["sentence_id"].map(_safe_str)
    df["text"] = df["text"].map(lambda x: str(x) if x is not None else "")
    df["pos"] = df["pos"].map(_safe_str)
    df["period_label"] = df["period_label"].map(_safe_str)
    df["start"] = df["start"].map(lambda x: _as_int(x, None))
    df["end"] = df["end"].map(lambda x: _as_int(x, None))
    df["year"] = df["year"].map(_safe_str)
    df["year_int"] = df["year"].map(lambda x: _as_int(x, None))
    df["usage_id"] = df["sentence_id"].fillna("") + "|" + df["word"].fillna("") + "|" + df["year"].fillna("")
    return df


def _normalize_definition_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "word" not in df.columns:
        df["word"] = ""
    if "definition" not in df.columns:
        df["definition"] = ""
    df["word"] = df["word"].map(_safe_str)
    df["definition"] = df["definition"].map(lambda x: "" if x is None else str(x))
    return df


def _normalize_subtask_frame(df: pd.DataFrame, target_label: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "word" not in df.columns:
        df["word"] = ""
    if "sentence_id" not in df.columns:
        df["sentence_id"] = ""
    if target_label not in df.columns:
        df[target_label] = pd.NA
    df["word"] = df["word"].map(_safe_str)
    df["sentence_id"] = df["sentence_id"].map(_safe_str)
    df[target_label] = df[target_label].map(lambda x: x if x is not None else [])
    return df


def load_usage_data(project_root: Union[str, Path]) -> pd.DataFrame:
    project_root = Path(project_root)
    path = project_root / "data" / "raw" / "dev" / "SV" / "usages.jsonl"
    rows = load_jsonl(path)
    df = pd.DataFrame(rows)
    return _normalize_usage_frame(df)


def load_subtask1_data(project_root: Union[str, Path]) -> pd.DataFrame:
    project_root = Path(project_root)
    path = project_root / "data" / "raw" / "dev" / "SV" / "subtask1.jsonl"
    rows = load_jsonl(path)
    df = pd.DataFrame(rows)
    return _normalize_subtask_frame(df, "label")


def load_subtask2_data(project_root: Union[str, Path]) -> pd.DataFrame:
    project_root = Path(project_root)
    path = project_root / "data" / "raw" / "dev" / "SV" / "subtask2.jsonl"
    rows = load_jsonl(path)
    df = pd.DataFrame(rows)
    return _normalize_subtask_frame(df, "label")


def load_definitions(project_root: Union[str, Path]) -> pd.DataFrame:
    project_root = Path(project_root)
    candidate_paths = [
        project_root / "data" / "raw" / "dev" / "SV" / "definitions_tot.jsonl",
        project_root / "data" / "raw" / "dev" / "SV" / "definitions.jsonl",
    ]
    for path in candidate_paths:
        if path.exists():
            rows = load_jsonl(path)
            return _normalize_definition_frame(pd.DataFrame(rows))
    return pd.DataFrame(columns=["word", "definition"])


def build_dataset_bundle(project_root: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    project_root = Path(project_root)
    bundle = {
        "usage": load_usage_data(project_root),
        "subtask1": load_subtask1_data(project_root),
        "subtask2": load_subtask2_data(project_root),
        "definitions": load_definitions(project_root),
    }
    return bundle
