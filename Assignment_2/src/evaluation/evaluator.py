"""Comprehensive evaluation engine for Assignment 2 retrieval and Subtask 2.

Supports:
- Evaluation of ranked retrieval lists (MAP, MRR, nDCG@k, P@k, Recall@k).
- Evaluation of binary sense predictions against official Subtask 2 gold labels.
- Operating point optimization (rank cutoff tuning) on ranked predictions.
- Subgroup and robustness evaluation across words and historical time periods.
- Integration with Assignment 1 baselines for unified benchmarking.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from .metrics import (
    compute_classification_metrics,
    compute_retrieval_metrics,
)


def _to_dataframe(data: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")
    return pd.read_csv(path)


def load_qrels(qrels_input: Union[pd.DataFrame, str, Path]) -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
    """Load qrels and build a mapping of query_id -> set of relevant doc_ids."""
    df = _to_dataframe(qrels_input)
    required = {"query_id", "doc_id", "relevance"}
    if not required.issubset(df.columns):
        raise ValueError(f"Qrels must contain columns {required}, got {list(df.columns)}")

    df["query_id"] = df["query_id"].astype(str)
    df["doc_id"] = df["doc_id"].astype(str)
    df["relevance"] = df["relevance"].astype(int)

    positives = df[df["relevance"] > 0]
    qrels_map: Dict[str, Set[str]] = {}
    for qid, group in positives.groupby("query_id"):
        qrels_map[str(qid)] = set(group["doc_id"].tolist())

    # Ensure all queries in qrels have an entry (even if empty)
    for qid in df["query_id"].unique():
        qrels_map.setdefault(str(qid), set())

    return df, qrels_map


def evaluate_retrieval_run(
    ranked_input: Union[pd.DataFrame, str, Path],
    qrels_input: Union[pd.DataFrame, str, Path],
    cutoffs: Sequence[int] = (5, 10, 20, 50, 100),
) -> Dict[str, float]:
    """Evaluate a ranked prediction CSV (query_id, doc_id, rank, score)."""
    ranked_df = _to_dataframe(ranked_input)
    _, qrels_map = load_qrels(qrels_input)

    required = {"query_id", "doc_id", "rank"}
    if not required.issubset(ranked_df.columns):
        raise ValueError(f"Ranked predictions require {required}, got {list(ranked_df.columns)}")

    ranked_df["query_id"] = ranked_df["query_id"].astype(str)
    ranked_df["doc_id"] = ranked_df["doc_id"].astype(str)
    ranked_df = ranked_df.sort_values(["query_id", "rank"], ascending=[True, True])

    ranked_by_query: Dict[str, List[str]] = {}
    for qid, group in ranked_df.groupby("query_id"):
        ranked_by_query[str(qid)] = group["doc_id"].tolist()

    return compute_retrieval_metrics(ranked_by_query, qrels_map, cutoffs=cutoffs)


def evaluate_binary_run(
    binary_input: Union[pd.DataFrame, str, Path],
    qrels_input: Union[pd.DataFrame, str, Path],
) -> Dict[str, float]:
    """Evaluate binary classification predictions against Subtask 2 qrels."""
    binary_df = _to_dataframe(binary_input)
    qrels_df, _ = load_qrels(qrels_input)

    # Standardize column names (binary_predictions.csv uses sentence_id or doc_id)
    if "sentence_id" in binary_df.columns and "doc_id" not in binary_df.columns:
        binary_df["doc_id"] = binary_df["sentence_id"].astype(str)

    # Join on (query_id, doc_id)
    binary_df["query_id"] = binary_df["query_id"].astype(str)
    binary_df["doc_id"] = binary_df["doc_id"].astype(str)
    binary_df["label"] = binary_df["label"].astype(int)

    merged = pd.merge(
        qrels_df,
        binary_df[["query_id", "doc_id", "label"] + (["score"] if "score" in binary_df.columns else [])],
        on=["query_id", "doc_id"],
        how="inner",
    )

    if len(merged) == 0:
        raise ValueError("No matching (query_id, doc_id) pairs found between binary predictions and qrels")

    y_score = merged["score"] if "score" in merged.columns else None
    return compute_classification_metrics(
        y_true=merged["relevance"],
        y_pred=merged["label"],
        y_score=y_score,
    )


def tune_operating_point(
    ranked_input: Union[pd.DataFrame, str, Path],
    qrels_input: Union[pd.DataFrame, str, Path],
    k_range: Sequence[int] = range(5, 105, 5),
) -> Tuple[int, float, pd.DataFrame, pd.DataFrame]:
    """Sweep rank cutoff K to maximize official Macro-F1 on the candidate pools.

    Returns:
    - best_k: optimal rank cutoff
    - best_macro_f1: maximum Macro-F1 achieved
    - tuned_binary_df: DataFrame of binary predictions with label = 1 iff rank <= best_k
    - sweep_results_df: DataFrame showing (k, macro_f1, binary_f1, accuracy, precision, recall)
    """
    ranked_df = _to_dataframe(ranked_input)
    qrels_df, _ = load_qrels(qrels_input)

    ranked_df["query_id"] = ranked_df["query_id"].astype(str)
    ranked_df["doc_id"] = ranked_df["doc_id"].astype(str)

    # All pairs in qrels form the evaluation universe
    eval_universe = qrels_df[["query_id", "doc_id", "relevance"]].copy()

    records: List[Dict[str, Any]] = []
    best_k = 50
    best_macro_f1 = -1.0

    for k in k_range:
        # Determine hits within top-k
        top_k_hits = ranked_df[ranked_df["rank"] <= k][["query_id", "doc_id"]].copy()
        top_k_hits["predicted_label"] = 1

        merged = pd.merge(
            eval_universe,
            top_k_hits,
            on=["query_id", "doc_id"],
            how="left",
        )
        merged["predicted_label"] = merged["predicted_label"].fillna(0).astype(int)

        metrics = compute_classification_metrics(
            y_true=merged["relevance"],
            y_pred=merged["predicted_label"],
        )
        metrics["k"] = k
        records.append(metrics)

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_k = k

    sweep_df = pd.DataFrame(records).sort_values("k")

    # Generate tuned binary predictions at best_k
    top_best = ranked_df[ranked_df["rank"] <= best_k][["query_id", "doc_id", "rank", "score"]].copy()
    top_best["label"] = 1

    tuned_binary = pd.merge(
        eval_universe[["query_id", "doc_id"]],
        top_best,
        on=["query_id", "doc_id"],
        how="left",
    )
    tuned_binary["label"] = tuned_binary["label"].fillna(0).astype(int)
    tuned_binary["sentence_id"] = tuned_binary["doc_id"]
    tuned_binary["word"] = tuned_binary["query_id"]

    return best_k, best_macro_f1, tuned_binary, sweep_df


def subgroup_evaluation(
    binary_input: Union[pd.DataFrame, str, Path],
    qrels_input: Union[pd.DataFrame, str, Path],
    documents_metadata: Union[pd.DataFrame, Sequence[Any]],
    group_by: str = "word",
) -> pd.DataFrame:
    """Compute performance breakdown across metadata groups (e.g., target word or time period)."""
    binary_df = _to_dataframe(binary_input)
    qrels_df, _ = load_qrels(qrels_input)

    if isinstance(documents_metadata, pd.DataFrame):
        meta_df = documents_metadata.copy()
    else:
        # Sequence of Document objects from data_loader
        meta_df = pd.DataFrame([
            {
                "doc_id": str(d.doc_id),
                "word": str(d.word),
                "period_label": str(d.period_label),
                "year": str(d.year),
            }
            for d in documents_metadata
        ])

    if "sentence_id" in binary_df.columns and "doc_id" not in binary_df.columns:
        binary_df["doc_id"] = binary_df["sentence_id"].astype(str)

    binary_df["doc_id"] = binary_df["doc_id"].astype(str)
    binary_df["query_id"] = binary_df["query_id"].astype(str)
    meta_df["doc_id"] = meta_df["doc_id"].astype(str)

    merged = pd.merge(qrels_df, binary_df[["query_id", "doc_id", "label"]], on=["query_id", "doc_id"])
    merged = pd.merge(merged, meta_df[["doc_id", "word", "period_label", "year"]], on="doc_id")

    if group_by not in merged.columns:
        raise ValueError(f"group_by column '{group_by}' not found in metadata: {list(merged.columns)}")

    rows = []
    for group_val, grp in merged.groupby(group_by):
        m = compute_classification_metrics(grp["relevance"], grp["label"])
        rows.append({
            group_by: group_val,
            "instances": len(grp),
            "positives": int(grp["relevance"].sum()),
            "pos_ratio": float(grp["relevance"].mean()),
            "macro_f1": m["macro_f1"],
            "binary_f1": m["binary_f1"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
        })

    return pd.DataFrame(rows).sort_values("instances", ascending=False)

