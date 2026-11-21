"""Evaluation metrics for information retrieval and binary sense matching.

Implements official and secondary metrics specified in the Assignment 2 design:
- Retrieval metrics: P@K, Recall@K, MRR, MAP, nDCG@K.
- Classification metrics: Official Subtask 2 Macro-F1, binary F1, precision,
  recall, accuracy, balanced accuracy.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Set, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# --------------------------------------------------------------------------- #
# Information Retrieval Ranking Metrics (Per Query)
# --------------------------------------------------------------------------- #
def precision_at_k(ranked_doc_ids: Sequence[str], relevant_doc_ids: Set[str], k: int) -> float:
    """Compute Precision at rank cutoff K.

    P@K = |Retrieved_K ∩ Relevant| / K
    """
    if k <= 0:
        return 0.0
    cutoff_docs = ranked_doc_ids[:k]
    if not cutoff_docs:
        return 0.0
    hits = sum(1 for doc_id in cutoff_docs if doc_id in relevant_doc_ids)
    return hits / float(k)


def recall_at_k(ranked_doc_ids: Sequence[str], relevant_doc_ids: Set[str], k: int) -> float:
    """Compute Recall at rank cutoff K.

    Recall@K = |Retrieved_K ∩ Relevant| / |Relevant|
    """
    if not relevant_doc_ids:
        return 0.0
    if k <= 0:
        return 0.0
    cutoff_docs = ranked_doc_ids[:k]
    hits = sum(1 for doc_id in cutoff_docs if doc_id in relevant_doc_ids)
    return hits / float(len(relevant_doc_ids))


def reciprocal_rank(ranked_doc_ids: Sequence[str], relevant_doc_ids: Set[str]) -> float:
    """Compute Reciprocal Rank (RR) for a single query.

    RR = 1 / (rank of first relevant document), or 0.0 if none found.
    Rank is 1-indexed.
    """
    if not relevant_doc_ids:
        return 0.0
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / float(rank)
    return 0.0


def average_precision(ranked_doc_ids: Sequence[str], relevant_doc_ids: Set[str]) -> float:
    """Compute Average Precision (AP) for a single query.

    AP = (1 / |Relevant|) * sum_{k=1}^N (P@k * I(doc_k is relevant))
    """
    if not relevant_doc_ids:
        return 0.0

    hit_count = 0
    precisions_sum = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            hit_count += 1
            precisions_sum += hit_count / float(rank)

    return precisions_sum / float(len(relevant_doc_ids))


def ndcg_at_k(ranked_doc_ids: Sequence[str], relevant_doc_ids: Set[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank cutoff K.

    DCG@K = sum_{i=1}^K (rel_i / log2(i + 1)) with binary rel_i in {0, 1}.
    nDCG@K = DCG@K / IDCG@K
    """
    if not relevant_doc_ids or k <= 0:
        return 0.0

    cutoff_docs = ranked_doc_ids[:k]
    dcg = 0.0
    for i, doc_id in enumerate(cutoff_docs, start=1):
        if doc_id in relevant_doc_ids:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG: all relevant documents placed at top ranks up to min(k, |rel|)
    ideal_hits = min(k, len(relevant_doc_ids))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


# --------------------------------------------------------------------------- #
# Aggregated Retrieval Metrics (Across All Queries)
# --------------------------------------------------------------------------- #
def compute_retrieval_metrics(
    ranked_by_query: Dict[str, Sequence[str]],
    qrels_by_query: Dict[str, Set[str]],
    cutoffs: Sequence[int] = (5, 10, 20, 50, 100),
) -> Dict[str, float]:
    """Compute mean retrieval metrics across all queries.

    Returns a dictionary of metric names mapped to macro-averaged scores.
    """
    query_ids = sorted(qrels_by_query.keys())
    if not query_ids:
        return {}

    per_query_metrics: Dict[str, List[float]] = {
        "mrr": [],
        "map": [],
    }
    for k in cutoffs:
        per_query_metrics[f"p@{k}"] = []
        per_query_metrics[f"recall@{k}"] = []
        per_query_metrics[f"ndcg@{k}"] = []

    for qid in query_ids:
        ranked_docs = ranked_by_query.get(qid, [])
        relevant_docs = qrels_by_query.get(qid, set())

        per_query_metrics["mrr"].append(reciprocal_rank(ranked_docs, relevant_docs))
        per_query_metrics["map"].append(average_precision(ranked_docs, relevant_docs))

        for k in cutoffs:
            per_query_metrics[f"p@{k}"].append(precision_at_k(ranked_docs, relevant_docs, k))
            per_query_metrics[f"recall@{k}"].append(recall_at_k(ranked_docs, relevant_docs, k))
            per_query_metrics[f"ndcg@{k}"].append(ndcg_at_k(ranked_docs, relevant_docs, k))

    summary: Dict[str, float] = {}
    for metric_name, values in per_query_metrics.items():
        summary[metric_name] = float(np.mean(values)) if values else 0.0

    return summary


# --------------------------------------------------------------------------- #
# Classification Metrics (pooled secondary metrics and optional official grouping)
# --------------------------------------------------------------------------- #
def compute_classification_metrics(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    y_score: Union[Sequence[float], np.ndarray, pd.Series, None] = None,
    words: Union[Sequence[str], np.ndarray, pd.Series, None] = None,
) -> Dict[str, float]:
    """Compute secondary metrics; official F1 requires word identities.

    ``macro_f1`` retains its historical pooled two-class meaning. The pinned
    upstream metric averages two-class macro-F1 *within each word*, then across
    words; it is returned separately as ``official_f1`` when ``words`` is given.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    metrics = {
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "binary_f1": float(f1_score(y_true_arr, y_pred_arr, average="binary", pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)),
    }
    if words is not None:
        from Assignment_3.src.evaluation import fast_official_f1
        metrics["official_f1"] = fast_official_f1(words, y_true_arr, y_pred_arr)

    if y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, np.asarray(y_score, dtype=float)))
        except ValueError:
            metrics["roc_auc"] = float("nan")

    return metrics

