"""Ablation experiments framework for Assignment 2.

Implements the experimental ablation plan required by Section 4:
1. Candidate Scope Ablation: per_word vs global.
2. Encoder Backbone Ablation: multilingual-e5-small vs xlm-roberta-base.
3. Fusion Parameter Ablation: RRF k in {10, 30, 60, 100}.
4. Decision Cutoff Ablation: naive k=50 vs tuned operating point k*.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..data.data_loader import RetrievalBundle, load_bundle_validated
from ..pipeline import run_system
from ..retrieval.bm25_retriever import BM25Retriever
from ..retrieval.dense_retriever import DenseRetriever
from ..retrieval.fusion import reciprocal_rank_fusion
from ..retrieval.prediction_io import save_ranked_predictions
from ..utils.paths import resolve_a2_path
from .evaluator import evaluate_binary_run, evaluate_retrieval_run, tune_operating_point


def run_scope_ablation(
    bundle: RetrievalBundle,
    qrels_path: Path,
    system: str = "bm25",
) -> pd.DataFrame:
    """Compare per_word vs global candidate scoping."""
    results = []
    for scope in ["per_word", "global"]:
        summary = run_system(system=system, scope=scope, top_k=100)
        ranked_path = Path(summary["outputs"]["ranked"])
        metrics = evaluate_retrieval_run(ranked_path, qrels_path)
        metrics["system"] = system
        metrics["ablation_axis"] = "candidate_scope"
        metrics["variant"] = scope
        results.append(metrics)
    return pd.DataFrame(results)


def run_fusion_k_ablation(
    bm25_ranked_path: Path,
    dense_ranked_path: Path,
    qrels_path: Path,
    k_values: Tuple[int, ...] = (10, 30, 60, 100),
) -> pd.DataFrame:
    """Evaluate sensitivity of Reciprocal Rank Fusion to the smoothing parameter k."""
    bm25_df = pd.read_csv(bm25_ranked_path)
    dense_df = pd.read_csv(dense_ranked_path)

    # Group by query
    from ..retrieval.base import Hit

    bm25_hits_by_query = {
        str(qid): [
            Hit(doc_id=str(row["doc_id"]), score=float(row["score"]), rank=int(row["rank"]))
            for _, row in grp.iterrows()
        ]
        for qid, grp in bm25_df.groupby("query_id")
    }
    dense_hits_by_query = {
        str(qid): [
            Hit(doc_id=str(row["doc_id"]), score=float(row["score"]), rank=int(row["rank"]))
            for _, row in grp.iterrows()
        ]
        for qid, grp in dense_df.groupby("query_id")
    }

    records = []
    all_queries = sorted(set(bm25_hits_by_query.keys()) & set(dense_hits_by_query.keys()))

    for k_val in k_values:
        fused_by_query: Dict[str, List[Hit]] = {}
        for qid in all_queries:
            b_hits = bm25_hits_by_query.get(qid, [])
            d_hits = dense_hits_by_query.get(qid, [])
            fused_by_query[qid] = reciprocal_rank_fusion([b_hits, d_hits], k=k_val, top_k=100)

        # Build temporary DataFrame for evaluation
        fused_rows = []
        for qid, hits in fused_by_query.items():
            for h in hits:
                fused_rows.append({"query_id": qid, "doc_id": h.doc_id, "score": h.score, "rank": h.rank})
        fused_df = pd.DataFrame(fused_rows)

        metrics = evaluate_retrieval_run(fused_df, qrels_path)
        metrics["ablation_axis"] = "rrf_smoothing_k"
        metrics["variant"] = f"k={k_val}"
        metrics["k_parameter"] = k_val
        records.append(metrics)

    return pd.DataFrame(records)


def run_threshold_ablation(
    systems_ranked: Dict[str, Path],
    qrels_path: Path,
) -> pd.DataFrame:
    """Compare naive top-50 cutoff vs optimized cutoff K*."""
    records = []
    for sys_name, r_path in systems_ranked.items():
        # Naive k=50
        from .evaluator import _to_dataframe, load_qrels
        r_df = _to_dataframe(r_path)
        q_df, _ = load_qrels(qrels_path)

        # Naive binary at k=50
        naive_hits = r_df[r_df["rank"] <= 50][["query_id", "doc_id"]].copy()
        naive_hits["predicted_label"] = 1
        m_naive = pd.merge(q_df[["query_id", "doc_id", "relevance"]], naive_hits, on=["query_id", "doc_id"], how="left")
        m_naive["predicted_label"] = m_naive["predicted_label"].fillna(0).astype(int)

        from .metrics import compute_classification_metrics
        naive_metrics = compute_classification_metrics(m_naive["relevance"], m_naive["predicted_label"])
        naive_metrics["system"] = sys_name
        naive_metrics["ablation_axis"] = "operating_point"
        naive_metrics["variant"] = "naive_k=50"
        naive_metrics["cutoff_k"] = 50
        records.append(naive_metrics)

        # Tuned k*
        best_k, best_f1, _, _ = tune_operating_point(r_path, qrels_path, k_range=range(5, 105, 5))
        tuned_hits = r_df[r_df["rank"] <= best_k][["query_id", "doc_id"]].copy()
        tuned_hits["predicted_label"] = 1
        m_tuned = pd.merge(q_df[["query_id", "doc_id", "relevance"]], tuned_hits, on=["query_id", "doc_id"], how="left")
        m_tuned["predicted_label"] = m_tuned["predicted_label"].fillna(0).astype(int)

        tuned_metrics = compute_classification_metrics(m_tuned["relevance"], m_tuned["predicted_label"])
        tuned_metrics["system"] = sys_name
        tuned_metrics["ablation_axis"] = "operating_point"
        tuned_metrics["variant"] = f"tuned_k*={best_k}"
        tuned_metrics["cutoff_k"] = best_k
        records.append(tuned_metrics)

    return pd.DataFrame(records)
