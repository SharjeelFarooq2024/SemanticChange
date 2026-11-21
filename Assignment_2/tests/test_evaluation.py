"""Tests for the Assignment 2 evaluation engine and metrics."""
from __future__ import annotations

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from Assignment_2.src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    average_precision,
    ndcg_at_k,
    compute_retrieval_metrics,
    compute_classification_metrics,
)
from Assignment_2.src.evaluation.evaluator import (
    evaluate_retrieval_run,
    evaluate_binary_run,
    tune_operating_point,
    subgroup_evaluation,
)


def test_precision_at_k_exact():
    ranked = ["d1", "d2", "d3", "d4"]
    rel = {"d1", "d3"}
    assert precision_at_k(ranked, rel, k=2) == 0.5  # 1 hit (d1) in 2
    assert precision_at_k(ranked, rel, k=4) == 0.5  # 2 hits (d1, d3) in 4
    assert precision_at_k(ranked, rel, k=1) == 1.0  # 1 hit in 1
    assert precision_at_k([], rel, k=2) == 0.0
    assert precision_at_k(ranked, rel, k=0) == 0.0


def test_recall_at_k_exact():
    ranked = ["d1", "d2", "d3", "d4"]
    rel = {"d1", "d3", "d5"}
    assert recall_at_k(ranked, rel, k=1) == pytest.approx(1.0 / 3.0)
    assert recall_at_k(ranked, rel, k=3) == pytest.approx(2.0 / 3.0)
    assert recall_at_k(ranked, rel, k=4) == pytest.approx(2.0 / 3.0)
    assert recall_at_k(ranked, set(), k=2) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["d1", "d2", "d3"], {"d1"}) == 1.0
    assert reciprocal_rank(["d2", "d1", "d3"], {"d1"}) == 0.5
    assert reciprocal_rank(["d2", "d3", "d1"], {"d1"}) == pytest.approx(1.0 / 3.0)
    assert reciprocal_rank(["d2", "d3"], {"d1"}) == 0.0


def test_average_precision():
    # Hits at rank 1 and 3 out of 2 total relevant
    # P@1 = 1/1 = 1.0, P@3 = 2/3 = 0.6667 -> AP = (1.0 + 0.6667) / 2 = 0.8333
    ranked = ["d1", "d2", "d3", "d4"]
    rel = {"d1", "d3"}
    assert average_precision(ranked, rel) == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert average_precision(ranked, set()) == 0.0


def test_ndcg_at_k():
    ranked = ["d1", "d2", "d3"]
    rel = {"d1", "d2"}
    # Perfect ranking for top 2
    assert ndcg_at_k(ranked, rel, k=2) == pytest.approx(1.0)
    # Reversed ranking
    rev_ranked = ["d3", "d2", "d1"]
    score = ndcg_at_k(rev_ranked, rel, k=3)
    assert 0.0 < score < 1.0


def test_classification_metrics_macro_f1():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]
    # class 1: TP=1, FP=0, FN=1 -> P=1.0, R=0.5 -> F1=0.6667
    # class 0: TN=2, FP=0, FN=0 -> P=2/3, R=2/2 -> F1=0.8
    # macro_f1 = (2/3 + 0.8) / 2 = 0.7333
    m = compute_classification_metrics(y_true, y_pred)
    assert "macro_f1" in m
    assert "accuracy" in m
    assert m["accuracy"] == 0.75
    assert m["macro_f1"] == pytest.approx((2.0 / 3.0 + 0.8) / 2.0)


def test_tune_operating_point(tmp_path: Path):
    ranked_csv = tmp_path / "ranked.csv"
    qrels_csv = tmp_path / "qrels.csv"

    # 1 query, 4 docs, d1 and d2 are positive
    qrels_df = pd.DataFrame({
        "query_id": ["q1", "q1", "q1", "q1"],
        "doc_id": ["d1", "d2", "d3", "d4"],
        "relevance": [1, 1, 0, 0],
    })
    qrels_df.to_csv(qrels_csv, index=False)

    ranked_df = pd.DataFrame({
        "query_id": ["q1", "q1", "q1", "q1"],
        "doc_id": ["d1", "d2", "d3", "d4"],
        "rank": [1, 2, 3, 4],
        "score": [0.9, 0.8, 0.3, 0.1],
    })
    ranked_df.to_csv(ranked_csv, index=False)

    best_k, best_f1, tuned_binary, sweep_df = tune_operating_point(
        ranked_csv, qrels_csv, k_range=[1, 2, 3, 4]
    )
    assert best_k == 2  # top 2 perfectly captures both positives
    assert best_f1 == 1.0
    assert len(tuned_binary) == 4
    assert tuned_binary["label"].tolist() == [1, 1, 0, 0]

