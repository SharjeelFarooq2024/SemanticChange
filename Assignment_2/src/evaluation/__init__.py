"""Evaluation package for Assignment 2 retrieval and Subtask 2 classification."""
from .metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    average_precision,
    ndcg_at_k,
    compute_retrieval_metrics,
    compute_classification_metrics,
)
from .evaluator import (
    evaluate_retrieval_run,
    evaluate_binary_run,
    tune_operating_point,
    subgroup_evaluation,
)

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "average_precision",
    "ndcg_at_k",
    "compute_retrieval_metrics",
    "compute_classification_metrics",
    "evaluate_retrieval_run",
    "evaluate_binary_run",
    "tune_operating_point",
    "subgroup_evaluation",
]

