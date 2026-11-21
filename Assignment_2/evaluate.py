"""Unified evaluation and preliminary verification entry point for Assignment 2.

Evaluates retrieval rankings, computes official Subtask 2 classification metrics,
tunes operating thresholds, conducts subgroup robustness analysis (by word and period),
benchmarks against Assignment 1 baselines, runs ablations, and generates summary tables.

This entry point performs exploratory sweeps on all supplied development labels.
Use Assignment_3/scripts/run_complete_experiments.py for the final held-out,
training-only-calibrated experiments and the pinned official scorer.

Usage:
    python Assignment_2/evaluate.py [--output-dir outputs/evaluation]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Bootstrap Assignment_2 package path
A2_ROOT = Path(__file__).resolve().parent
REPO_ROOT = A2_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Assignment_2.src.data.data_loader import load_bundle_validated
from Assignment_2.src.evaluation.ablation import (
    run_fusion_k_ablation,
    run_threshold_ablation,
)
from Assignment_2.src.evaluation.evaluator import (
    evaluate_binary_run,
    evaluate_retrieval_run,
    subgroup_evaluation,
    tune_operating_point,
)
from Assignment_2.src.utils.logging_utils import get_logger
from Assignment_2.src.utils.paths import ensure_dir, resolve_a2_path


def load_assignment1_comparison(repo_root: Path) -> Optional[pd.DataFrame]:
    """Load baseline results from Assignment 1 for side-by-side benchmark comparison."""
    a1_table_path = repo_root / "outputs" / "baselines" / "tables" / "subtask2_comparison.csv"
    if a1_table_path.exists():
        df = pd.read_csv(a1_table_path)
        df["phase"] = "Assignment 1"
        return df
    return None


def run_all_evaluations(output_dir: Path) -> Dict[str, Any]:
    log = get_logger("evaluate")
    log.info("Starting comprehensive Assignment 2 evaluation...")
    ensure_dir(output_dir)

    bundle = load_bundle_validated()
    qrels_path = A2_ROOT / "predictions" / "qrels.csv"
    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file missing at {qrels_path}")

    systems = ["bm25", "dense", "hybrid"]
    retrieval_results = []
    classification_results = []
    subgroup_word_results = []
    subgroup_period_results = []
    tuned_k_values = {}

    systems_ranked_paths = {}

    for sys_name in systems:
        sys_dir = A2_ROOT / "predictions" / sys_name
        ranked_path = sys_dir / "ranked.csv"
        binary_path = sys_dir / "binary_predictions.csv"

        if not ranked_path.exists() or not binary_path.exists():
            log.warning("Skipping %s: predictions not found in %s", sys_name, sys_dir)
            continue

        systems_ranked_paths[sys_name] = ranked_path

        # 1. Retrieval Ranking Metrics
        ret_metrics = evaluate_retrieval_run(ranked_path, qrels_path)
        ret_metrics["system"] = sys_name
        retrieval_results.append(ret_metrics)

        # 2. Binary Classification Metrics (Naive k=50)
        naive_cls_metrics = evaluate_binary_run(binary_path, qrels_path)
        naive_cls_metrics["system"] = f"{sys_name}_naive_k50"
        naive_cls_metrics["operating_point"] = "naive_top_50"
        naive_cls_metrics["phase"] = "Assignment 2"
        classification_results.append(naive_cls_metrics)

        # 3. Operating Point Tuning (Optimal rank cutoff k*)
        best_k, best_macro_f1, tuned_binary_df, sweep_df = tune_operating_point(
            ranked_path, qrels_path, k_range=range(5, 105, 5)
        )
        tuned_k_values[sys_name] = best_k
        tuned_binary_path = sys_dir / "binary_predictions_tuned.csv"
        tuned_binary_df.to_csv(tuned_binary_path, index=False)
        sweep_df.to_csv(output_dir / f"sweep_{sys_name}.csv", index=False)

        tuned_cls_metrics = evaluate_binary_run(tuned_binary_path, qrels_path)
        tuned_cls_metrics["system"] = f"{sys_name}_tuned_k{best_k}"
        tuned_cls_metrics["operating_point"] = f"tuned_top_{best_k}"
        tuned_cls_metrics["phase"] = "Assignment 2"
        classification_results.append(tuned_cls_metrics)

        # 4. Subgroup Robustness Analysis (on tuned predictions)
        word_breakdown = subgroup_evaluation(tuned_binary_path, qrels_path, bundle.documents, group_by="word")
        word_breakdown["system"] = sys_name
        subgroup_word_results.append(word_breakdown)

        period_breakdown = subgroup_evaluation(tuned_binary_path, qrels_path, bundle.documents, group_by="period_label")
        period_breakdown["system"] = sys_name
        subgroup_period_results.append(period_breakdown)

    # Save Retrieval Comparison Table
    ret_df = pd.DataFrame(retrieval_results)
    if not ret_df.empty:
        # Reorder key columns
        first_cols = ["system", "mrr", "map", "ndcg@10", "p@10", "recall@10", "p@20", "recall@50"]
        ordered = [c for c in first_cols if c in ret_df.columns] + [c for c in ret_df.columns if c not in first_cols]
        ret_df = ret_df[ordered]
        ret_path = output_dir / "retrieval_metrics_comparison.csv"
        ret_df.to_csv(ret_path, index=False)
        log.info("Saved retrieval metrics to %s", ret_path)

    # Save Unified Subtask 2 Classification Comparison (with Assignment 1 Baselines)
    cls_df = pd.DataFrame(classification_results)
    a1_df = load_assignment1_comparison(REPO_ROOT)
    if a1_df is not None:
        # Select common columns for benchmark comparison
        common_cols = ["system", "phase", "accuracy", "macro_f1", "precision", "recall"]
        a1_renamed = a1_df.rename(columns={"model": "system"})
        a1_formatted = a1_renamed[
            [c for c in ["system", "phase", "accuracy", "macro_f1", "precision", "recall"] if c in a1_renamed.columns]
        ]
        cls_df_formatted = cls_df[
            [c for c in ["system", "phase", "accuracy", "macro_f1", "precision", "recall"] if c in cls_df.columns]
        ]
        unified_benchmark = pd.concat([a1_formatted, cls_df_formatted], ignore_index=True)
        unified_benchmark = unified_benchmark.sort_values("macro_f1", ascending=False)
        unified_benchmark.to_csv(output_dir / "unified_subtask2_benchmark.csv", index=False)
        log.info("Saved unified benchmark comparison to %s", output_dir / "unified_subtask2_benchmark.csv")

    cls_path = output_dir / "subtask2_classification_comparison.csv"
    cls_df.to_csv(cls_path, index=False)

    # Save Subgroup Breakdown Tables
    if subgroup_word_results:
        all_word_df = pd.concat(subgroup_word_results, ignore_index=True)
        all_word_df.to_csv(output_dir / "subgroup_per_word.csv", index=False)
        log.info("Saved per-word breakdown to %s", output_dir / "subgroup_per_word.csv")

    if subgroup_period_results:
        all_period_df = pd.concat(subgroup_period_results, ignore_index=True)
        all_period_df.to_csv(output_dir / "subgroup_per_period.csv", index=False)
        log.info("Saved per-period breakdown to %s", output_dir / "subgroup_per_period.csv")

    # 5. Run Ablations
    log.info("Running ablation studies...")
    ablation_records = []

    # Operating Point Ablation
    if systems_ranked_paths:
        thresh_ablation_df = run_threshold_ablation(systems_ranked_paths, qrels_path)
        thresh_ablation_df.to_csv(output_dir / "ablation_operating_point.csv", index=False)
        ablation_records.append(thresh_ablation_df)

    # Fusion k Ablation
    if "bm25" in systems_ranked_paths and "dense" in systems_ranked_paths:
        fusion_ablation_df = run_fusion_k_ablation(
            systems_ranked_paths["bm25"],
            systems_ranked_paths["dense"],
            qrels_path,
            k_values=(10, 30, 60, 100),
        )
        fusion_ablation_df.to_csv(output_dir / "ablation_fusion_k.csv", index=False)
        ablation_records.append(fusion_ablation_df)

    # Scope Ablation (bm25 per_word vs global)
    global_bm25_ranked = A2_ROOT / "predictions" / "bm25_global" / "ranked.csv"
    if global_bm25_ranked.exists() and "bm25" in systems_ranked_paths:
        m_per_word = evaluate_retrieval_run(systems_ranked_paths["bm25"], qrels_path)
        m_per_word["scope"] = "per_word"
        m_per_word["system"] = "bm25"
        m_global = evaluate_retrieval_run(global_bm25_ranked, qrels_path)
        m_global["scope"] = "global"
        m_global["system"] = "bm25"
        scope_df = pd.DataFrame([m_per_word, m_global])
        scope_df.to_csv(output_dir / "ablation_scope.csv", index=False)
        log.info("Saved scope ablation to %s", output_dir / "ablation_scope.csv")

    log.info("All evaluations completed successfully!")
    return {
        "retrieval_df": ret_df,
        "classification_df": cls_df,
        "tuned_k": tuned_k_values,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Assignment 2 retrieval systems")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation artifacts (relative to Assignment_2/)",
    )
    args = parser.parse_args()

    out_dir = resolve_a2_path(args.output_dir)
    run_all_evaluations(out_dir)


if __name__ == "__main__":
    main()
