"""Compatibility helpers and entry point for the completed experiment suite.

The canonical implementation is run_complete_experiments.py. Historical result
files outside runs/completed use an obsolete primary metric and are not final.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Repository paths
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.classical_features import (  # noqa: E402
    build_ablated_preprocessor,
    build_feature_frame,
    build_interpretable_preprocessor,
)
from src.baselines.common import (  # noqa: E402
    join_subtask2_labels,
    load_baseline_bundle,
    load_config,
    load_frozen_usage_embeddings,
    set_seed,
)
from Assignment_2.src.data.data_loader import load_bundle_validated  # noqa: E402
from Assignment_2.src.evaluation.metrics import (  # noqa: E402
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from Assignment_2.src.retrieval.fusion import reciprocal_rank_fusion  # noqa: E402
from Assignment_2.src.retrieval.base import Hit  # noqa: E402

A3_ROOT = REPO_ROOT / "Assignment_3"
SEEDS = [42, 100, 2023, 2026, 777]
N_BOOT = 1000
K_GRID = list(range(5, 105, 5))


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("assignment_3_part1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_dir / "experiment_execution.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def official_subtask2_f1(widx: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, n_words: int) -> float:
    """Official within-word two-class macro-F1, averaged over observed words."""
    from Assignment_3.src.evaluation import fast_official_f1
    return fast_official_f1(widx, y_true, y_pred)


def per_word_f1_vector(widx: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, n_words: int) -> np.ndarray:
    """Compute per-word positive-class F1 vector of length n_words."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    tp = np.bincount(widx, weights=(y == 1) & (p == 1), minlength=n_words)
    fp = np.bincount(widx, weights=(y == 0) & (p == 1), minlength=n_words)
    fn = np.bincount(widx, weights=(y == 1) & (p == 0), minlength=n_words)
    den = 2 * tp + fp + fn
    return np.divide(2 * tp, den, out=np.zeros_like(den, dtype=float), where=den > 0)


def compute_all_metrics(widx: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, n_words: int) -> Dict[str, float]:
    """Compute official and secondary classification metrics."""
    return {
        "official_f1": official_subtask2_f1(widx, y_true, y_pred, n_words),
        "pooled_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def pred_at_k(rows: pd.DataFrame, ranked: pd.DataFrame, k: int) -> np.ndarray:
    """Predict 1 iff (word, sentence_id) is ranked <= k by the retriever."""
    r = ranked.set_index(["query_id", "doc_id"])["rank"]
    ranks = pd.Series(list(zip(rows["word"], rows["sentence_id"]))).map(r).to_numpy(dtype=float)
    return (np.nan_to_num(ranks, nan=1e9) <= k).astype(int)


def bootstrap_ci(stat_fn, n: int, rng: np.random.Generator, n_boot: int = N_BOOT) -> Tuple[float, float]:
    vals = np.array([stat_fn(rng.integers(0, n, n)) for _ in range(n_boot)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_permutation_test(a: np.ndarray, b: np.ndarray, n_permutations: int = 10000, seed: int = 42) -> float:
    """Exact two-sided paired permutation test on difference in means."""
    diff = a - b
    obs_diff = np.abs(np.mean(diff))
    rng = np.random.default_rng(seed)
    # Randomly flip signs of differences
    signs = rng.choice([-1, 1], size=(n_permutations, len(diff)))
    perm_diffs = np.abs(np.mean(diff * signs, axis=1))
    return float((np.sum(perm_diffs >= obs_diff) + 1) / (n_permutations + 1))


def get_hardware_info() -> Dict[str, Any]:
    """Collect hardware specifications and environment details using standard library."""
    import ctypes
    info: Dict[str, Any] = {
        "os": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "python_version": sys.version,
    }
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            info["total_ram_gb"] = round(stat.ullTotalPhys / (1024**3), 2)
            info["available_ram_gb"] = round(stat.ullAvailPhys / (1024**3), 2)
    except Exception:
        pass
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_device_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["torch_version"] = str(e)
    return info


def build_classical_models(config: Dict[str, Any], seed: int) -> Dict[str, Pipeline]:
    """Instantiate classical baselines pipelines with specified seed."""
    c_cfg = config["classical"]
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "tfidf_logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=tuple(c_cfg["word_ngram_range"]),
                lowercase=True,
                min_df=c_cfg["min_df"],
                max_features=c_cfg["max_features"],
                sublinear_tf=True,
            )),
            ("model", LogisticRegression(
                max_iter=c_cfg["max_iter"],
                class_weight=c_cfg["class_weight"],
                random_state=seed,
            )),
        ]),
        "tfidf_complement_nb": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=tuple(c_cfg["word_ngram_range"]),
                lowercase=True,
                min_df=c_cfg["min_df"],
                max_features=c_cfg["max_features"],
                sublinear_tf=True,
            )),
            ("model", ComplementNB(norm=True)),
        ]),
        "word_char_tfidf_linear_svm": Pipeline([
            ("tfidf", FeatureUnion([
                ("word", TfidfVectorizer(
                    ngram_range=tuple(c_cfg["word_ngram_range"]),
                    lowercase=True,
                    min_df=c_cfg["min_df"],
                    max_features=c_cfg["max_features"],
                    sublinear_tf=True,
                )),
                ("char", TfidfVectorizer(
                    analyzer="char",
                    ngram_range=tuple(c_cfg["char_ngram_range"]),
                    lowercase=True,
                    min_df=c_cfg["min_df"],
                    max_features=c_cfg["max_features"],
                    sublinear_tf=True,
                )),
            ])),
            ("model", LinearSVC(class_weight=c_cfg["class_weight"], random_state=seed)),
        ]),
    }


def main() -> None:
    from Assignment_3.scripts.run_complete_experiments import main as run_complete
    run_complete()


if __name__ == "__main__":
    main()
