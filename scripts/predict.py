"""Inference and prediction script for SemEval-2027 Task 3, Subtask 2.

Separates training/calibration source (--train_dir) from target prediction input (--data_dir)
so unlabeled unseen JSONL inputs are supported natively.

Modes:
- 'auto'        : Uses 'dense_xlmr' (0.6873 Nested CV) if cached XLM-R features or local weights cover
                  the inputs; cleanly falls back to calibrated 'baseline_svm' (0.6268 Nested CV) otherwise.
- 'dense_xlmr'  : Official adopted Dense XLM-R Definition Interaction Head (Docs, C=0.1).
- 'baseline_svm': Word/Char TF-IDF Linear SVM with inner-OOF calibrated thresholds.
- 'ensemble_xlmr_svm': Exploratory Dual-Context Dense XLM-R + Lemma-Balanced SVM (0.7006 Nested CV, 0.7168 LOPO).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Assignment_3/.deps"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from eval_harness import (
    official_f1,
    best_threshold,
    apply_calibrated_thresholds,
    near_dup_groups,
)
from scripts.stage1_measure import BaselineSVMWrapper
from scripts.stage3_definition_matching import (
    DenseFeatureStore,
    DenseDefinitionModel,
    EnsembleDenseSVM,
)


def load_custom_data(data_dir: Path, require_labels: bool = False) -> pd.DataFrame:
    """Load usages.jsonl and optionally merge labels from subtask2.jsonl."""
    usages_file = data_dir / "usages.jsonl"
    sub2_file = data_dir / "subtask2.jsonl"

    if not usages_file.exists():
        raise FileNotFoundError(f"Missing usages file: {usages_file}")

    df_usages = pd.read_json(usages_file, lines=True).reset_index(drop=True)
    if "sentence_id" not in df_usages.columns:
        df_usages["sentence_id"] = [f"usage_{i+1}" for i in range(len(df_usages))]

    if sub2_file.exists():
        df_sub2 = pd.read_json(sub2_file, lines=True)
        if "label" in df_sub2.columns:
            data = df_usages.merge(df_sub2[["word", "sentence_id", "label"]], on=["word", "sentence_id"], validate="one_to_one")
            data["label"] = data["label"].astype(int)
        else:
            data = df_usages.merge(df_sub2[["word", "sentence_id"]], on=["word", "sentence_id"], validate="one_to_one")
    else:
        data = df_usages.copy()

    if require_labels and "label" not in data.columns:
        raise ValueError(f"Training directory {data_dir} must contain labeled subtask2.jsonl")

    data = data.reset_index(drop=True)
    data["near_dup"] = near_dup_groups(data["text"].tolist(), threshold=0.80)
    return data


def _build_model_factory(model_type: str, train_df: pd.DataFrame, target_df: pd.DataFrame):
    """Resolve model architecture and cleanly handle fallback if dense artifacts are unavailable."""
    if model_type == "baseline_svm":
        return "baseline_svm", lambda s: BaselineSVMWrapper(seed=s)

    mode = "both" if model_type == "ensemble_xlmr_svm" else "docs"
    try:
        store = DenseFeatureStore(encoder="xlmr", mode=mode)
        # Preflight check on target rows if not all cached (verify weights exist locally)
        if not store.has_all_cached(target_df) or not store.has_all_cached(train_df):
            # Trigger resolution check without mutating if weights absent
            meta_path = ROOT / "Assignment_3/runs/completed/artifacts/xlmr_base.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                from Assignment_3.src.runtime import resolve_model
                resolve_model(meta["identity"]["model"], allow_download=False)
            else:
                raise FileNotFoundError(f"Missing {meta_path}")
    except (FileNotFoundError, KeyError, ImportError) as exc:
        if model_type == "auto":
            print(f"[auto fallback] Dense XLM-R unavailable for target inputs ({exc}). Falling back to calibrated baseline_svm.")
            return "baseline_svm", lambda s: BaselineSVMWrapper(seed=s)
        raise

    resolved = "dense_xlmr" if model_type == "auto" else model_type
    if resolved == "ensemble_xlmr_svm":
        return resolved, lambda s: EnsembleDenseSVM(store, alpha=0.8, C=0.05, per_lemma_weights=True, seed=s)
    return resolved, lambda s: DenseDefinitionModel(store, seed=s, C=0.1)


def train_and_calibrate(train_df: pd.DataFrame, model_factory, shrinkage: float = 0.5):
    """Calibrate per-word thresholds strictly on inner OOF splits of train_df, then fit full model."""
    n_samples = len(train_df)
    strat_keys = train_df["word"].astype(str) + "_" + train_df["label"].astype(str)
    groups = (
        train_df["near_dup"].to_numpy()
        if "near_dup" in train_df.columns
        else near_dup_groups(train_df["text"].tolist(), threshold=0.80)
    )

    n_unique_groups = len(np.unique(groups))
    n_splits = min(5, max(2, n_unique_groups))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    inner_oof_probs = np.full(n_samples, np.nan, dtype=float)

    for fit_idx, val_idx in splitter.split(train_df, strat_keys, groups):
        fit_df = train_df.iloc[fit_idx].copy()
        val_df = train_df.iloc[val_idx].copy()
        model = model_factory(42)
        model.fit(fit_df)
        inner_oof_probs[val_idx] = model.predict_proba(val_df)

    calibration = best_threshold(
        train_df["label"].to_numpy().astype(int),
        inner_oof_probs,
        train_df["word"].to_numpy().astype(str),
        shrinkage=shrinkage,
    )

    full_model = model_factory(42)
    full_model.fit(train_df)
    return full_model, calibration


def run_prediction(
    data_dir: Path,
    output_path: Path,
    train_dir: Path | None = None,
    model_type: str = "auto",
    shrinkage: float = 0.5,
):
    train_source = Path(train_dir) if train_dir is not None else ROOT / "data/raw/dev/SV"
    data_dir = Path(data_dir)

    print(f"Loading target inputs from {data_dir}...")
    target_df = load_custom_data(data_dir=data_dir, require_labels=False)
    print(f"Loaded {len(target_df)} rows across {target_df['word'].nunique()} lemmata.")

    print(f"Loading training/calibration data from {train_source}...")
    train_df = load_custom_data(data_dir=train_source, require_labels=True)

    resolved_name, model_factory = _build_model_factory(model_type, train_df, target_df)
    print(f"Fitting and calibrating system: {resolved_name} (shrinkage={shrinkage})...")
    model, calibration = train_and_calibrate(train_df, model_factory, shrinkage=shrinkage)

    print("Computed calibrated thresholds:")
    print(f"  Global threshold: {calibration['global_threshold']:.4f}")
    for w, t in sorted(calibration["per_word"].items()):
        print(f"  {w:12s}: {t:.4f}")

    probs = model.predict_proba(target_df)
    preds = apply_calibrated_thresholds(target_df["word"].to_numpy(), probs, calibration)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pred_label, row in zip(preds, target_df.itertuples(index=False)):
            record = {
                "word": str(row.word),
                "sentence_id": str(row.sentence_id),
                "label": int(pred_label),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Predictions successfully written to: {output_path}")

    if "label" in target_df.columns:
        score = official_f1(target_df["word"], target_df["label"], preds)
        same_dir = train_source.resolve() == data_dir.resolve()
        scope = "In-sample training reference Official F1 (NOT out-of-fold)" if same_dir else "Held-out Official F1"
        print(f"{scope}: {score:.6f}")


def main():
    parser = argparse.ArgumentParser(description="SemEval-2027 Subtask 2 Prediction Script")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=ROOT / "data/raw/dev/SV",
        help="Path to directory containing target usages.jsonl (and optional subtask2.jsonl)",
    )
    parser.add_argument(
        "--train_dir",
        type=Path,
        default=ROOT / "data/raw/dev/SV",
        help="Path to labeled training directory used to fit and calibrate the model",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=ROOT / "results/subtask2_predictions.jsonl",
        help="Path to save output predictions in JSONL format",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        choices=["auto", "dense_xlmr", "baseline_svm", "ensemble_xlmr_svm"],
        help="Model architecture (default: auto -> dense_xlmr with baseline_svm fallback)",
    )
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=0.5,
        help="Calibration shrinkage toward global threshold (default 0.5)",
    )
    args = parser.parse_args()
    run_prediction(args.data_dir, args.output_path, args.train_dir, args.model, args.shrinkage)


if __name__ == "__main__":
    main()
