"""Inference and prediction script for SemEval-2027 Task 3, Subtask 2.
Supports both the adopted winning systems:
1. 'dense_xlmr': Dense Definition Interaction Head (Official nested F1: 0.6873, Calibrated 5x5: 0.6920) [DEFAULT]
2. 'baseline_svm': Calibrated Linear SVM (Official nested F1: 0.6268, Calibrated 5x5: 0.6358)
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root and deps to sys.path
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
from scripts.stage1_measure import BaselineSVMWrapper, load_data
from scripts.stage3_definition_matching import build_definition_features, DenseDefinitionModel

def train_and_calibrate(train_df: pd.DataFrame, model_type: str = "dense_xlmr", shrinkage: float = 0.5):
    """Calibrate thresholds on training data using 5-fold inner OOF, then fit on full data."""
    n_samples = len(train_df)
    strat_keys = train_df["word"].astype(str) + "_" + train_df["label"].astype(str)
    groups = train_df["near_dup"].to_numpy() if "near_dup" in train_df else near_dup_groups(train_df["context"].tolist())

    if model_type == "dense_xlmr":
        dense_feats = build_definition_features(train_df, encoder="xlmr")
        model_factory = lambda s: DenseDefinitionModel(dense_feats, seed=s)
    else:
        model_factory = lambda s: BaselineSVMWrapper(seed=s)

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
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

def load_custom_data(data_dir: Path):
    usages_file = data_dir / "usages.jsonl"
    sub2_file = data_dir / "subtask2.jsonl"

    if not usages_file.exists():
        raise FileNotFoundError(f"Missing usages file: {usages_file}")

    df_usages = pd.read_json(usages_file, lines=True)
    if sub2_file.exists():
        df_sub2 = pd.read_json(sub2_file, lines=True)
        if "label" in df_sub2.columns:
            data = df_usages.merge(df_sub2, on=["word", "sentence_id"])
            if "label" not in data.columns and "label_x" in data.columns:
                data["label"] = data["label_x"]
        else:
            data = df_usages.merge(df_sub2[["word", "sentence_id"]], on=["word", "sentence_id"])
    else:
        data = df_usages.copy()

    if "near_dup" not in data.columns:
        data["near_dup"] = near_dup_groups(data["text"], threshold=0.80)
    return data

def run_prediction(
    data_dir: Path,
    output_path: Path,
    model_type: str = "dense_xlmr",
    shrinkage: float = 0.5,
):
    print(f"Loading data from {data_dir}...")
    df = load_custom_data(data_dir=data_dir)
    print(f"Loaded {len(df)} rows across {df['word'].nunique()} unique lemmata.")

    print(f"Fitting and calibrating system: {model_type} (shrinkage={shrinkage})...")
    model, calibration = train_and_calibrate(df, model_type=model_type, shrinkage=shrinkage)

    print("Computed calibrated thresholds:")
    print(f"  Global threshold: {calibration['global_threshold']:.4f}")
    for w, t in sorted(calibration["per_word"].items()):
        print(f"  {w:12s}: {t:.4f}")

    probs = model.predict_proba(df)
    preds = apply_calibrated_thresholds(df["word"].to_numpy(), probs, calibration)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            record = {
                "word": row["word"],
                "sentence_id": row["sentence_id"],
                "label": int(preds[idx]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Predictions successfully written to: {output_path}")

    if "label" in df.columns:
        score = official_f1(df["word"], df["label"], preds)
        print(f"Training-fit Official F1 (reference): {score:.6f}")

def main():
    parser = argparse.ArgumentParser(description="SemEval-2027 Subtask 2 Prediction Script")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=ROOT / "data/raw/dev/SV",
        help="Path to SV raw data directory containing usages.jsonl, definitions.jsonl, subtask2.jsonl",
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
        default="dense_xlmr",
        choices=["dense_xlmr", "baseline_svm"],
        help="Model architecture: 'dense_xlmr' (SOTA definition interaction head) or 'baseline_svm'",
    )
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=0.5,
        help="Calibration shrinkage toward global threshold (default 0.5)",
    )
    args = parser.parse_args()
    run_prediction(args.data_dir, args.output_path, args.model, args.shrinkage)

if __name__ == "__main__":
    main()
