"""Unbiased Nested Cross-Validation for the newly adopted Dense Definition Interaction Head (XLM-R).
5-outer folds x 3-inner folds on near_dup_groups with shrinkage=0.5.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Assignment_3/.deps"))

import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score
from eval_harness import (
    official_f1,
    best_threshold,
    apply_calibrated_thresholds,
    diagnostics,
)
from scripts.stage1_measure import load_data
from scripts.stage3_definition_matching import build_definition_features, DenseDefinitionModel

def main():
    results_dir = ROOT / "results"
    data = load_data().reset_index(drop=True)
    data["group"] = data["near_dup"]
    dense_xlmr = build_definition_features(data, encoder="xlmr")

    n_samples = len(data)
    labels = data["label"].to_numpy().astype(int)
    words = data["word"].to_numpy().astype(str)
    groups = data["group"].to_numpy()
    strat_keys = data["word"].astype(str) + "_" + data["label"].astype(str)

    print("--- Running Final Nested CV for Dense XLM-R (5 outer x 3 inner folds, shrinkage=0.5) ---")
    outer_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.full(n_samples, -1, dtype=int)
    oof_probs = np.full(n_samples, np.nan, dtype=float)
    fold_records = []

    t0 = time.time()
    for fold, (train_idx, test_idx) in enumerate(outer_splitter.split(data, strat_keys, groups)):
        train_fold = data.iloc[train_idx].copy()
        test_fold = data.iloc[test_idx].copy()

        inner_strat = train_fold["word"].astype(str) + "_" + train_fold["label"].astype(str)
        inner_groups = train_fold["group"].to_numpy()
        inner_splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42 + fold * 10)
        inner_oof_probs = np.full(len(train_fold), np.nan, dtype=float)

        for inner_fit_idx, inner_val_idx in inner_splitter.split(train_fold, inner_strat, inner_groups):
            inner_fit = train_fold.iloc[inner_fit_idx].copy()
            inner_val = train_fold.iloc[inner_val_idx].copy()

            inner_model = DenseDefinitionModel(dense_xlmr, seed=42 + fold * 10 + 1)
            inner_model.fit(inner_fit)
            inner_oof_probs[inner_val_idx] = inner_model.predict_proba(inner_val)

        # Calibrate on inner OOF scores
        calibration = best_threshold(
            train_fold["label"].to_numpy(),
            inner_oof_probs,
            train_fold["word"].to_numpy(),
            shrinkage=0.5,
        )

        # Fit model on outer train fold
        model = DenseDefinitionModel(dense_xlmr, seed=42 + fold)
        model.fit(train_fold)

        test_probs = model.predict_proba(test_fold)
        test_preds = apply_calibrated_thresholds(test_fold["word"].to_numpy(), test_probs, calibration)

        oof_probs[test_idx] = test_probs
        oof_preds[test_idx] = test_preds

        fold_score = official_f1(test_fold["word"], test_fold["label"], test_preds)
        fold_records.append({"fold": fold, "official_f1": fold_score})
        print(f"  Fold {fold+1}/5 Official F1: {fold_score:.6f}")

    unbiased_official_f1 = official_f1(words, labels, oof_preds)
    pooled_macro_f1 = float(f1_score(labels, oof_preds, average="macro", zero_division=0))
    accuracy = float(accuracy_score(labels, oof_preds))

    print(f"\nDense XLM-R Nested CV Time: {time.time()-t0:.1f}s")
    print(f"Unbiased Official F1:    {unbiased_official_f1:.6f}")
    print(f"Pooled Macro F1:         {pooled_macro_f1:.6f}")
    print(f"Accuracy:                {accuracy:.6f}")

    payload = {
        "final_unbiased_official_f1": unbiased_official_f1,
        "pooled_macro_f1": pooled_macro_f1,
        "accuracy": accuracy,
        "outer_folds": fold_records,
        "pipeline": "DenseDefinitionInteraction(XLM-R: |u-d|, u*d, cos(u,d)) + LogisticRegression(class_weight='balanced', C=0.1) + InnerOOF_best_threshold(shrinkage=0.5)",
        "protocol": "5-fold StratifiedGroupKFold on near_dup_groups",
    }
    with open(results_dir / "final_nested_cv_score_dense_xlmr.json", "w") as f:
        json.dump(payload, f, indent=2)

    diag = diagnostics(data, oof_preds, oof_probs)
    diag.to_csv(results_dir / "diagnostics_final_calibrated_dense_xlmr.csv", index=False)
    print("\nFinal Nested Diagnostics Table:")
    print(diag[["word", "total_support", "minority_support", "unstable_flag", "auc", "minority_ap", "oracle_f1", "actual_f1", "threshold_loss", "minority_recall"]])

if __name__ == "__main__":
    main()
