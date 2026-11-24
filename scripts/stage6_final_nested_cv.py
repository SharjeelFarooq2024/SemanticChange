"""Stage 6: Final nested CV run on the entire selection-plus-calibration pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Assignment_3/.deps"))

import json
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score

from eval_harness import (
    official_f1,
    best_threshold,
    apply_calibrated_thresholds,
    diagnostics,
)
from scripts.stage1_measure import load_data, BaselineSVMWrapper
from Assignment_3.src.runtime import ROOT

def main():
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    data = load_data()

    df = data.reset_index(drop=True).copy()
    n_samples = len(df)
    labels = df["label"].to_numpy().astype(int)
    words = df["word"].to_numpy().astype(str)
    groups = df["near_dup"].to_numpy()
    strat_keys = df["word"].astype(str) + "_" + df["label"].astype(str)

    print("--- Running Final Nested CV (5 outer folds x 3 inner folds, shrinkage=0.5) ---")
    outer_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.full(n_samples, -1, dtype=int)
    oof_probs = np.full(n_samples, np.nan, dtype=float)
    fold_records = []

    t0 = time.time()
    for fold, (train_idx, test_idx) in enumerate(outer_splitter.split(df, strat_keys, groups)):
        train_fold = df.iloc[train_idx].reset_index(drop=True)
        test_fold = df.iloc[test_idx].reset_index(drop=True)

        inner_strat = train_fold["word"].astype(str) + "_" + train_fold["label"].astype(str)
        inner_groups = train_fold["near_dup"].to_numpy()
        inner_splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42 + fold * 10)
        inner_oof_probs = np.full(len(train_fold), np.nan, dtype=float)

        for inner_fit_idx, inner_val_idx in inner_splitter.split(train_fold, inner_strat, inner_groups):
            inner_fit = train_fold.iloc[inner_fit_idx].copy()
            inner_val = train_fold.iloc[inner_val_idx].copy()

            inner_model = BaselineSVMWrapper(seed=42 + fold * 10 + 1)
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
        model = BaselineSVMWrapper(seed=42 + fold)
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

    print(f"\nFinal Nested CV Execution Time: {time.time()-t0:.1f}s")
    print(f"Unbiased Official F1:    {unbiased_official_f1:.6f}")
    print(f"Pooled Macro F1:         {pooled_macro_f1:.6f}")
    print(f"Accuracy:                {accuracy:.6f}")

    # Save final nested CV metrics
    final_payload = {
        "final_unbiased_official_f1": unbiased_official_f1,
        "pooled_macro_f1": pooled_macro_f1,
        "accuracy": accuracy,
        "outer_folds": fold_records,
        "pipeline": "FeatureUnion(word_1_2, char_3_5) + LinearSVC(class_weight='balanced') + InnerOOF_best_threshold(shrinkage=0.5)",
        "protocol": "5-fold StratifiedGroupKFold on near_dup_groups",
    }
    with open(results_dir / "final_nested_cv_score.json", "w") as f:
        json.dump(final_payload, f, indent=2)

    # Save final predictions
    df_final = df[["word", "sentence_id", "period_label", "label"]].copy()
    df_final["calibrated_prediction"] = oof_preds
    df_final["probability"] = oof_probs
    df_final.to_csv(results_dir / "final_unbiased_predictions.csv", index=False)

    # Final diagnostics on unbiased predictions
    diag_final = diagnostics(df, oof_preds, oof_probs)
    diag_final.to_csv(results_dir / "diagnostics_final_calibrated_svm.csv", index=False)
    print("\nFinal Diagnostics Table:")
    print(diag_final[["word", "total_support", "minority_support", "unstable_flag", "auc", "minority_ap", "oracle_f1", "actual_f1", "threshold_loss", "minority_recall"]])

if __name__ == "__main__":
    main()
