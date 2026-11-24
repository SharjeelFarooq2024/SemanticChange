"""Stage 2: Calibrate cheaply with best_threshold fitted on inner out-of-fold scores."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Assignment_3/.deps"))

import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from eval_harness import (
    official_f1,
    near_dup_groups,
    best_threshold,
    apply_calibrated_thresholds,
    paired_bootstrap,
    diagnostics,
)
from scripts.stage1_measure import load_data, BaselineSVMWrapper, AdvancedHybridWrapper
from Assignment_3.src.runtime import ROOT

def run_calibrated_cv(
    data: pd.DataFrame,
    model_factory_fn,
    shrinkage: float,
    n_splits: int = 5,
    n_repeats: int = 5,
    base_seed: int = 42,
    inner_splits: int = 3,
):
    df = data.reset_index(drop=True).copy()
    n_samples = len(df)
    labels = df["label"].to_numpy().astype(int)
    words = df["word"].to_numpy().astype(str)
    groups = df["near_dup"].to_numpy()
    strat_keys = df["word"].astype(str) + "_" + df["label"].astype(str)

    repeat_scores = []
    all_oof_preds = np.zeros((n_repeats, n_samples), dtype=int)
    all_oof_probs = np.zeros((n_repeats, n_samples), dtype=float)

    for r in range(n_repeats):
        seed = base_seed + r * 17
        outer_splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        oof_pred = np.full(n_samples, -1, dtype=int)
        oof_prob = np.full(n_samples, np.nan, dtype=float)

        for fold, (train_idx, test_idx) in enumerate(outer_splitter.split(df, strat_keys, groups)):
            train_fold = df.iloc[train_idx].reset_index(drop=True)
            test_fold = df.iloc[test_idx].reset_index(drop=True)

            inner_strat = train_fold["word"].astype(str) + "_" + train_fold["label"].astype(str)
            inner_groups = train_fold["near_dup"].to_numpy()

            inner_splitter = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed + fold * 10)
            inner_oof_probs = np.full(len(train_fold), np.nan, dtype=float)

            # Generate inner OOF probabilities strictly inside train_fold
            for inner_fit_idx, inner_val_idx in inner_splitter.split(train_fold, inner_strat, inner_groups):
                inner_fit = train_fold.iloc[inner_fit_idx].copy()
                inner_val = train_fold.iloc[inner_val_idx].copy()

                inner_model = model_factory_fn(seed + fold * 10 + 1)
                inner_model.fit(inner_fit)
                inner_oof_probs[inner_val_idx] = inner_model.predict_proba(inner_val)

            # Fit best_threshold on inner OOF scores
            calibration = best_threshold(
                train_fold["label"].to_numpy(),
                inner_oof_probs,
                train_fold["word"].to_numpy(),
                shrinkage=shrinkage,
            )

            # Fit full model on outer train_fold
            full_model = model_factory_fn(seed + fold)
            full_model.fit(train_fold)

            test_probs = full_model.predict_proba(test_fold)
            test_preds = apply_calibrated_thresholds(test_fold["word"].to_numpy(), test_probs, calibration)

            oof_prob[test_idx] = test_probs
            oof_pred[test_idx] = test_preds

        score = official_f1(words, labels, oof_pred)
        repeat_scores.append(score)
        all_oof_preds[r] = oof_pred
        all_oof_probs[r] = oof_prob

    mean_f1 = float(np.mean(repeat_scores))
    std_f1 = float(np.std(repeat_scores, ddof=1)) if len(repeat_scores) > 1 else 0.0

    return {
        "mean_official_f1": mean_f1,
        "std_official_f1": std_f1,
        "repeat_scores": repeat_scores,
        "oof_predictions": all_oof_preds,
        "oof_probabilities": all_oof_probs,
    }

def main():
    results_dir = ROOT / "results"
    data = load_data()
    config = json.loads((ROOT / "Assignment_3/configs/advanced_hybrid.json").read_text(encoding="utf-8"))

    # Load Stage 1 uncalibrated predictions
    stage1_file = results_dir / "stage1_oof_predictions.npz"
    if not stage1_file.exists():
        raise FileNotFoundError(f"Missing {stage1_file}")
    s1 = np.load(stage1_file)
    svm_uncal_preds = s1["svm_preds"]
    hyb_uncal_preds = s1["hyb_preds"]

    svm_uncal_mean = official_f1(data["word"], data["label"], svm_uncal_preds[0]) # check base
    print(f"Loaded Stage 1 OOF predictions.")

    # In Stage 1:
    # Baseline SVM uncalibrated: 0.613576
    # Advanced Hybrid uncalibrated: 0.605677
    # Baseline SVM was the better uncalibrated system, but Advanced Hybrid had 0.8347 AUC vs 0.7662 AUC.
    # Let's test calibration on Baseline SVM (the better system) and also on Advanced Hybrid!

    print("\n--- Calibrating Baseline SVM (The Better Uncalibrated System) ---")
    for shrink in [0.5, 0.75]:
        t0 = time.time()
        res = run_calibrated_cv(data, lambda s: BaselineSVMWrapper(s), shrinkage=shrink)
        print(f"Baseline SVM (shrinkage={shrink}): {res['mean_official_f1']:.6f} +/- {res['std_official_f1']:.6f} (took {time.time()-t0:.1f}s)")
        # Run paired bootstrap against uncalibrated SVM
        boot = paired_bootstrap(data, res["oof_predictions"], svm_uncal_preds)
        print(f"  Paired Bootstrap vs Uncalibrated SVM:")
        print(f"    Diff: {boot['point_diff']:+.6f}, 95% CI: [{boot['ci95_lo']:+.6f}, {boot['ci95_hi']:+.6f}], P(cal not better): {boot['p_a_not_better']:.4f}, Adopted: {boot['adopted']}")

        # Save results
        out_name = f"paired_bootstrap_svm_cal_{int(shrink*100)}"
        with open(results_dir / f"{out_name}.json", "w") as f:
            json.dump({**boot, "system": f"baseline_svm_cal_shrink_{shrink}", "baseline": "baseline_svm_uncalibrated"}, f, indent=2)

    print("\n--- Calibrating Advanced Hybrid ---")
    for shrink in [0.5, 0.75]:
        t0 = time.time()
        res = run_calibrated_cv(data, lambda s: AdvancedHybridWrapper(s, config), shrinkage=shrink)
        print(f"Advanced Hybrid (shrinkage={shrink}): {res['mean_official_f1']:.6f} +/- {res['std_official_f1']:.6f} (took {time.time()-t0:.1f}s)")
        # Run paired bootstrap against uncalibrated Hybrid
        boot = paired_bootstrap(data, res["oof_predictions"], hyb_uncal_preds)
        print(f"  Paired Bootstrap vs Uncalibrated Hybrid:")
        print(f"    Diff: {boot['point_diff']:+.6f}, 95% CI: [{boot['ci95_lo']:+.6f}, {boot['ci95_hi']:+.6f}], P(cal not better): {boot['p_a_not_better']:.4f}, Adopted: {boot['adopted']}")

        out_name = f"paired_bootstrap_hyb_cal_{int(shrink*100)}"
        with open(results_dir / f"{out_name}.json", "w") as f:
            json.dump({**boot, "system": f"advanced_hybrid_cal_shrink_{shrink}", "baseline": "advanced_hybrid_uncalibrated"}, f, indent=2)

        # Also paired bootstrap vs Baseline SVM uncalibrated
        boot_vs_svm = paired_bootstrap(data, res["oof_predictions"], svm_uncal_preds)
        print(f"  Paired Bootstrap vs Uncalibrated Baseline SVM:")
        print(f"    Diff: {boot_vs_svm['point_diff']:+.6f}, 95% CI: [{boot_vs_svm['ci95_lo']:+.6f}, {boot_vs_svm['ci95_hi']:+.6f}], P(hyb not better): {boot_vs_svm['p_a_not_better']:.4f}, Adopted: {boot_vs_svm['adopted']}")
        with open(results_dir / f"{out_name}_vs_svm.json", "w") as f:
            json.dump({**boot_vs_svm, "system": f"advanced_hybrid_cal_shrink_{shrink}", "baseline": "baseline_svm_uncalibrated"}, f, indent=2)

    print("\nStage 2 completed.")

if __name__ == "__main__":
    main()
