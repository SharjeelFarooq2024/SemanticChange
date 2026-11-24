"""Stage 1: Rigorous measurement of baseline SVM and advanced_hybrid under near-duplicate vs exact-hash grouping."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Assignment_3/.deps"))

import hashlib
import json
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from eval_harness import run_cv, near_dup_groups, diagnostics, official_f1
from Assignment_3.src.advanced_hybrid import AdvancedBuilder, AdvancedHead, AdvancedHybridModel
from Assignment_3.src.improved_hybrid import apply_thresholds, fit_thresholds
from Assignment_3.src.runtime import ROOT

def load_data():
    df_usages = pd.read_json(ROOT / "data/raw/dev/SV/usages.jsonl", lines=True)
    df_sub2 = pd.read_json(ROOT / "data/raw/dev/SV/subtask2.jsonl", lines=True)
    data = df_usages.merge(df_sub2, on=["word", "sentence_id"])
    data = data.rename(columns={"label": "label_value"})
    data["label"] = data["label_value"]

    # Compute exact-hash grouping
    data["exact_hash"] = data["text"].apply(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest())
    # Compute near-duplicate grouping
    data["near_dup"] = near_dup_groups(data["text"], threshold=0.80)
    return data

# ---------------------------------------------------------------------------- #
# Baseline SVM Model Factory
# ---------------------------------------------------------------------------- #
class BaselineSVMWrapper:
    def __init__(self, seed: int):
        self.pipe = Pipeline([
            ("tfidf", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1, max_features=50000, sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True, min_df=1, max_features=50000, sublinear_tf=True)),
            ])),
            ("model", LinearSVC(class_weight="balanced", random_state=seed, max_iter=10000)),
        ])

    def fit(self, df: pd.DataFrame):
        self.pipe.fit(df["text"], df["label"])
        return self

    def predict(self, df: pd.DataFrame):
        return self.pipe.predict(df["text"])

    def decision_function(self, df: pd.DataFrame):
        return self.pipe.decision_function(df["text"])

    def predict_proba(self, df: pd.DataFrame):
        margins = self.decision_function(df)
        probs = 1.0 / (1.0 + np.exp(-margins))
        return probs

# ---------------------------------------------------------------------------- #
# Advanced Hybrid Model Factory
# ---------------------------------------------------------------------------- #
class AdvancedHybridWrapper:
    def __init__(self, seed: int, config: dict):
        self.seed = seed
        self.config = config
        self.builder = AdvancedBuilder(config)
        self.head_setting = {"name": "mixed_experts_C1", "kind": "experts", "C": 1.0, "global_mix": 0.5}

    def fit(self, df: pd.DataFrame):
        views = self.builder.fit_transform(df, self.seed)
        self.head = AdvancedHead(self.head_setting, self.seed).fit(views, df)
        return self

    def decision_function(self, df: pd.DataFrame):
        views = self.builder.transform(df)
        return self.head.decision_function(views, df)

    def predict_proba(self, df: pd.DataFrame):
        margins = self.decision_function(df)
        return 1.0 / (1.0 + np.exp(-margins))

    def predict(self, df: pd.DataFrame):
        margins = self.decision_function(df)
        return (margins >= 0.0).astype(int)

def main():
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data and computing groups...")
    data = load_data()
    config = json.loads((ROOT / "Assignment_3/configs/advanced_hybrid.json").read_text(encoding="utf-8"))

    print(f"Total dataset: {len(data)} rows.")
    print(f"Exact hash unique groups: {data['exact_hash'].nunique()}")
    print(f"Near-duplicate unique groups: {data['near_dup'].nunique()}")

    leakage_records = []

    # 1. Baseline SVM - Near Dup Grouping (Headline)
    print("\n[1/4] Running Baseline SVM with Near-Duplicate Grouping...")
    t0 = time.time()
    svm_nd_res = run_cv(data, lambda s: BaselineSVMWrapper(s), n_splits=5, n_repeats=5, base_seed=42, group_col="near_dup")
    print(f"SVM Near-Dup CV: {svm_nd_res['mean_official_f1']:.6f} +/- {svm_nd_res['std_official_f1']:.6f} (took {time.time()-t0:.1f}s)")
    leakage_records.append({
        "system": "baseline_svm",
        "grouping": "near_dup_groups",
        "mean_official_f1": svm_nd_res["mean_official_f1"],
        "std_official_f1": svm_nd_res["std_official_f1"],
        "pooled_macro_f1": svm_nd_res["mean_pooled_macro_f1"],
    })

    # 2. Baseline SVM - Exact Hash Grouping
    print("\n[2/4] Running Baseline SVM with Exact-Hash Grouping...")
    t0 = time.time()
    svm_eh_res = run_cv(data, lambda s: BaselineSVMWrapper(s), n_splits=5, n_repeats=5, base_seed=42, group_col="exact_hash")
    print(f"SVM Exact-Hash CV: {svm_eh_res['mean_official_f1']:.6f} +/- {svm_eh_res['std_official_f1']:.6f} (took {time.time()-t0:.1f}s)")
    leakage_records.append({
        "system": "baseline_svm",
        "grouping": "exact_hash",
        "mean_official_f1": svm_eh_res["mean_official_f1"],
        "std_official_f1": svm_eh_res["std_official_f1"],
        "pooled_macro_f1": svm_eh_res["mean_pooled_macro_f1"],
    })

    # 3. Advanced Hybrid - Near Dup Grouping (Headline)
    print("\n[3/4] Running Advanced Hybrid with Near-Duplicate Grouping...")
    t0 = time.time()
    hyb_nd_res = run_cv(data, lambda s: AdvancedHybridWrapper(s, config), n_splits=5, n_repeats=5, base_seed=42, group_col="near_dup")
    print(f"Hybrid Near-Dup CV: {hyb_nd_res['mean_official_f1']:.6f} +/- {hyb_nd_res['std_official_f1']:.6f} (took {time.time()-t0:.1f}s)")
    leakage_records.append({
        "system": "advanced_hybrid",
        "grouping": "near_dup_groups",
        "mean_official_f1": hyb_nd_res["mean_official_f1"],
        "std_official_f1": hyb_nd_res["std_official_f1"],
        "pooled_macro_f1": hyb_nd_res["mean_pooled_macro_f1"],
    })

    # 4. Advanced Hybrid - Exact Hash Grouping
    print("\n[4/4] Running Advanced Hybrid with Exact-Hash Grouping...")
    t0 = time.time()
    hyb_eh_res = run_cv(data, lambda s: AdvancedHybridWrapper(s, config), n_splits=5, n_repeats=5, base_seed=42, group_col="exact_hash")
    print(f"Hybrid Exact-Hash CV: {hyb_eh_res['mean_official_f1']:.6f} +/- {hyb_eh_res['std_official_f1']:.6f} (took {time.time()-t0:.1f}s)")
    leakage_records.append({
        "system": "advanced_hybrid",
        "grouping": "exact_hash",
        "mean_official_f1": hyb_eh_res["mean_official_f1"],
        "std_official_f1": hyb_eh_res["std_official_f1"],
        "pooled_macro_f1": hyb_eh_res["mean_pooled_macro_f1"],
    })

    # Save Leakage Comparison
    df_leakage = pd.DataFrame(leakage_records)
    leakage_path = results_dir / "leakage_comparison.csv"
    df_leakage.to_csv(leakage_path, index=False)
    print(f"\nSaved leakage comparison to {leakage_path}")
    print(df_leakage)

    # 5. Diagnostics Tables (Mean OOF Probabilities & Predictions on Near-Dup Grouping)
    print("\nComputing Diagnostics Tables on Near-Dup Grouping...")
    # Average OOF probabilities across repeats
    svm_mean_prob = np.mean(svm_nd_res["oof_probabilities"], axis=0)
    svm_mean_pred = (svm_mean_prob >= 0.5).astype(int)
    diag_svm = diagnostics(data, svm_mean_pred, svm_mean_prob)
    diag_svm_path = results_dir / "diagnostics_baseline_svm.csv"
    diag_svm.to_csv(diag_svm_path, index=False)
    print(f"Saved SVM diagnostics to {diag_svm_path}")
    print("\nBaseline SVM Diagnostics:")
    print(diag_svm[["word", "total_support", "minority_support", "unstable_flag", "auc", "minority_ap", "oracle_f1", "actual_f1", "threshold_loss", "minority_recall"]])

    hyb_mean_prob = np.mean(hyb_nd_res["oof_probabilities"], axis=0)
    hyb_mean_pred = (hyb_mean_prob >= 0.5).astype(int)
    diag_hyb = diagnostics(data, hyb_mean_pred, hyb_mean_prob)
    diag_hyb_path = results_dir / "diagnostics_advanced_hybrid.csv"
    diag_hyb.to_csv(diag_hyb_path, index=False)
    print(f"\nSaved Hybrid diagnostics to {diag_hyb_path}")
    print("\nAdvanced Hybrid Diagnostics:")
    print(diag_hyb[["word", "total_support", "minority_support", "unstable_flag", "auc", "minority_ap", "oracle_f1", "actual_f1", "threshold_loss", "minority_recall"]])

    # Save OOF arrays for downstream paired bootstrap testing
    np.savez_compressed(
        results_dir / "stage1_oof_predictions.npz",
        svm_preds=svm_nd_res["oof_predictions"],
        svm_probs=svm_nd_res["oof_probabilities"],
        hyb_preds=hyb_nd_res["oof_predictions"],
        hyb_probs=hyb_nd_res["oof_probabilities"],
        labels=data["label"].to_numpy(),
        words=data["word"].to_numpy(),
    )
    print("\nStage 1 completed successfully.")

if __name__ == "__main__":
    main()
