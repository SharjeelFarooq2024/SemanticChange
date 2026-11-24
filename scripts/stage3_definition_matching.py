"""Stage 3: Definition Matching and Lemma-Specific Cost Rebalancing Experiments.
Evaluated under 5-fold x 5-repeat StratifiedGroupKFold on near_dup_groups.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Assignment_3/.deps"))

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from eval_harness import (
    official_f1,
    run_cv,
    best_threshold,
    apply_calibrated_thresholds,
    paired_bootstrap,
    diagnostics,
)
from scripts.stage1_measure import load_data, BaselineSVMWrapper

def build_definition_features(data: pd.DataFrame, encoder: str = "xlmr"):
    if encoder == "xlmr":
        cache = np.load(ROOT / "Assignment_3/runs/completed/artifacts/xlmr_base.npz")
    else:
        cache = np.load(ROOT / "Assignment_3/runs/completed/artifacts/e5_small.npz")

    docs = cache["documents"]
    queries = cache["queries"]
    query_words = list(cache["query_words"])
    word_to_query_idx = {w: i for i, w in enumerate(query_words)}

    n_samples = len(data)
    d_dim = queries.shape[1]
    
    # Compute matched definition vector for each row
    matched_queries = np.zeros((n_samples, d_dim), dtype=np.float32)
    for i, row in data.reset_index(drop=True).iterrows():
        w = row["word"]
        q_idx = word_to_query_idx[w]
        matched_queries[i] = queries[q_idx]

    # Interaction features:
    # 1. Cosine similarity
    u_norm = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-9)
    q_norm = matched_queries / (np.linalg.norm(matched_queries, axis=1, keepdims=True) + 1e-9)
    cos_sim = np.sum(u_norm * q_norm, axis=1, keepdims=True)

    # 2. Absolute difference
    diff = np.abs(docs - matched_queries)

    # 3. Element-wise product
    prod = docs * matched_queries

    # Concatenate interaction features: diff, prod, cos_sim
    dense_features = np.hstack([diff, prod, cos_sim])
    return dense_features

# ---------------------------------------------------------------------------- #
# Candidate Models
# ---------------------------------------------------------------------------- #
class DenseDefinitionModel:
    def __init__(self, dense_features, seed=42):
        self.dense_features = dense_features
        self.scaler = StandardScaler()
        self.model = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=2000, C=0.1)

    def fit(self, df: pd.DataFrame):
        indices = df.index.to_numpy()
        X = self.dense_features[indices]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, df["label"])
        return self

    def predict(self, df: pd.DataFrame):
        return (self.predict_proba(df) >= 0.5).astype(int)

    def predict_proba(self, df: pd.DataFrame):
        indices = df.index.to_numpy()
        X = self.dense_features[indices]
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

class EarlyFusionModel:
    def __init__(self, dense_features, seed=42):
        self.dense_features = dense_features
        self.tfidf = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1, max_features=50000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True, min_df=1, max_features=50000, sublinear_tf=True)),
        ])
        self.scaler = StandardScaler()
        self.model = LinearSVC(class_weight="balanced", random_state=seed, max_iter=10000)

    def fit(self, df: pd.DataFrame):
        indices = df.index.to_numpy()
        X_sparse = self.tfidf.fit_transform(df["text"])
        X_dense = self.scaler.fit_transform(self.dense_features[indices])
        X_combined = hstack([X_sparse, csr_matrix(X_dense)])
        self.model.fit(X_combined, df["label"])
        return self

    def predict(self, df: pd.DataFrame):
        return (self.predict_proba(df) >= 0.5).astype(int)

    def predict_proba(self, df: pd.DataFrame):
        indices = df.index.to_numpy()
        X_sparse = self.tfidf.transform(df["text"])
        X_dense = self.scaler.transform(self.dense_features[indices])
        X_combined = hstack([X_sparse, csr_matrix(X_dense)])
        raw_scores = self.model.decision_function(X_combined)
        return 1.0 / (1.0 + np.exp(-raw_scores))

class LemmaBalancedSVM:
    """Baseline SVM with sample weights inversely balanced per lemma to prevent minority collapse."""
    def __init__(self, seed=42):
        self.pipe = Pipeline([
            ("tfidf", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1, max_features=50000, sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True, min_df=1, max_features=50000, sublinear_tf=True)),
            ])),
            ("model", LinearSVC(random_state=seed, max_iter=10000)),
        ])

    def fit(self, df: pd.DataFrame):
        # Compute sample weights inversely proportional to class counts within each word
        weights = np.ones(len(df), dtype=float)
        for w in df["word"].unique():
            mask_w = df["word"] == w
            sub = df[mask_w]
            n_w = len(sub)
            n_pos = (sub["label"] == 1).sum()
            n_neg = (sub["label"] == 0).sum()
            if n_pos > 0 and n_neg > 0:
                pos_weight = n_w / (2.0 * n_pos)
                neg_weight = n_w / (2.0 * n_neg)
                weights[mask_w & (df["label"] == 1)] = pos_weight
                weights[mask_w & (df["label"] == 0)] = neg_weight
        
        self.pipe.fit(df["text"], df["label"], model__sample_weight=weights)
        return self

    def predict(self, df: pd.DataFrame):
        return (self.predict_proba(df) >= 0.5).astype(int)

    def predict_proba(self, df: pd.DataFrame):
        raw_scores = self.pipe.decision_function(df["text"])
        return 1.0 / (1.0 + np.exp(-raw_scores))

def main():
    data = load_data().reset_index(drop=True)
    data["group"] = data["near_dup"]
    print("Loaded data:", len(data))

    print("\nExtracting XLM-R Definition Interaction Features [diff, prod, cos]...")
    dense_xlmr = build_definition_features(data, encoder="xlmr")
    print("Dense XLM-R feature shape:", dense_xlmr.shape)

    print("\nExtracting E5 Definition Interaction Features [diff, prod, cos]...")
    dense_e5 = build_definition_features(data, encoder="e5")
    print("Dense E5 feature shape:", dense_e5.shape)

    cache_path = ROOT / "results/stage3_intermediate_oof.npz"
    if cache_path.exists():
        print(f"Loading cached intermediate CV predictions from {cache_path}...")
        c_data = np.load(cache_path)
        res_a = {"oof_probabilities": c_data["a_probs"], "mean_official_f1": float(c_data["a_mean"]), "std_official_f1": float(c_data["a_std"])}
        res_b = {"oof_probabilities": c_data["b_probs"], "mean_official_f1": float(c_data["b_mean"]), "std_official_f1": float(c_data["b_std"])}
        res_c = {"oof_probabilities": c_data["c_probs"], "mean_official_f1": float(c_data["c_mean"]), "std_official_f1": float(c_data["c_std"])}
        res_d = {"oof_probabilities": c_data["d_probs"], "mean_official_f1": float(c_data["d_mean"]), "std_official_f1": float(c_data["d_std"])}
    else:
        # 1. Candidate A: Dense Definition Interaction Model (XLM-R)
        print("\n--- Evaluating Candidate A: Dense Definition Interaction Head (XLM-R) ---")
        res_a = run_cv(data, lambda s: DenseDefinitionModel(dense_xlmr, seed=s), n_repeats=5, n_splits=5)
        print(f"Candidate A Official F1: {res_a['mean_official_f1']:.6f} +/- {res_a['std_official_f1']:.6f}")

        # 2. Candidate B: Dense Definition Interaction Model (E5)
        print("\n--- Evaluating Candidate B: Dense Definition Interaction Head (E5) ---")
        res_b = run_cv(data, lambda s: DenseDefinitionModel(dense_e5, seed=s), n_repeats=5, n_splits=5)
        print(f"Candidate B Official F1: {res_b['mean_official_f1']:.6f} +/- {res_b['std_official_f1']:.6f}")

        # 3. Candidate C: Early Fusion (N-gram TF-IDF + E5 Definition Features)
        print("\n--- Evaluating Candidate C: Early Fusion (TF-IDF + E5 Definition Features) ---")
        res_c = run_cv(data, lambda s: EarlyFusionModel(dense_e5, seed=s), n_repeats=5, n_splits=5)
        print(f"Candidate C Official F1: {res_c['mean_official_f1']:.6f} +/- {res_c['std_official_f1']:.6f}")

        # 4. Candidate D: Lemma-Balanced Cost Weighting Linear SVM
        print("\n--- Evaluating Candidate D: Lemma-Balanced Cost Weighting Linear SVM ---")
        res_d = run_cv(data, lambda s: LemmaBalancedSVM(seed=s), n_repeats=5, n_splits=5)
        print(f"Candidate D Official F1: {res_d['mean_official_f1']:.6f} +/- {res_d['std_official_f1']:.6f}")

        np.savez(
            cache_path,
            a_probs=res_a["oof_probabilities"], a_mean=res_a["mean_official_f1"], a_std=res_a["std_official_f1"],
            b_probs=res_b["oof_probabilities"], b_mean=res_b["mean_official_f1"], b_std=res_b["std_official_f1"],
            c_probs=res_c["oof_probabilities"], c_mean=res_c["mean_official_f1"], c_std=res_c["std_official_f1"],
            d_probs=res_d["oof_probabilities"], d_mean=res_d["mean_official_f1"], d_std=res_d["std_official_f1"],
        )

    # Now calibrate each candidate with shrinkage=0.5
    print("\n--- Calibrating Candidates with Inner OOF Thresholds (shrinkage=0.5) ---")
    candidates = [
        ("Dense_XLM-R", res_a),
        ("Dense_E5", res_b),
        ("Early_Fusion_E5", res_c),
        ("Lemma_Balanced_SVM", res_d),
    ]

    # Load baseline OOF probabilities from Stage 1 for paired bootstrap
    stage1_file = ROOT / "results/stage1_oof_predictions.npz"
    s1_data = np.load(stage1_file)
    baseline_probs = s1_data["svm_probs"]
    labels = data["label"].to_numpy().astype(int)
    words = data["word"].to_numpy().astype(str)

    # Compute calibrated baseline predictions across repeats
    n_repeats = 5
    baseline_cal_preds = np.zeros_like(baseline_probs, dtype=int)
    for r in range(n_repeats):
        cal = best_threshold(labels, baseline_probs[r], words, shrinkage=0.5)
        baseline_cal_preds[r] = apply_calibrated_thresholds(words, baseline_probs[r], cal)

    results_summary = {}
    for name, res in candidates:
        cand_probs = res["oof_probabilities"]
        cand_cal_preds = np.zeros_like(cand_probs, dtype=int)
        for r in range(n_repeats):
            cal = best_threshold(labels, cand_probs[r], words, shrinkage=0.5)
            cand_cal_preds[r] = apply_calibrated_thresholds(words, cand_probs[r], cal)
        
        # Paired bootstrap vs Calibrated Baseline SVM
        boot_res = paired_bootstrap(data, cand_cal_preds, baseline_cal_preds, n_boot=10000)
        print(f"\n{name} (Calibrated): Official F1 = {boot_res['mean_a']:.6f}")
        print(f"  vs Calibrated Baseline ({boot_res['mean_b']:.6f}): Diff = {boot_res['point_diff']:+.6f}")
        print(f"  95% CI: [{boot_res['ci95_lo']:+.6f}, {boot_res['ci95_hi']:+.6f}], P(A not better) = {boot_res['p_a_not_better']:.4f}")
        print(f"  ADOPTED: {boot_res['adopted']}")

        results_summary[name] = {
            "uncalibrated_mean_f1": res["mean_official_f1"],
            "uncalibrated_std_f1": res["std_official_f1"],
            "calibrated_mean_f1": boot_res["mean_a"],
            "point_diff": boot_res["point_diff"],
            "ci95_lo": boot_res["ci95_lo"],
            "ci95_hi": boot_res["ci95_hi"],
            "p_a_not_better": boot_res["p_a_not_better"],
            "adopted": boot_res["adopted"],
        }

    import json
    with open(ROOT / "results/stage3_definition_matching_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    # Save diagnostics for Candidate A (Dense XLM-R)
    cand_a_probs = res_a["oof_probabilities"][0]
    cal_a = best_threshold(labels, cand_a_probs, words, shrinkage=0.5)
    preds_a = apply_calibrated_thresholds(words, cand_a_probs, cal_a)
    diag_a = diagnostics(data, preds_a, cand_a_probs)
    diag_a.to_csv(ROOT / "results/diagnostics_dense_definition_xlmr.csv", index=False)
    print("\nSaved diagnostics to results/diagnostics_dense_definition_xlmr.csv")

if __name__ == "__main__":
    main()
