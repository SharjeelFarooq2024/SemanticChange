"""Stage 3: Definition Matching, Interaction Modeling & Shuffled-Definition Ablations.
Evaluated strictly via inner-fold threshold calibration under StratifiedGroupKFold (near_dup_groups).
"""
import hashlib
import json
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from eval_harness import (
    official_f1,
    best_threshold,
    apply_calibrated_thresholds,
    paired_bootstrap,
    diagnostics,
)
from scripts.stage1_measure import load_data, BaselineSVMWrapper


def _interaction_vectors(u: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute [|u - q|, u * q, cos(u, q)] interaction features."""
    u_norm = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-9)
    q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    cos_sim = np.sum(u_norm * q_norm, axis=1, keepdims=True)
    return np.hstack([np.abs(u - q), u * q, cos_sim])


class DenseFeatureStore:
    """Key-indexed dense interaction store resolving rows safely regardless of DataFrame index.

    Looks up precomputed embeddings by (word, text) or (word, sentence_id) against the
    verified artifact caches, and computes unseen rows on the fly when local encoder weights exist.
    """

    def __init__(
        self,
        encoder: str = "xlmr",
        mode: str = "docs",
        shuffle_definitions: bool = False,
        seed: int = 42,
        root: Path = ROOT,
    ):
        self.encoder = encoder
        self.mode = mode
        self.shuffle_definitions = shuffle_definitions
        self.seed = seed
        self.root = Path(root)

        artifact_name = "xlmr_base.npz" if encoder == "xlmr" else "e5_small.npz"
        cache_path = self.root / "Assignment_3/runs/completed/artifacts" / artifact_name
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing dense representation cache: {cache_path}. "
                "Generate via Assignment_3/scripts/prepare_representations.py or use --model baseline_svm."
            )

        with np.load(cache_path) as cache:
            docs = cache["documents"].astype(np.float32)
            queries = cache["queries"].astype(np.float32)
            doc_ids = [str(x) for x in cache["doc_ids"]]
            query_words = [str(x) for x in cache["query_words"]]
            clf_ctx = (
                cache["classifier_contexts"].astype(np.float32)
                if "classifier_contexts" in cache
                else None
            )

        if shuffle_definitions:
            rng = np.random.default_rng(seed)
            queries = queries[rng.permutation(len(queries))]

        self.query_map = {w: queries[i] for i, w in enumerate(query_words)}

        # Map exact text & sentence_id identities from usages.jsonl to cache indices
        usages_path = self.root / "data/raw/dev/SV/usages.jsonl"
        self.sid_word_to_idx: dict[tuple[str, str], int] = {}
        self.text_word_to_idx: dict[tuple[str, str], int] = {}
        if usages_path.exists():
            df_raw = pd.read_json(usages_path, lines=True)
            if len(df_raw) == len(doc_ids) and df_raw["sentence_id"].astype(str).tolist() == doc_ids:
                for idx, row in enumerate(df_raw.itertuples(index=False)):
                    w_str = str(row.word)
                    self.sid_word_to_idx[(w_str, str(row.sentence_id))] = idx
                    self.text_word_to_idx[(w_str, str(row.text))] = idx

        self._docs = docs
        self._clf_ctx = clf_ctx
        self._live_cache: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}

    def has_all_cached(self, df: pd.DataFrame) -> bool:
        """Check whether all rows in df are covered by the precomputed artifact cache."""
        for row in df.itertuples(index=False):
            w = str(row.word)
            if w not in self.query_map:
                return False
            sid = str(getattr(row, "sentence_id", ""))
            txt = str(row.text)
            if (w, txt) not in self.text_word_to_idx and (w, sid) not in self.sid_word_to_idx:
                return False
        return True

    def _encode_unseen(self, missing_rows: list[tuple[int, str, str]]) -> None:
        """Encode unseen sentences locally when model weights are cached on disk."""
        if self.encoder != "xlmr":
            raise FileNotFoundError(
                "On-the-fly dense encoding is supported only for 'xlmr' when local model weights exist."
            )
        meta_path = self.root / "Assignment_3/runs/completed/artifacts/xlmr_base.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing model metadata: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        from Assignment_3.src.runtime import resolve_model
        from src.eda.embedding_analysis import build_usage_context, encode_texts

        model_path = resolve_model(meta["identity"]["model"], allow_download=False)
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(2)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModel.from_pretrained(str(model_path), local_files_only=True).eval().requires_grad_(False)

        raw_texts = [t for _, _, t in missing_rows]
        max_len = int(meta["identity"].get("max_length", 512))
        doc_vecs = encode_texts(tokenizer, model, raw_texts, torch.device("cpu"), batch_size=4, max_length=max_len)
        doc_vecs = doc_vecs / np.clip(np.linalg.norm(doc_vecs, axis=1, keepdims=True), 1e-12, None)

        ctx_vecs = None
        if self.mode in ("ctx", "both"):
            ctx_texts = [build_usage_context({"word": w, "text": t}) for _, w, t in missing_rows]
            ctx_vecs = encode_texts(tokenizer, model, ctx_texts, torch.device("cpu"), batch_size=4, max_length=max_len)
            ctx_vecs = ctx_vecs / np.clip(np.linalg.norm(ctx_vecs, axis=1, keepdims=True), 1e-12, None)

        for pos, (_, w, txt) in enumerate(missing_rows):
            key = hashlib.sha256(f"{w}\t{txt}".encode("utf-8")).hexdigest()
            self._live_cache[key] = (doc_vecs[pos], ctx_vecs[pos] if ctx_vecs is not None else None)

    def transform_df(self, df: pd.DataFrame) -> np.ndarray:
        """Build key-aligned interaction feature matrix for an arbitrary DataFrame."""
        n = len(df)
        d_dim = next(iter(self.query_map.values())).shape[0]
        u_docs = np.empty((n, d_dim), dtype=np.float32)
        u_ctx = np.empty((n, d_dim), dtype=np.float32) if self.mode in ("ctx", "both") else None
        q_vecs = np.empty((n, d_dim), dtype=np.float32)
        missing: list[tuple[int, str, str]] = []

        for i, row in enumerate(df.itertuples(index=False)):
            w = str(row.word)
            if w not in self.query_map:
                raise KeyError(f"Unknown target word {w!r}; no definition embedding cached for this lemma.")
            q_vecs[i] = self.query_map[w]
            sid = str(getattr(row, "sentence_id", ""))
            txt = str(row.text)
            idx = self.text_word_to_idx.get((w, txt))
            if idx is None:
                idx = self.sid_word_to_idx.get((w, sid))
            if idx is not None:
                u_docs[i] = self._docs[idx]
                if u_ctx is not None:
                    u_ctx[i] = self._clf_ctx[idx] if self._clf_ctx is not None else self._docs[idx]
            else:
                key = hashlib.sha256(f"{w}\t{txt}".encode("utf-8")).hexdigest()
                if key in self._live_cache:
                    d_v, c_v = self._live_cache[key]
                    u_docs[i] = d_v
                    if u_ctx is not None:
                        u_ctx[i] = c_v if c_v is not None else d_v
                else:
                    missing.append((i, w, txt))

        if missing:
            self._encode_unseen(missing)
            for i, w, txt in missing:
                key = hashlib.sha256(f"{w}\t{txt}".encode("utf-8")).hexdigest()
                d_v, c_v = self._live_cache[key]
                u_docs[i] = d_v
                if u_ctx is not None:
                    u_ctx[i] = c_v if c_v is not None else d_v

        if self.mode == "docs":
            return _interaction_vectors(u_docs, q_vecs)
        if self.mode == "ctx":
            assert u_ctx is not None
            return _interaction_vectors(u_ctx, q_vecs)
        if self.mode == "both":
            assert u_ctx is not None
            return np.hstack([_interaction_vectors(u_docs, q_vecs), _interaction_vectors(u_ctx, q_vecs)])
        raise ValueError(f"Unsupported dense interaction mode: {self.mode}")


def build_definition_features(data: pd.DataFrame, encoder: str = "xlmr", mode: str = "docs") -> np.ndarray:
    """Build key-aligned dense interaction features for backward compatibility."""
    store = DenseFeatureStore(encoder=encoder, mode=mode)
    return store.transform_df(data)


class DenseDefinitionModel:
    """Regularized Logistic Regression head over dense usage-definition interactions."""

    def __init__(
        self,
        feature_source,
        seed: int = 42,
        C: float = 0.1,
        per_lemma_weights: bool = False,
    ):
        self.feature_source = feature_source
        self.C = C
        self.per_lemma_weights = per_lemma_weights
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            class_weight=None if per_lemma_weights else "balanced",
            random_state=seed,
            max_iter=2000,
            C=C,
        )

    def _get_X(self, df: pd.DataFrame) -> np.ndarray:
        if isinstance(self.feature_source, DenseFeatureStore):
            return self.feature_source.transform_df(df)
        return self.feature_source[df.index.to_numpy()]

    def fit(self, df: pd.DataFrame):
        X_scaled = self.scaler.fit_transform(self._get_X(df))
        if self.per_lemma_weights:
            weights = np.ones(len(df), dtype=float)
            words = df["word"].astype(str).to_numpy()
            labels = df["label"].to_numpy().astype(int)
            for w in np.unique(words):
                mask = words == w
                n_w = int(mask.sum())
                n_pos = int((labels[mask] == 1).sum())
                n_neg = n_w - n_pos
                if n_pos > 0 and n_neg > 0:
                    weights[mask & (labels == 1)] = n_w / (2.0 * n_pos)
                    weights[mask & (labels == 0)] = n_w / (2.0 * n_neg)
            self.model.fit(X_scaled, labels, sample_weight=np.clip(weights, 0.25, 10.0))
        else:
            self.model.fit(X_scaled, df["label"])
        return self

    def predict(self, df: pd.DataFrame):
        return (self.predict_proba(df) >= 0.5).astype(int)

    def predict_proba(self, df: pd.DataFrame):
        X_scaled = self.scaler.transform(self._get_X(df))
        return self.model.predict_proba(X_scaled)[:, 1]


class EarlyFusionModel:
    def __init__(self, feature_source, seed: int = 42):
        self.feature_source = feature_source
        self.tfidf = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1, max_features=50000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True, min_df=1, max_features=50000, sublinear_tf=True)),
        ])
        self.scaler = StandardScaler()
        self.model = LinearSVC(class_weight="balanced", random_state=seed, max_iter=10000)

    def _get_dense(self, df: pd.DataFrame) -> np.ndarray:
        if isinstance(self.feature_source, DenseFeatureStore):
            return self.feature_source.transform_df(df)
        return self.feature_source[df.index.to_numpy()]

    def fit(self, df: pd.DataFrame):
        X_sparse = self.tfidf.fit_transform(df["text"])
        X_dense = self.scaler.fit_transform(self._get_dense(df))
        self.model.fit(hstack([X_sparse, csr_matrix(X_dense)]), df["label"])
        return self

    def predict(self, df: pd.DataFrame):
        return (self.predict_proba(df) >= 0.5).astype(int)

    def predict_proba(self, df: pd.DataFrame):
        X_sparse = self.tfidf.transform(df["text"])
        X_dense = self.scaler.transform(self._get_dense(df))
        raw_scores = self.model.decision_function(hstack([X_sparse, csr_matrix(X_dense)]))
        return 1.0 / (1.0 + np.exp(-raw_scores))


class LemmaBalancedSVM:
    """Baseline SVM with sample weights inversely balanced per lemma to prevent minority collapse."""

    def __init__(self, seed: int = 42):
        self.pipe = Pipeline([
            ("tfidf", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1, max_features=50000, sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True, min_df=1, max_features=50000, sublinear_tf=True)),
            ])),
            ("model", LinearSVC(random_state=seed, max_iter=10000)),
        ])

    def fit(self, df: pd.DataFrame):
        weights = np.ones(len(df), dtype=float)
        words = df["word"].astype(str).to_numpy()
        labels = df["label"].to_numpy().astype(int)
        for w in np.unique(words):
            mask_w = words == w
            n_w = int(mask_w.sum())
            n_pos = int((labels[mask_w] == 1).sum())
            n_neg = n_w - n_pos
            if n_pos > 0 and n_neg > 0:
                weights[mask_w & (labels == 1)] = n_w / (2.0 * n_pos)
                weights[mask_w & (labels == 0)] = n_w / (2.0 * n_neg)
        self.pipe.fit(df["text"], labels, model__sample_weight=weights)
        return self

    def predict(self, df: pd.DataFrame):
        return (self.predict_proba(df) >= 0.5).astype(int)

    def predict_proba(self, df: pd.DataFrame):
        raw_scores = self.pipe.decision_function(df["text"])
        return 1.0 / (1.0 + np.exp(-raw_scores))


class EnsembleDenseSVM:
    """Probability blend between DenseDefinitionModel and LemmaBalancedSVM."""

    def __init__(self, feature_source, alpha: float = 0.8, C: float = 0.05, per_lemma_weights: bool = True, seed: int = 42):
        self.dense = DenseDefinitionModel(feature_source, seed=seed, C=C, per_lemma_weights=per_lemma_weights)
        self.svm = LemmaBalancedSVM(seed=seed)
        self.alpha = alpha

    def fit(self, df: pd.DataFrame):
        self.dense.fit(df)
        self.svm.fit(df)
        return self

    def predict(self, df: pd.DataFrame):
        return (self.predict_proba(df) >= 0.5).astype(int)

    def predict_proba(self, df: pd.DataFrame):
        return self.alpha * self.dense.predict_proba(df) + (1.0 - self.alpha) * self.svm.predict_proba(df)


def run_strict_calibrated_cv(
    data: pd.DataFrame,
    model_factory_fn,
    shrinkage: float = 0.5,
    n_splits: int = 5,
    n_repeats: int = 5,
    inner_splits: int = 3,
    base_seed: int = 42,
):
    """Run repeated StratifiedGroupKFold with strict inner-fold threshold calibration."""
    df = data.reset_index(drop=True).copy()
    n_samples = len(df)
    labels = df["label"].to_numpy().astype(int)
    words = df["word"].to_numpy().astype(str)
    groups = df["near_dup"].to_numpy()
    strat_keys = df["word"].astype(str) + "_" + df["label"].astype(str)

    uncal_scores, cal_scores = [], []
    oof_cal_preds = np.zeros((n_repeats, n_samples), dtype=int)
    oof_probs = np.zeros((n_repeats, n_samples), dtype=float)

    for r in range(n_repeats):
        seed = base_seed + r * 17
        outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        rep_uncal = np.full(n_samples, -1, dtype=int)
        rep_cal = np.full(n_samples, -1, dtype=int)
        rep_prob = np.full(n_samples, np.nan, dtype=float)

        for fold, (tr_idx, te_idx) in enumerate(outer.split(df, strat_keys, groups)):
            tr_df = df.iloc[tr_idx].copy()
            te_df = df.iloc[te_idx].copy()

            in_strat = tr_df["word"].astype(str) + "_" + tr_df["label"].astype(str)
            in_grp = tr_df["near_dup"].to_numpy()
            inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed + fold * 10)
            in_oof = np.full(len(tr_df), np.nan, dtype=float)

            for f_i, v_i in inner.split(tr_df, in_strat, in_grp):
                m_in = model_factory_fn(seed + fold * 10 + 1)
                m_in.fit(tr_df.iloc[f_i])
                in_oof[v_i] = m_in.predict_proba(tr_df.iloc[v_i])

            cal = best_threshold(tr_df["label"].to_numpy(), in_oof, tr_df["word"].to_numpy(), shrinkage=shrinkage)
            m_full = model_factory_fn(seed + fold)
            m_full.fit(tr_df)
            probs = m_full.predict_proba(te_df)

            rep_prob[te_idx] = probs
            rep_uncal[te_idx] = (probs >= 0.5).astype(int)
            rep_cal[te_idx] = apply_calibrated_thresholds(te_df["word"].to_numpy(), probs, cal)

        uncal_scores.append(official_f1(words, labels, rep_uncal))
        cal_scores.append(official_f1(words, labels, rep_cal))
        oof_cal_preds[r] = rep_cal
        oof_probs[r] = rep_prob

    return {
        "uncal_mean_official_f1": float(np.mean(uncal_scores)),
        "uncal_std_official_f1": float(np.std(uncal_scores, ddof=1)),
        "cal_mean_official_f1": float(np.mean(cal_scores)),
        "cal_std_official_f1": float(np.std(cal_scores, ddof=1)),
        "oof_calibrated_predictions": oof_cal_preds,
        "oof_probabilities": oof_probs,
    }


def main():
    data = load_data().reset_index(drop=True)
    data["group"] = data["near_dup"]
    print(f"Loaded data: {len(data)} rows across {data['word'].nunique()} lemmata and {data['period_label'].nunique()} periods.")

    store_xlmr = DenseFeatureStore(encoder="xlmr", mode="docs")
    store_xlmr_shuf = DenseFeatureStore(encoder="xlmr", mode="docs", shuffle_definitions=True, seed=42)
    store_e5 = DenseFeatureStore(encoder="e5", mode="docs")

    # Baseline calibrated comparison (strict inner OOF from Stage 2)
    print("\n--- Computing Strict Inner-OOF Reference: Calibrated Baseline SVM (5x5 CV) ---")
    ref_svm = run_strict_calibrated_cv(data, lambda s: BaselineSVMWrapper(seed=s), shrinkage=0.5)
    print(f"Calibrated Baseline SVM (strict inner OOF): {ref_svm['cal_mean_official_f1']:.6f} +/- {ref_svm['cal_std_official_f1']:.6f}")

    candidates = [
        ("Dense_XLM-R", lambda s: DenseDefinitionModel(store_xlmr, seed=s, C=0.1)),
        ("Dense_XLM-R_ShuffledDef_Ablation", lambda s: DenseDefinitionModel(store_xlmr_shuf, seed=s, C=0.1)),
        ("Dense_E5", lambda s: DenseDefinitionModel(store_e5, seed=s, C=0.1)),
        ("Lemma_Balanced_SVM", lambda s: LemmaBalancedSVM(seed=s)),
    ]

    results_summary = {}
    for name, factory in candidates:
        print(f"\n--- Evaluating {name} (5x5 Strict Inner-OOF Calibrated CV) ---")
        res = run_strict_calibrated_cv(data, factory, shrinkage=0.5)
        boot = paired_bootstrap(data, res["oof_calibrated_predictions"], ref_svm["oof_calibrated_predictions"], n_boot=10000)
        print(f"  Uncalibrated: {res['uncal_mean_official_f1']:.6f} +/- {res['uncal_std_official_f1']:.6f}")
        print(f"  Calibrated:   {res['cal_mean_official_f1']:.6f} +/- {res['cal_std_official_f1']:.6f}")
        print(f"  vs Calibrated SVM ({boot['mean_b']:.6f}): Diff = {boot['point_diff']:+.6f}, 95% CI = [{boot['ci95_lo']:+.6f}, {boot['ci95_hi']:+.6f}], P = {boot['p_a_not_better']:.4f}, Adopted = {boot['adopted']}")

        results_summary[name] = {
            "uncalibrated_mean_f1": res["uncal_mean_official_f1"],
            "uncalibrated_std_f1": res["uncal_std_official_f1"],
            "calibrated_mean_f1": res["cal_mean_official_f1"],
            "calibrated_std_f1": res["cal_std_official_f1"],
            "point_diff_vs_cal_svm": boot["point_diff"],
            "ci95_lo": boot["ci95_lo"],
            "ci95_hi": boot["ci95_hi"],
            "p_a_not_better": boot["p_a_not_better"],
            "adopted": boot["adopted"],
        }

        if name == "Dense_XLM-R":
            diag = diagnostics(data, res["oof_calibrated_predictions"][0], res["oof_probabilities"][0])
            diag.to_csv(ROOT / "results/diagnostics_dense_definition_xlmr.csv", index=False)

    with open(ROOT / "results/stage3_definition_matching_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print("\nSaved Stage 3 strict inner-OOF results to results/stage3_definition_matching_results.json")


if __name__ == "__main__":
    main()
