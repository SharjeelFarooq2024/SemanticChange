"""Supervised hybrid sense reranking with fold-fitted text/retrieval features.

Frozen, label-free BM25/E5 features cover the complete lemma pool. Vocabulary,
scaling, the linear head and decision calibration are fitted on training data.
"""
from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import LinearSVC

from Assignment_2.src.retrieval.bm25_retriever import BM25Retriever
from Assignment_3.src.evaluation import fast_official_f1
from Assignment_3.src.runtime import ROOT, load_data, resolve_model, sha256

RETRIEVAL_FEATURE_NAMES = ["bm25_score", "bm25_pool_minmax", "e5_cosine", "e5_pool_zscore",
    "bm25_rank_percentile", "e5_rank_percentile", "rrf_score_times_60",
    "component_rank_disagreement", "lexical_semantic_interaction", "definition_token_overlap"]


@lru_cache(maxsize=2)
def resources(reference_relative):
    folder = ROOT / reference_relative
    raw, definitions = load_data()
    definitions = definitions.set_index("word").definition.to_dict()
    cache_path = folder / "artifacts/e5_small.npz"
    meta = json.loads((folder / "artifacts/e5_small.json").read_text(encoding="utf-8"))
    if sha256(cache_path) != meta["cache_sha256"]:
        raise ValueError("E5 representation cache integrity failed")
    with np.load(cache_path) as cache:
        if list(cache["doc_ids"]) != raw.sentence_id.tolist():
            raise ValueError("E5 cache identity mismatch")
        vectors = cache["documents"].copy()
        query_vectors = {str(word): vector.copy() for word, vector in zip(cache["query_words"], cache["queries"])}
    bm25 = BM25Retriever.load(folder / "artifacts/bm25_index.pkl")
    frames = {s: pd.read_csv(folder / f"rankings/{s}_full.csv") for s in ["bm25", "dense"]}
    scores = {s: frame.set_index(["query_id", "doc_id"]) for s, frame in frames.items()}
    pools = {word: {s: frame.loc[frame.query_id == word, "score"].to_numpy()
                    for s, frame in frames.items()} for word in definitions}
    return {"raw": raw.set_index("sentence_id"), "positions": {sid:i for i,sid in enumerate(raw.sentence_id)},
            "vectors": vectors, "queries": query_vectors, "bm25": bm25,
            "scores": scores, "pools": pools, "definitions": definitions,
            "profile": meta["identity"]["model"]}


def query_document_features(frame, reference_relative):
    store = resources(reference_relative)
    numeric = np.zeros((len(frame), len(RETRIEVAL_FEATURE_NAMES)), dtype=np.float64)
    embedding = np.empty((len(frame), store["vectors"].shape[1]*3), dtype=np.float32)
    unknown = []
    vectors = np.empty((len(frame), store["vectors"].shape[1]), dtype=np.float32)
    for i, row in enumerate(frame.itertuples()):
        if row.word not in store["definitions"]:
            raise ValueError(f"Unknown target lemma {row.word!r}; this checkpoint uses the ten provided definitions")
        sid = getattr(row, "sentence_id", None)
        if sid in store["positions"] and store["raw"].loc[sid, "text"] == row.text and store["raw"].loc[sid, "word"] == row.word:
            vectors[i] = store["vectors"][store["positions"][sid]]
        else:
            unknown.append(i)
    if unknown:
        import torch
        from transformers import AutoModel, AutoTokenizer
        profile = store["profile"]
        path = resolve_model(profile)
        torch.set_num_threads(2)
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        model = AutoModel.from_pretrained(str(path), local_files_only=True).eval().requires_grad_(False)
        for offset in range(0, len(unknown), 4):
            indices = unknown[offset:offset+4]
            inputs = tokenizer([profile["passage_prefix"]+frame.iloc[i].text for i in indices], return_tensors="pt",
                               padding=True, truncation=True, max_length=512)
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
                vectors[indices] = torch.nn.functional.normalize(pooled, dim=1).numpy()
        del model, tokenizer
    unknown_set = set(unknown)
    for i, row in enumerate(frame.itertuples()):
        q = store["queries"][row.word]
        doc = vectors[i]
        pools = store["pools"][row.word]
        if i not in unknown_set:
            key = (row.word, row.sentence_id)
            b = float(store["scores"]["bm25"].loc[key, "score"])
            d = float(store["scores"]["dense"].loc[key, "score"])
            rb = int(store["scores"]["bm25"].loc[key, "rank"])
            rd = int(store["scores"]["dense"].loc[key, "rank"])
        else:
            tokens = store["bm25"].tokenizer.tokenize(row.text)
            tf = Counter(tokens)
            index = store["bm25"]._index
            b = sum(index.idf.get(token, 0)*tf[token]*(index.k1+1)/
                    (tf[token]+index.k1*(1-index.b+index.b*len(tokens)/index.avgdl))
                    for token in store["bm25"].tokenizer.tokenize(store["definitions"][row.word]) if tf[token])
            d = float(doc @ q)
            rb = 1+int((pools["bm25"] > b).sum())
            rd = 1+int((pools["dense"] > d).sum())
        n = len(pools["bm25"])
        bmin, bmax = pools["bm25"].min(), pools["bm25"].max()
        bn = (b-bmin)/max(bmax-bmin, 1e-12)
        dz = (d-pools["dense"].mean())/max(pools["dense"].std(), 1e-12)
        pb, pdense = (rb-1)/max(n-1, 1), (rd-1)/max(n-1, 1)
        definition_tokens = set(store["bm25"].tokenizer.tokenize(store["definitions"][row.word]))
        text_tokens = set(store["bm25"].tokenizer.tokenize(row.text))
        overlap = len(definition_tokens & text_tokens)/max(len(definition_tokens), 1)
        numeric[i] = [b, bn, d, dz, pb, pdense, 60/(60+rb)+60/(60+rd), abs(pb-pdense), bn*d, overlap]
        embedding[i] = np.concatenate([doc, doc*q, abs(doc-q)])
    return numeric, normalize(embedding, norm="l2", copy=False)


class HybridFeatureBuilder:
    def __init__(self, config):
        self.config = config

    def fit_transform(self, frame):
        settings = self.config["text_features"]
        self.word = TfidfVectorizer(ngram_range=tuple(settings["word_ngram_range"]),
            max_features=settings["max_features_per_view"], min_df=settings["min_df"], sublinear_tf=settings["sublinear_tf"])
        self.char = TfidfVectorizer(analyzer="char", ngram_range=tuple(settings["char_ngram_range"]),
            max_features=settings["max_features_per_view"], min_df=settings["min_df"], sublinear_tf=settings["sublinear_tf"])
        context = sparse.hstack([self.word.fit_transform(frame.text), self.char.fit_transform(frame.text)], format="csr")
        numeric, embedding = query_document_features(frame, self.config["reference_run"])
        self.scaler = StandardScaler().fit(numeric)
        return {"context": context, "retrieval": sparse.csr_matrix(self.scaler.transform(numeric)),
                "embedding": sparse.csr_matrix(embedding)}

    def transform(self, frame):
        context = sparse.hstack([self.word.transform(frame.text), self.char.transform(frame.text)], format="csr")
        numeric, embedding = query_document_features(frame, self.config["reference_run"])
        return {"context": context, "retrieval": sparse.csr_matrix(self.scaler.transform(numeric)),
                "embedding": sparse.csr_matrix(embedding)}


def combine_blocks(blocks, setting):
    parts = [blocks[name]*setting[f"{name}_weight"] for name in ["context", "retrieval", "embedding"] if setting[f"{name}_weight"] > 0]
    if not parts:
        raise ValueError("No active model features")
    return sparse.hstack(parts, format="csr")


def make_head(config, setting, seed):
    return LinearSVC(C=setting["C"], class_weight=config["class_weight"], max_iter=config["max_iter"], random_state=seed)


def threshold_candidates(margins, n_quantiles):
    margins = np.asarray(margins)
    return np.unique(np.r_[0.0, margins.min()-1e-6, margins.max()+1e-6,
                           np.quantile(margins, np.linspace(.05, .95, n_quantiles))])


def fit_thresholds(words, gold, margins, shrinkage, n_quantiles=19):
    words, gold, margins = map(np.asarray, (words, gold, margins))
    choices = [(float(t), fast_official_f1(words, gold, (margins >= t).astype(int))) for t in threshold_candidates(margins, n_quantiles)]
    global_t, _ = max(choices, key=lambda item: (item[1], -abs(item[0])))
    per_word = {}
    if shrinkage > 0:
        for word in sorted(np.unique(words)):
            keep = words == word
            options = np.unique(np.r_[global_t, threshold_candidates(margins[keep], n_quantiles)])
            t, _ = max([(float(t), fast_official_f1(words[keep], gold[keep], (margins[keep] >= t).astype(int))) for t in options],
                       key=lambda item: (item[1], -abs(item[0]-global_t)))
            per_word[str(word)] = float(global_t+shrinkage*(t-global_t))
    return {"global": float(global_t), "per_word": per_word, "word_shrinkage": float(shrinkage)}


def apply_thresholds(words, margins, calibration):
    thresholds = np.array([calibration["per_word"].get(str(word), calibration["global"]) for word in words])
    return (np.asarray(margins) >= thresholds).astype(int)


class ImprovedHybridModel:
    """Fitted head plus preprocessing and calibrated per-lemma decision rule."""
    def __init__(self, builder, head, setting, calibration):
        self.builder, self.head = builder, head
        self.setting, self.calibration = setting, calibration

    def decision_function(self, frame):
        return self.head.decision_function(combine_blocks(self.builder.transform(frame), self.setting))

    def predict(self, frame):
        return apply_thresholds(frame.word, self.decision_function(frame), self.calibration)
