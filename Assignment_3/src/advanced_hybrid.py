"""Word-aware experts and nonlinear multi-view hybrid classifiers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize
from sklearn.svm import LinearSVC, SVC
from threadpoolctl import threadpool_limits

from Assignment_3.src.improved_hybrid import HybridFeatureBuilder, apply_thresholds
from Assignment_3.src.runtime import ROOT, load_data


def metadata_frame(frame):
    result = pd.DataFrame(index=frame.index)
    for column, default in [("word", ""), ("pos", "NN"), ("period_label", "")]:
        result[column] = frame[column].fillna(default).astype(str) if column in frame else default
    return result


def word_class_weights(words, labels):
    keys = pd.DataFrame({"word": np.asarray(words), "label": np.asarray(labels)})
    counts = keys.groupby(["word", "label"]).size()
    weights = np.array([1/counts.loc[(w,int(y))] for w,y in zip(keys.word, keys.label)])
    weights /= weights.mean()
    return np.clip(weights, .1, 20.)


class AdvancedBuilder:
    def __init__(self, config):
        self.config = config

    def _xlmr(self, frame):
        folder = ROOT / self.config["reference_run"]
        raw, _ = load_data()
        lookup = raw.set_index("sentence_id")
        with np.load(folder / "artifacts/xlmr_base.npz") as cache:
            cached_contexts = cache["classifier_contexts"]
            position = {str(s):i for i,s in enumerate(cache["doc_ids"])}
            vectors = np.empty((len(frame), cached_contexts.shape[1]), dtype=np.float32)
            unknown = []
            for i,row in enumerate(frame.itertuples()):
                sid = str(row.sentence_id)
                if sid in position and lookup.loc[sid,"text"] == row.text and lookup.loc[sid,"word"] == row.word:
                    vectors[i] = cached_contexts[position[sid]]
                else:
                    unknown.append(i)
        if unknown:
            import json
            import torch
            from transformers import AutoTokenizer, AutoModel
            from src.eda.embedding_analysis import build_usage_context, encode_texts
            from Assignment_3.src.runtime import resolve_model
            metadata = json.loads((folder / "artifacts/xlmr_base.json").read_text(encoding="utf-8"))
            path = resolve_model(metadata["identity"]["model"])
            torch.set_num_threads(2)
            tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            model = AutoModel.from_pretrained(str(path), local_files_only=True).eval().requires_grad_(False)
            vectors[unknown] = encode_texts(tokenizer, model, [build_usage_context(frame.iloc[i]) for i in unknown],
                                           torch.device("cpu"), batch_size=4, max_length=metadata["identity"]["max_length"])
            del model, tokenizer
        return normalize(vectors)

    def fit_transform(self, frame, seed):
        self.base = HybridFeatureBuilder(self.config)
        blocks = self.base.fit_transform(frame)
        self.onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        meta = self.onehot.fit_transform(metadata_frame(frame))
        with threadpool_limits(limits=2):
            n_components = min(self.config["svd_dimensions"], blocks["context"].shape[0]-1, blocks["context"].shape[1]-1)
            self.svd = TruncatedSVD(n_components=n_components, n_iter=5, random_state=seed)
            latent = normalize(self.svd.fit_transform(blocks["context"]))
        self.dense_scaler = StandardScaler()
        # Every view remains present in the nonlinear model, including lexical
        # and semantic retrieval evidence. Scaling is fitted on this fold only.
        xlmr = self._xlmr(frame)
        dense = np.hstack([latent, blocks["embedding"].toarray(), xlmr,
                           blocks["retrieval"].toarray(), meta])
        widths = [latent.shape[1], blocks["embedding"].shape[1], xlmr.shape[1], blocks["retrieval"].shape[1], meta.shape[1]]
        self.dense_weights = np.concatenate([np.full(width, weight/np.sqrt(max(width,1)))
                                            for width,weight in zip(widths, [1., .5, .5, .25, .5])])
        dense = self.dense_scaler.fit_transform(dense)*self.dense_weights
        return self._combine(blocks, meta, dense)

    def transform(self, frame):
        blocks = self.base.transform(frame)
        meta = self.onehot.transform(metadata_frame(frame))
        latent = normalize(self.svd.transform(blocks["context"]))
        dense = np.hstack([latent, blocks["embedding"].toarray(), self._xlmr(frame),
                           blocks["retrieval"].toarray(), meta])
        return self._combine(blocks, meta, self.dense_scaler.transform(dense)*self.dense_weights)

    @staticmethod
    def _combine(blocks, meta, dense):
        linear = sparse.hstack([blocks["context"], blocks["embedding"]*.5, blocks["retrieval"]*.25,
                                sparse.csr_matrix(meta)*.25], format="csr")
        return {"linear": linear, "dense": normalize(dense)}


class AdvancedHead:
    def __init__(self, setting, seed):
        self.setting, self.seed = setting, seed

    def fit(self, views, frame):
        setting = self.setting
        y, words = frame.label_value.to_numpy(), frame.word.to_numpy()
        weights = word_class_weights(words, y)
        kind = setting["kind"]
        if kind in ["linear", "experts"]:
            self.global_model = LinearSVC(C=setting["C"], class_weight=None, max_iter=20000, random_state=self.seed)
            self.global_model.fit(views["linear"], y, sample_weight=weights)
            self.experts = {}
            if kind == "experts":
                for word in np.unique(words):
                    keep = words == word
                    if len(np.unique(y[keep])) < 2:
                        self.experts[str(word)] = float(1 if y[keep][0] else -1)
                    else:
                        expert = LinearSVC(C=setting["C"], class_weight="balanced", max_iter=20000, random_state=self.seed)
                        expert.fit(views["linear"][keep], y[keep])
                        self.experts[str(word)] = expert
        elif kind == "rbf":
            self.global_model = SVC(C=setting["C"], kernel="rbf", gamma="scale", cache_size=128, random_state=self.seed)
            self.global_model.fit(views["dense"], y, sample_weight=weights)
        elif kind == "boosting":
            self.global_model = HistGradientBoostingClassifier(max_iter=150, learning_rate=.06,
                max_leaf_nodes=setting["max_leaf_nodes"], min_samples_leaf=5,
                l2_regularization=1., early_stopping=False, random_state=self.seed)
            with threadpool_limits(limits=2):
                self.global_model.fit(views["dense"], y, sample_weight=weights)
        else:
            raise ValueError(kind)
        return self

    def decision_function(self, views, frame):
        kind = self.setting["kind"]
        matrix = views["linear"] if kind in ["linear", "experts"] else views["dense"]
        with threadpool_limits(limits=2):
            result = self.global_model.decision_function(matrix)
        if kind == "experts":
            local = result.copy()
            words = frame.word.to_numpy()
            for word in np.unique(words):
                keep = words == word
                expert = self.experts.get(str(word))
                if isinstance(expert, float):
                    local[keep] = expert
                elif expert is not None:
                    local[keep] = expert.decision_function(matrix[keep])
            mix = self.setting["global_mix"]
            result = mix*result+(1-mix)*local
        return np.asarray(result)


class AdvancedHybridModel:
    def __init__(self, builder, heads, scales, calibration):
        self.builder, self.heads, self.scales, self.calibration = builder, heads, scales, calibration

    def decision_function(self, frame):
        views = self.builder.transform(frame)
        return np.mean([head.decision_function(views, frame)/scale for head,scale in zip(self.heads, self.scales)], axis=0)

    def predict(self, frame):
        return apply_thresholds(frame.word, self.decision_function(frame), self.calibration)
