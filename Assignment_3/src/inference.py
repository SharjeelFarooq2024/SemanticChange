"""Inference from fitted pipelines, plus live and saved retrieval demonstrations."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from Assignment_2.src.retrieval.base import sort_and_rank
from Assignment_2.src.retrieval.bm25_retriever import BM25Retriever
from Assignment_2.src.retrieval.fusion import reciprocal_rank_fusion
from Assignment_3.src.runtime import dump_json, encode_batches, load_data, resolve_model, sha256
from src.baselines.classical_features import build_feature_frame
from src.eda.embedding_analysis import build_usage_context


def predict_checkpoint(checkpoint, inputs, output, log=None):
    checkpoint = checkpoint.resolve()
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    if sha256(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint integrity check failed")
    if not {"text", "word"}.issubset(inputs):
        raise ValueError("Inference inputs require text and word")
    data = inputs.copy()
    if data[["text", "word"]].isna().any().any():
        raise ValueError("Inference text and word must not be missing")
    if "sentence_id" not in data:
        data["sentence_id"] = [f"input_{i+1}" for i in range(len(data))]
    if data.duplicated(["word", "sentence_id"]).any():
        raise ValueError("Duplicate inference identity")
    for key, default in [("sentence_id", ""), ("period_label", ""), ("pos", "NN")]:
        if key not in data:
            data[key] = default
    if metadata["input_kind"] == "text":
        features = data.text
    elif metadata["input_kind"] == "eda_frame":
        config = json.loads((output / "baseline_config.json").read_text(encoding="utf-8"))
        features = build_feature_frame(data, config)
    elif metadata["input_kind"] == "frozen_xlmr_context":
        raw, _ = load_data()
        lookup = raw.set_index("sentence_id")
        cache = np.load(output / "artifacts/xlmr_base.npz")
        positions = {key: i for i, key in enumerate(cache["doc_ids"])}
        features = np.empty((len(data), cache["classifier_contexts"].shape[1]), dtype=np.float32)
        missing = []
        for i, row in enumerate(data.itertuples()):
            if row.sentence_id in positions and lookup.loc[row.sentence_id, "text"] == row.text and lookup.loc[row.sentence_id, "word"] == row.word:
                features[i] = cache["classifier_contexts"][positions[row.sentence_id]]
            else:
                missing.append(i)
        if missing:
            import torch
            from transformers import AutoModel, AutoTokenizer
            from Assignment_3.src.runtime import logger
            torch.set_num_threads(2)
            path = resolve_model(metadata["models"]["xlmr_base"])
            tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            model = AutoModel.from_pretrained(str(path), local_files_only=True).eval().requires_grad_(False)
            texts = [build_usage_context(data.iloc[i]) for i in missing]
            features[missing] = encode_batches(tokenizer, model, texts, 4, metadata["max_length"], log or logger(output), "live classifier inference")
    else:
        raise ValueError(f"Unsupported checkpoint input kind {metadata['input_kind']}")
    model = joblib.load(checkpoint)
    result = data[["word", "sentence_id"]].copy()
    result["prediction"] = model.predict(features).astype(int)
    return result


def retrieval_demo(config, output, word, system="hybrid", mode="saved", definition=None, top_k=5):
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    data, definitions = load_data()
    if word not in set(data.word):
        raise ValueError(f"Unknown word; choose from {sorted(data.word.unique())}")
    original = definitions.set_index("word").loc[word, "definition"]
    definition = definition or original
    cutoff_path = output / f"checkpoints/grouped_text/seed_42/{system}_train_tuned.json"
    cutoff = json.loads(cutoff_path.read_text(encoding="utf-8"))["cutoff"]
    if mode == "saved":
        if definition != original:
            raise ValueError("Custom definitions require --mode live")
        ranked = pd.read_csv(output / f"rankings/{system}.csv")
        hits = ranked[ranked.query_id == word].sort_values("rank")
    elif mode == "live":
        allowed = data.loc[data.word == word, "sentence_id"].tolist()
        lists = []
        if system in ["bm25", "hybrid"]:
            bm25 = BM25Retriever.load(output / "artifacts/bm25_index.pkl")
            lists.append(bm25.retrieve(definition, top_k=100, allowed_doc_ids=allowed))
        if system in ["dense", "hybrid", "dense_xlmr"]:
            import torch
            from transformers import AutoModel, AutoTokenizer
            encoder = "xlmr_base" if system == "dense_xlmr" else "e5_small"
            profile = config["models"][encoder]
            path = resolve_model(profile)
            torch.set_num_threads(config["torch_threads"])
            tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            model = AutoModel.from_pretrained(str(path), local_files_only=True).eval().requires_grad_(False)
            with torch.inference_mode():
                inputs = tokenizer([profile["query_prefix"]+definition], return_tensors="pt", truncation=True, max_length=config["max_length"])
                hidden = model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
                vector = torch.nn.functional.normalize(pooled, dim=1).numpy()[0]
            cache = np.load(output / f"artifacts/{encoder}.npz")
            scores = cache["documents"] @ vector
            keep = set(allowed)
            lists.append(sort_and_rank({str(i): float(s) for i,s in zip(cache["doc_ids"], scores) if str(i) in keep}, top_k=100))
        result = reciprocal_rank_fusion(lists, k=config["rrf_k"], top_k=100) if system == "hybrid" else lists[0]
        hits = pd.DataFrame([hit.as_dict() for hit in result])
        hits["query_id"] = word
    else:
        raise ValueError("Mode must be saved or live")
    hits = hits.merge(data[["sentence_id", "text", "label_value"]], left_on="doc_id", right_on="sentence_id", validate="one_to_one")
    hits["prediction"] = (hits["rank"] <= cutoff).astype(int)
    selected = hits.sort_values("rank").head(top_k)
    return {"mode": "saved_rankings" if mode == "saved" else "live_model_inference",
            "system": system, "word": word, "definition": definition, "cutoff": cutoff,
            "cutoff_source": "grouped_text training labels, seed 42", "prediction_rule": "rank<=cutoff",
            "gold_display_scope": "Released development annotations; illustrative, not independent demo evaluation",
            "hits": selected[["sentence_id", "text", "rank", "score", "prediction", "label_value"]].to_dict("records")}
