from __future__ import annotations

import math
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


SEED = 42


def set_seed(seed: int = SEED) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str = "auto") -> torch.device:
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    pooled = torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
    return pooled


def load_model(model_name: str, cache_dir: str | Path, device: torch.device) -> Tuple[AutoTokenizer, AutoModel, torch.device]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
    model = AutoModel.from_pretrained(model_name, cache_dir=str(cache_dir))
    model.to(device)
    model.eval()
    return tokenizer, model, device


def build_usage_context(row: pd.Series) -> str:
    text = str(row.get("text", "") or "")
    word = str(row.get("word", "") or "")
    if not text:
        return ""
    if word:
        lower_text = text.lower()
        lower_word = word.lower()
        idx = lower_text.find(lower_word)
        if idx >= 0:
            left = max(0, idx - 80)
            right = min(len(text), idx + len(word) + 80)
            return text[left:right]
    return text


def encode_texts(tokenizer: AutoTokenizer, model: AutoModel, texts: List[str], device: torch.device, batch_size: int = 32, max_length: int = 512) -> np.ndarray:
    import torch

    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    all_embeddings: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        all_embeddings.append(pooled.cpu().numpy().astype(np.float32))
    return np.vstack(all_embeddings)


def get_embedding_cache_path(cache_dir: str | Path) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "usage_embeddings.pkl"


def ensure_embedding_cache(
    usage_df: pd.DataFrame,
    config: Dict[str, Any],
    tokenizer: AutoTokenizer | None = None,
    model: AutoModel | None = None,
) -> pd.DataFrame:
    cache_path = get_embedding_cache_path(config["paths"]["embedding_cache_dir"])
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        if isinstance(cached, pd.DataFrame):
            return cached

    set_seed(config.get("model", {}).get("seed", SEED))
    if tokenizer is None or model is None:
        device = resolve_device(config.get("model", {}).get("device", "auto"))
        model_name = config.get("model", {}).get("name", "FacebookAI/xlm-roberta-base")
        cache_dir = Path(config["paths"]["embedding_cache_dir"])
        tokenizer, model, device = load_model(model_name, cache_dir, device)
    else:
        device = next(model.parameters()).device

    texts = [build_usage_context(row) for _, row in usage_df.iterrows()]
    embeddings = encode_texts(tokenizer, model, texts, device, batch_size=config.get("model", {}).get("batch_size", 32))

    emb_df = usage_df[["word", "sentence_id", "period_label", "year", "text"]].copy()
    emb_df["embedding"] = [embedding for embedding in embeddings]
    emb_df["context_used"] = texts
    with cache_path.open("wb") as handle:
        pickle.dump(emb_df, handle)
    return emb_df


def compute_pca_embeddings(emb_df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    matrix = np.vstack(emb_df["embedding"].to_numpy())
    pca = PCA(n_components=min(n_components, matrix.shape[0], matrix.shape[1]))
    reduced = pca.fit_transform(matrix)
    out = emb_df.copy()
    for i in range(reduced.shape[1]):
        out[f"pca_{i + 1}"] = reduced[:, i]
    return out


def compute_adjacent_period_similarity(emb_df: pd.DataFrame) -> pd.DataFrame:
    if emb_df.empty:
        return pd.DataFrame(columns=["word", "period_from", "period_to", "similarity", "distance"])

    records = []
    for word, group in emb_df.groupby("word"):
        ordered = group.sort_values("period_label")
        period_means: Dict[str, np.ndarray] = {}
        for period, sub in ordered.groupby("period_label"):
            vectors = np.vstack(sub["embedding"].tolist())
            period_means[period] = vectors.mean(axis=0)
        periods = sorted(period_means.keys())
        for current, nxt in zip(periods, periods[1:]):
            v1 = period_means[current]
            v2 = period_means[nxt]
            sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]
            if math.isnan(sim):
                sim = 0.0
            records.append(
                {
                    "word": word,
                    "period_from": current,
                    "period_to": nxt,
                    "similarity": float(sim),
                    "distance": float(1.0 - sim),
                    "n_from": int((ordered["period_label"] == current).sum()),
                    "n_to": int((ordered["period_label"] == nxt).sum()),
                }
            )
    return pd.DataFrame(records)


def compute_definition_usage_similarity(
    emb_df: pd.DataFrame,
    definitions_df: pd.DataFrame,
    subtask2_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int = 32,
) -> pd.DataFrame:
    if emb_df.empty or definitions_df.empty:
        return pd.DataFrame(columns=["word", "sentence_id", "label", "cosine_similarity", "period_label"])

    if not subtask2_df.empty:
        subtask2_df = subtask2_df.copy()
        subtask2_df["sentence_id"] = subtask2_df["sentence_id"].astype(str)

    def_vectors = {}
    definition_texts = []
    definition_words = []
    for _, row in definitions_df.iterrows():
        word = str(row.get("word", "")).strip()
        text = str(row.get("definition", "") or "")
        if word and text:
            definition_words.append(word)
            definition_texts.append(text)
    if definition_texts:
        def_embeds = encode_texts(tokenizer, model, definition_texts, next(model.parameters()).device, batch_size=batch_size)
        for word, vec in zip(definition_words, def_embeds):
            def_vectors[word] = vec

    usage_map = {str(row["sentence_id"]): row for _, row in emb_df.iterrows()}
    rows = []
    for _, row in subtask2_df.iterrows():
        sentence_id = str(row.get("sentence_id", ""))
        word = str(row.get("word", "")).strip()
        usage_row = usage_map.get(sentence_id)
        if usage_row is None or word not in def_vectors:
            continue
        sim = cosine_similarity(np.asarray(usage_row["embedding"]).reshape(1, -1), def_vectors[word].reshape(1, -1))[0, 0]
        rows.append(
            {
                "word": word,
                "sentence_id": sentence_id,
                "label": row.get("label"),
                "cosine_similarity": float(sim),
                "period_label": usage_row.get("period_label"),
            }
        )
    return pd.DataFrame(rows)


def analyze_embeddings(usage_df: pd.DataFrame, config: Dict[str, Any], definitions_df: pd.DataFrame | None = None, subtask2_df: pd.DataFrame | None = None) -> Dict[str, pd.DataFrame]:
    model_config = config.get("model", {})
    device = resolve_device(model_config.get("device", "auto"))
    model_name = model_config.get("name", "FacebookAI/xlm-roberta-base")
    cache_dir = Path(config["paths"]["embedding_cache_dir"])
    tokenizer, model, loaded_device = load_model(model_name, cache_dir, device)
    emb_df = ensure_embedding_cache(usage_df, config, tokenizer=tokenizer, model=model)
    pca_df = compute_pca_embeddings(emb_df, n_components=model_config.get("pca_components", 2))
    similarity = compute_adjacent_period_similarity(pca_df)
    result = {
        "embeddings": emb_df,
        "pca": pca_df,
        "temporal_similarity": similarity,
    }
    if definitions_df is not None and subtask2_df is not None:
        result["definition_usage_similarity"] = compute_definition_usage_similarity(
            emb_df,
            definitions_df,
            subtask2_df,
            tokenizer=tokenizer,
            model=model,
            batch_size=model_config.get("batch_size", 32),
        )
    return result
