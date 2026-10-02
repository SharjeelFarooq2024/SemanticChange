from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from src.eda.load_data import build_dataset_bundle


class BaselineDataError(ValueError):
    """Raised when baseline inputs are missing or cannot be joined safely."""


def load_config(project_root: Path | None = None) -> Dict[str, Any]:
    import yaml

    root = project_root or Path(__file__).resolve().parents[2]
    path = root / "configs" / "baselines.yaml"
    if not path.exists():
        raise BaselineDataError(f"Baseline configuration is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["project_root"] = str(root)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def prepare_output_dirs(root: Path, config: Dict[str, Any]) -> None:
    for key in ("classical", "pretrained", "figures", "tables"):
        (root / config["outputs"][key]).mkdir(parents=True, exist_ok=True)


def load_baseline_bundle(root: Path, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    data_root = root / config["data"]["root"]
    required = [config["data"][key] for key in ("usages", "subtask1", "subtask2", "definitions")]
    missing = [str(data_root / name) for name in required if not (data_root / name).exists()]
    if missing:
        raise BaselineDataError("Missing required baseline input file(s): " + ", ".join(missing))
    bundle = build_dataset_bundle(root)
    if bundle["usage"].empty:
        raise BaselineDataError("usages.jsonl exists but contains no usable records")
    return bundle


def _scalar_label(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    return None


def join_subtask2_labels(usage: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    required = {"sentence_id", "label"}
    if not required.issubset(labels.columns):
        raise BaselineDataError("Subtask 2 must contain sentence_id and label columns")
    label_frame = labels[["sentence_id", "label"]].copy()
    label_frame["label_value"] = label_frame["label"].map(_scalar_label)
    malformed = int(label_frame["label_value"].isna().sum())
    label_frame = label_frame.dropna(subset=["label_value"]).copy()
    label_frame["label_value"] = label_frame["label_value"].astype(int)
    duplicate_labels = int(label_frame["sentence_id"].duplicated(keep=False).sum())
    label_frame = label_frame.drop_duplicates("sentence_id", keep="first")
    merged = usage.merge(label_frame[["sentence_id", "label_value"]], on="sentence_id", how="left", indicator=True)
    unmatched = int((merged["_merge"] != "both").sum())
    diagnostics = {
        "usage_records": int(len(usage)),
        "label_records": int(len(labels)),
        "matched_records": int((merged["_merge"] == "both").sum()),
        "unmatched_usage_records": unmatched,
        "malformed_labels": malformed,
        "duplicate_label_records": duplicate_labels,
    }
    return merged.loc[merged["_merge"] == "both"].drop(columns="_merge"), diagnostics


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def save_diagnostics(path: Path, diagnostics: Dict[str, Any]) -> None:
    write_json(path, {"data_join_diagnostics": diagnostics})


def _fingerprint_usage(usage: pd.DataFrame, contexts: list[str]) -> str:
    import hashlib

    identity = usage[["word", "sentence_id", "period_label", "year", "text"]].astype(str).to_dict("records")
    payload = json.dumps({"records": identity, "contexts": contexts}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_paths(root: Path, config: dict) -> tuple[Path, Path]:
    cache_dir = root / config["pretrained"]["cache_dir"]
    return cache_dir / "usage_embeddings.pkl", cache_dir / "usage_embeddings_metadata.json"


def _load_cached(usage: pd.DataFrame, contexts: list[str], config: dict, root: Path) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    import pickle

    cache_path, metadata_path = _cache_paths(root, config)
    if not cache_path.exists() or not metadata_path.exists():
        return None, {"cache_used": False, "cache_reason": "cache_file_or_metadata_missing"}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
    except (OSError, ValueError, pickle.PickleError, EOFError) as exc:
        return None, {"cache_used": False, "cache_reason": f"cache_read_error: {exc}"}
    expected = {
        "fingerprint": _fingerprint_usage(usage, contexts),
        "model_name": config["pretrained"]["model_name"],
        "max_length": config["pretrained"]["max_length"],
        "pooling": config["pretrained"]["pooling"],
    }
    if not isinstance(cached, pd.DataFrame) or metadata.get("fingerprint") != expected["fingerprint"]:
        return None, {"cache_used": False, "cache_reason": "record_identity_or_context_mismatch"}
    for key in ("model_name", "max_length", "pooling"):
        if metadata.get(key) != expected[key]:
            return None, {"cache_used": False, "cache_reason": f"cache_{key}_mismatch"}
    required = {"word", "sentence_id", "period_label", "year", "text", "embedding", "context_used"}
    if not required.issubset(cached.columns) or len(cached) != len(usage):
        return None, {"cache_used": False, "cache_reason": "cache_schema_or_length_mismatch"}
    if cached["context_used"].tolist() != contexts:
        return None, {"cache_used": False, "cache_reason": "cached_context_mismatch"}
    return cached, {"cache_used": True, "cache_reason": "validated"}


def load_frozen_usage_embeddings(usage: pd.DataFrame, config: dict, root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load validated cached XLM-R features or create them with frozen parameters."""
    import pickle
    from src.eda.embedding_analysis import build_usage_context

    contexts = [build_usage_context(row) for _, row in usage.iterrows()]
    cached, cache_status = _load_cached(usage, contexts, config, root)
    if cached is not None:
        metadata = json.loads((_cache_paths(root, config)[1]).read_text(encoding="utf-8"))
        return cached, {**metadata, **cache_status}
    try:
        from src.eda.embedding_analysis import encode_texts, load_model, resolve_device, set_seed as eda_set_seed
    except ModuleNotFoundError as exc:
        raise BaselineDataError("Frozen XLM-R requires the installed torch and transformers packages") from exc
    try:
        eda_set_seed(int(config["seed"]))
        device_name = config["pretrained"].get("device", "auto")
        device = resolve_device(device_name)
        cache_dir = root / config["pretrained"]["cache_dir"]
        tokenizer, model, device = load_model(config["pretrained"]["model_name"], cache_dir, device)
    except Exception as exc:
        raise BaselineDataError(f"Unable to load XLM-R model '{config['pretrained']['model_name']}'. Check model files and network/cache access: {exc}") from exc
    embeddings = encode_texts(tokenizer, model, contexts, device, batch_size=config["pretrained"]["batch_size"], max_length=config["pretrained"]["max_length"])
    result = usage[["word", "sentence_id", "period_label", "year", "text"]].copy()
    result["embedding"] = [vector for vector in embeddings]
    result["context_used"] = contexts
    cache_path, metadata_path = _cache_paths(root, config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": config["pretrained"]["model_name"],
        "tokenizer_name": config["pretrained"]["model_name"],
        "seed": int(config["seed"]),
        "device": str(device),
        "max_length": int(config["pretrained"]["max_length"]),
        "pooling": config["pretrained"]["pooling"],
        "frozen": True,
        "fingerprint": _fingerprint_usage(usage, contexts),
        "embedding_dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
    }
    with cache_path.open("wb") as handle:
        pickle.dump(result, handle)
    write_json(metadata_path, metadata)
    return result, {**metadata, **cache_status, "cache_used": False, "cache_reason": cache_status["cache_reason"]}