"""Measured execution, immutable data loading, model resolution and cache identity."""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import platform
import sys
import threading
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "Assignment_3/configs/complete_experiments.json"


def load_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def text_group(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def load_data():
    folder = ROOT / "data/raw/dev/SV"
    data = pd.read_json(folder / "usages.jsonl", lines=True)
    labels = pd.read_json(folder / "subtask2.jsonl", lines=True).rename(columns={"label": "label_value"})
    definitions = pd.read_json(folder / "definitions.jsonl", lines=True)
    data = data.merge(labels, on=["word", "sentence_id"], validate="one_to_one")
    data = data.merge(definitions, on="word", validate="many_to_one")
    if len(data) != 1983 or data.sentence_id.duplicated().any():
        raise ValueError("Unexpected dataset identities or coverage")
    data["text_group"] = data.text.map(text_group).map(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())
    return data, definitions


def logger(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("completed_experiments")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler(output / "execution.log", encoding="utf-8")]:
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log


class Measurements:
    def __init__(self, output, interval=.1):
        self.path = Path(output) / "resource_measurements.jsonl"
        self.interval = interval
        self.records = []

    @contextmanager
    def measure(self, stage, **metadata):
        process = psutil.Process()
        samples = [process.memory_info().rss]
        stop = threading.Event()
        def sample():
            while not stop.wait(self.interval):
                samples.append(process.memory_info().rss)
        monitor = threading.Thread(target=sample, daemon=True)
        cpu_start = process.cpu_times()
        start = time.perf_counter()
        monitor.start()
        status = "failed"
        try:
            yield
            status = "ok"
        finally:
            wall = time.perf_counter()-start
            stop.set()
            monitor.join()
            samples.append(process.memory_info().rss)
            cpu_end = process.cpu_times()
            row = {"stage": stage, "status": status, **metadata,
                   "wall_seconds": wall, "cpu_seconds": cpu_end.user+cpu_end.system-cpu_start.user-cpu_start.system,
                   "rss_baseline_mb": samples[0]/2**20, "sampled_peak_rss_mb": max(samples)/2**20,
                   "sampled_peak_increase_mb": (max(samples)-samples[0])/2**20,
                   "sample_interval_seconds": self.interval,
                   "monetary_cost": None, "energy_kwh": None}
            self.records.append(row)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")


def resolve_model(profile, allow_download=False):
    """Prefer exact local snapshots; downloads are explicit and go inside workspace."""
    name, revision = profile["model_name"], profile["revision"]
    cache_name = "models--" + name.replace("/", "--")
    candidates = [ROOT / "Assignment_2/artifacts/model_cache" / cache_name / "snapshots" / revision,
                  Path.home() / ".cache/huggingface/hub" / cache_name / "snapshots" / revision,
                  ROOT / "Assignment_3/model_cache" / cache_name / "snapshots" / revision]
    for candidate in candidates:
        if (candidate / "model.safetensors").exists():
            return candidate
    if not allow_download:
        raise FileNotFoundError(f"Missing pinned model {name}@{revision}. Run prepare_representations.py --allow-download.")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(name, revision=revision, cache_dir=ROOT / "Assignment_3/model_cache",
                                  allow_patterns=["*.json", "*.safetensors", "*.model"]))


def hardware():
    import importlib.metadata
    import torch
    packages = {}
    for name in ["numpy", "pandas", "scikit-learn", "scipy", "torch", "transformers", "rank-bm25", "psutil", "joblib"]:
        packages[name] = importlib.metadata.version(name)
    return {"platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "physical_cpus": psutil.cpu_count(logical=False), "ram_total_gb": psutil.virtual_memory().total/2**30,
            "ram_available_gb": psutil.virtual_memory().available/2**30, "python": sys.version,
            "cuda_available": torch.cuda.is_available(), "packages": packages}


def encode_batches(tokenizer, model, texts, batch_size, max_length, log, label):
    import torch
    matrix = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    # Length sorting reduces padding; vectors are restored to original identities.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    start = time.perf_counter()
    for offset in range(0, len(order), batch_size):
        indices = order[offset:offset+batch_size]
        inputs = tokenizer([texts[i] for i in indices], return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length)
        with torch.inference_mode():
            state = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(state.dtype)
            pooled = (state*mask).sum(1)/mask.sum(1).clamp(min=1)
            matrix[indices] = pooled.cpu().numpy()
        if offset == 0 or (offset // batch_size) % 32 == 0 or offset+batch_size >= len(order):
            log.info("%s: %d/%d encoded (%.1fs)", label, min(offset+batch_size, len(order)), len(order), time.perf_counter()-start)
    return matrix


def prepare(config, output, log, measurements, allow_download=False, force=False):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from src.eda.embedding_analysis import build_usage_context
    torch.set_num_threads(config["torch_threads"])
    torch.manual_seed(42)
    data, definitions = load_data()
    folder = output / "artifacts"
    folder.mkdir(parents=True, exist_ok=True)
    for key in ["e5_small", "xlmr_base"]:
        profile = config["models"][key]
        if allow_download:
            # A packaged representation cache can be valid while neural weights
            # are absent. Explicit downloads must also enable the live demo.
            resolve_model(profile, allow_download=True)
        contexts = [build_usage_context(row) for _, row in data.iterrows()] if key == "xlmr_base" else []
        doc_texts = [profile["passage_prefix"]+t for t in data.text]
        query_texts = [profile["query_prefix"]+t for t in definitions.definition]
        identity = {"model": profile, "max_length": config["max_length"], "pooling": "mean",
                    "doc_ids": data.sentence_id.tolist(), "doc_texts_sha256": fingerprint(doc_texts),
                    "query_words": definitions.word.tolist(), "queries_sha256": fingerprint(query_texts),
                    "classifier_contexts_sha256": fingerprint(contexts)}
        cache_path, meta_path = folder / f"{key}.npz", folder / f"{key}.json"
        if not force and cache_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta["identity"] == identity and meta["cache_sha256"] == sha256(cache_path):
                log.info("Validated representation cache: %s", key)
                continue
        model_path = resolve_model(profile, allow_download)
        log.info("Loading pinned encoder %s on CPU", key)
        with measurements.measure("encoder_load", encoder=key, cache_status="local_snapshot"):
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
            model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
            model.eval().requires_grad_(False)
        # Deduplicate text inference only, never alter the evaluation universe.
        all_texts = list(dict.fromkeys(doc_texts+query_texts+contexts))
        with measurements.measure("encoder_encode", encoder=key, n_unique_texts=len(all_texts),
                                  cache_status="fresh_inference", batch_size=config["encoder_batch_size"],
                                  max_length=config["max_length"], threads=config["torch_threads"]):
            vectors = encode_batches(tokenizer, model, all_texts, config["encoder_batch_size"], config["max_length"], log, key)
        positions = {text: i for i, text in enumerate(all_texts)}
        docs = vectors[[positions[t] for t in doc_texts]]
        queries = vectors[[positions[t] for t in query_texts]]
        docs = docs / np.linalg.norm(docs, axis=1, keepdims=True).clip(min=1e-12)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True).clip(min=1e-12)
        payload = {"doc_ids": data.sentence_id.to_numpy(dtype=str), "query_words": definitions.word.to_numpy(dtype=str),
                   "documents": docs, "queries": queries}
        if contexts:
            payload["classifier_contexts"] = vectors[[positions[t] for t in contexts]]
        np.savez_compressed(cache_path, **payload)
        dump_json(meta_path, {"identity": identity, "cache_sha256": sha256(cache_path),
                             "model_weights_sha256": sha256(model_path / "model.safetensors"),
                             "model_config_sha256": sha256(model_path / "config.json"),
                             "parameters": sum(p.numel() for p in model.parameters()),
                             "dimension": int(docs.shape[1]), "bytes": cache_path.stat().st_size})
        del model, tokenizer, vectors, docs, queries, payload
        gc.collect()
    dump_json(output / "hardware.json", hardware())
