"""Document-embedding cache with strict compatibility validation.

Layout (per encoder, under ``artifacts/document_embeddings/``):

    {encoder_key}.npz         float32 matrix + ordered doc_ids
    {encoder_key}_meta.json   identity dict + shape/dtype + timestamp

A cached file is reused ONLY when every identity field matches exactly
(encoder key, model name, max_length, pooling, prefixes, normalization,
and the corpus fingerprint). Otherwise the caller regenerates — an
incompatible cache is never silently reused.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.logging_utils import get_logger
from ..utils.paths import ensure_dir


def fingerprint_documents(doc_ids: Sequence[str], texts: Sequence[str]) -> str:
    """SHA-256 over ordered (doc_id, text) pairs.

    Any change to document identity, content, or ordering invalidates
    the embedding cache.
    """
    if len(doc_ids) != len(texts):
        raise ValueError("doc_ids and texts must have the same length")
    payload = json.dumps(
        {"pairs": list(zip([str(i) for i in doc_ids], [str(t) for t in texts]))},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class EmbeddingCache:
    """Save / load / validate document embeddings for one encoder."""

    def __init__(self, cache_dir: Path | str, log=None) -> None:
        self.cache_dir = ensure_dir(cache_dir)
        self.log = log or get_logger("embedding_cache")

    def _paths(self, encoder_key: str) -> Tuple[Path, Path]:
        return (
            self.cache_dir / f"{encoder_key}.npz",
            self.cache_dir / f"{encoder_key}_meta.json",
        )

    def load(
        self, encoder_identity: Dict[str, Any], expected_doc_ids: Sequence[str]
    ) -> Tuple[Optional[np.ndarray], str]:
        """Load cached embeddings if fully compatible.

        Returns ``(embeddings, reason)``; ``embeddings`` is None when the
        cache is missing or incompatible, and ``reason`` explains why.
        """
        encoder_key = encoder_identity["encoder_key"]
        npz_path, meta_path = self._paths(encoder_key)

        if not npz_path.exists() or not meta_path.exists():
            return None, "cache_missing"

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            with np.load(npz_path, allow_pickle=False) as data:
                embeddings = data["embeddings"]
                cached_ids = [str(doc_id) for doc_id in data["doc_ids"]]
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return None, f"cache_unreadable:{exc}"

        cached_identity = meta.get("identity", {})
        for key, value in encoder_identity.items():
            if cached_identity.get(key) != value:
                return None, f"identity_mismatch:{key}"

        if cached_ids != [str(doc_id) for doc_id in expected_doc_ids]:
            return None, "doc_id_mismatch"
        if embeddings.ndim != 2 or embeddings.shape[0] != len(expected_doc_ids):
            return None, "shape_mismatch"
        if int(meta.get("dimension", -1)) != embeddings.shape[1]:
            return None, "dimension_mismatch"
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        return embeddings, "validated"

    def save(
        self,
        encoder_identity: Dict[str, Any],
        doc_ids: Sequence[str],
        embeddings: np.ndarray,
    ) -> Tuple[Path, Path]:
        """Persist embeddings + identity metadata (atomic-ish, two files)."""
        encoder_key = encoder_identity["encoder_key"]
        npz_path, meta_path = self._paths(encoder_key)

        if embeddings.ndim != 2 or embeddings.shape[0] != len(doc_ids):
            raise ValueError(
                f"Embedding shape {embeddings.shape} does not match "
                f"{len(doc_ids)} document IDs"
            )

        np.savez_compressed(
            npz_path,
            embeddings=np.asarray(embeddings, dtype=np.float32),
            doc_ids=np.asarray([str(doc_id) for doc_id in doc_ids]),
        )
        meta = {
            "identity": encoder_identity,
            "n_documents": int(embeddings.shape[0]),
            "dimension": int(embeddings.shape[1]),
            "dtype": "float32",
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.log.info(
            "Saved embeddings: %s (%d x %d)",
            npz_path.name,
            embeddings.shape[0],
            embeddings.shape[1],
        )
        return npz_path, meta_path
