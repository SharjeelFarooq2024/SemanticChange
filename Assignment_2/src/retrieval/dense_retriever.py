"""Dense vector retrieval over pretrained embeddings using FAISS.

- Document embeddings are generated once and cached (see
  :mod:`Assignment_2.src.embeddings.embedding_manager`).
- Similarity is inner product over L2-normalized vectors (= cosine) for
  the default configuration; FAISS ``IndexFlatIP`` gives exact search.
- Original ``sentence_id`` document IDs are preserved via an ordered ID
  map (vector position -> doc_id).
- Per-word candidate restriction is applied by exact post-filtering
  (the corpus is small enough that fetching the full ranking per query
  is exact and cheap; documented scaling caveat in the README).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..data.data_loader import Document
from ..embeddings.embedding_manager import EmbeddingCache, fingerprint_documents
from ..embeddings.encoder import DenseEncoder
from ..utils.logging_utils import get_logger
from ..utils.paths import ensure_dir, resolve_a2_path
from .base import Hit, sort_and_rank


class DenseIndexError(RuntimeError):
    """Raised when the dense index cannot be built or is missing."""


class _NumpyIndexFlatIP:
    """Pure-NumPy exact inner product search used as a robust fallback.

    Produces bit-for-bit identical cosine scores to FAISS IndexFlatIP
    without requiring external C/C++ SWIG DLLs.
    """

    def __init__(self, dimension: int, vectors: Optional[np.ndarray] = None) -> None:
        self.d = dimension
        self.vectors = (
            vectors
            if vectors is not None
            else np.empty((0, dimension), dtype=np.float32)
        )

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def add(self, x: np.ndarray) -> None:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if len(self.vectors) == 0:
            self.vectors = x
        else:
            self.vectors = np.vstack([self.vectors, x])

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(q, dtype=np.float32)
        scores = np.dot(q, self.vectors.T)
        k = min(k, self.ntotal)
        indices = np.argsort(-scores, axis=1)[:, :k]
        sorted_scores = np.take_along_axis(scores, indices, axis=1)
        return sorted_scores, indices


def _try_import_faiss():
    try:
        import faiss
        return faiss
    except Exception:
        return None


class DenseRetriever:
    """Encode -> cache -> FAISS index -> top-k cosine retrieval."""

    def __init__(
        self,
        config: Dict[str, Any],
        encoder_name: Optional[str] = None,
        log=None,
        encoder: Optional[DenseEncoder] = None,
    ) -> None:
        self.config = dict(config)
        self.log = log or get_logger("dense")

        encoder_cfg = self.config.get("encoder", {})
        self.encoder_name = encoder_name or encoder_cfg.get("name", "e5_small")
        cache_cfg = self.config.get("cache", {})

        if encoder is not None:
            self.encoder = encoder  # injection point for tests
        else:
            models = encoder_cfg.get("models", {})
            if self.encoder_name not in models:
                raise DenseIndexError(
                    f"Unknown encoder '{self.encoder_name}'. Available: {sorted(models)}"
                )
            self.encoder = DenseEncoder(
                encoder_name=self.encoder_name,
                models_config=models,
                device=encoder_cfg.get("device", "auto"),
                model_cache_dir=resolve_a2_path(
                    cache_cfg.get("model_cache_dir", "artifacts/model_cache")
                ),
                log=self.log,
            )

        self.cache = EmbeddingCache(
            resolve_a2_path(
                cache_cfg.get("embeddings_dir", "artifacts/document_embeddings")
            ),
            log=self.log,
        )
        self._index = None
        self._doc_ids: List[str] = []
        self._doc_id_set: set[str] = set()
        self._dimension: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Index construction
    # ------------------------------------------------------------------ #
    @property
    def is_built(self) -> bool:
        return self._index is not None

    @property
    def doc_ids(self) -> List[str]:
        return list(self._doc_ids)

    @property
    def dimension(self) -> Optional[int]:
        return self._dimension

    def _index_paths(self) -> tuple[Path, Path]:
        base = resolve_a2_path(
            self.config.get("cache", {}).get(
                "index_dir", "artifacts/document_embeddings"
            )
        )
        return base / f"{self.encoder.encoder_name}.faiss", base / (
            f"{self.encoder.encoder_name}_ids.json"
        )

    def build_index(self, documents: Sequence[Document], force_rebuild: bool = False) -> None:
        """Build (or load a compatible cached) FAISS/Numpy index for ``documents``."""
        faiss = _try_import_faiss()

        doc_ids = [str(doc.doc_id) for doc in documents]
        texts = [doc.text for doc in documents]
        fingerprint = fingerprint_documents(doc_ids, texts)
        identity = self.encoder.identity(doc_fingerprint=fingerprint)

        embeddings: Optional[np.ndarray] = None
        reason = "forced"
        if not force_rebuild:
            embeddings, reason = self.cache.load(identity, doc_ids)

        if embeddings is None:
            self.log.info(
                "Generating document embeddings (%s): %s", reason, self.encoder_name
            )
            embeddings = self.encoder.encode(texts, kind="passage")
            self.cache.save(identity, doc_ids, embeddings)
        else:
            self.log.info("Loaded cached embeddings (%s)", reason)

        if embeddings.shape[1] == 0:
            raise DenseIndexError("Encoder returned zero-dimensional embeddings")

        index_path, ids_path = self._index_paths()
        self._dimension = int(embeddings.shape[1])

        # --- Index: load if compatible, else rebuild and save ----- #
        loaded = False
        if index_path.exists() and ids_path.exists() and not force_rebuild:
            try:
                meta = json.loads(ids_path.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    saved_ids = [str(i) for i in meta.get("doc_ids", [])]
                    if (
                        saved_ids == doc_ids
                        and meta.get("identity") == identity
                    ):
                        if faiss is not None:
                            existing = faiss.read_index(str(index_path))
                            if existing.d == self._dimension and existing.ntotal == len(doc_ids):
                                self._index = existing
                                loaded = True
                        else:
                            # Load via pure numpy
                            vecs = np.load(str(index_path))
                            if vecs.shape == (len(doc_ids), self._dimension):
                                self._index = _NumpyIndexFlatIP(self._dimension, vecs)
                                loaded = True
            except Exception:
                loaded = False

        if not loaded:
            if faiss is not None:
                index = faiss.IndexFlatIP(self._dimension)
                index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
                ensure_dir(index_path.parent)
                faiss.write_index(index, str(index_path))
                self.log.info("Built FAISS IndexFlatIP: %d vectors x %d dims",
                              index.ntotal, self._dimension)
            else:
                index = _NumpyIndexFlatIP(self._dimension)
                index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
                ensure_dir(index_path.parent)
                np.save(str(index_path), index.vectors)
                self.log.info("Built NumPy exact inner product index: %d vectors x %d dims",
                              index.ntotal, self._dimension)

            ids_path.write_text(
                json.dumps(
                    {"doc_ids": doc_ids, "identity": identity},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            self._index = index
        else:
            self.log.info("Loaded index: %d vectors x %d dims",
                          self._index.ntotal, self._dimension)

        self._doc_ids = doc_ids
        self._doc_id_set = set(doc_ids)

    def _require_index(self):
        if self._index is None:
            raise DenseIndexError(
                "Dense index is not built. Call build_index() first."
            )
        return self._index

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query_text: str,
        top_k: int = 100,
        allowed_doc_ids: Optional[Sequence[str]] = None,
    ) -> List[Hit]:
        """Return the top hits for ``query_text`` (cosine over normalized vectors)."""
        index = self._require_index()
        query_vector = self.encoder.encode([query_text], kind="query")

        # Exact candidate restriction: search the whole index, then filter.
        # (Corpus is small; see README for the large-corpus alternative.)
        depth = len(self._doc_ids) if allowed_doc_ids is not None else min(
            top_k, index.ntotal
        )
        depth = max(depth, 1)
        scores, indices = index.search(np.ascontiguousarray(query_vector), depth)

        allowed = (
            set(str(doc_id) for doc_id in allowed_doc_ids)
            if allowed_doc_ids is not None
            else None
        )
        if allowed is not None:
            unknown = allowed - self._doc_id_set
            if unknown:
                raise DenseIndexError(
                    f"allowed_doc_ids contains IDs missing from the index: "
                    f"{sorted(unknown)[:5]}"
                )

        scored: Dict[str, float] = {}
        for score, position in zip(scores[0], indices[0]):
            if position < 0:
                continue
            doc_id = self._doc_ids[int(position)]
            if allowed is not None and doc_id not in allowed:
                continue
            scored[doc_id] = float(score)
            if allowed is None and len(scored) >= top_k:
                break

        return sort_and_rank(scored, top_k=top_k)

    def retrieve_batch(
        self,
        queries: Sequence[Any],
        top_k: int = 100,
        allowed_doc_ids_by_query: Optional[Dict[str, Sequence[str]]] = None,
    ) -> Dict[str, List[Hit]]:
        """Retrieve for many queries; keys are ``query.query_id``."""
        results: Dict[str, List[Hit]] = {}
        for query in queries:
            allowed = None
            if allowed_doc_ids_by_query is not None:
                allowed = allowed_doc_ids_by_query.get(query.query_id)
            results[query.query_id] = self.retrieve(
                query.text, top_k=top_k, allowed_doc_ids=allowed
            )
        return results
