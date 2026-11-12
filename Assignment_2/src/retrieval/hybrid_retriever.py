"""Hybrid retrieval: BM25 + dense candidates fused with RRF.

Each system retrieves its own candidate list independently (raw scores are
never mixed); the ranks are then combined by :mod:`fusion`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..utils.logging_utils import get_logger
from .base import Hit
from .fusion import DEFAULT_RRF_K, fuse


class HybridRetriever:
    """Combine two retrievers (BM25, dense) with configurable fusion."""

    def __init__(
        self,
        bm25_retriever,
        dense_retriever,
        config: Optional[Dict[str, Any]] = None,
        log=None,
    ) -> None:
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.config = dict(config or {})
        self.log = log or get_logger("hybrid")

        fusion_cfg = self.config.get("fusion", {})
        self.fusion_method = str(fusion_cfg.get("method", "rrf"))
        self.rrf_k = int(fusion_cfg.get("rrf_k", DEFAULT_RRF_K))
        candidates = self.config.get("candidates", {})
        self.bm25_candidates = int(candidates.get("bm25", 100))
        self.dense_candidates = int(candidates.get("dense", 100))

    def retrieve(
        self,
        query_text: str,
        top_k: int = 100,
        allowed_doc_ids: Optional[Sequence[str]] = None,
    ) -> List[Hit]:
        """Retrieve candidates from both systems and fuse them with RRF."""
        bm25_hits = self.bm25.retrieve(
            query_text,
            top_k=self.bm25_candidates,
            allowed_doc_ids=allowed_doc_ids,
        )
        dense_hits = self.dense.retrieve(
            query_text,
            top_k=self.dense_candidates,
            allowed_doc_ids=allowed_doc_ids,
        )
        return fuse(
            [bm25_hits, dense_hits],
            method=self.fusion_method,
            k=self.rrf_k,
            top_k=top_k,
        )

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
