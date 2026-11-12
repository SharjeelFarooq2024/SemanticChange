"""Shared retrieval types and the common retriever interface.

All three systems (BM25, dense, hybrid) return the same ``Hit`` structure so
evaluation code can consume them uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class Hit:
    """One ranked retrieval result."""

    doc_id: str
    score: float
    rank: int  # 1-based

    def as_dict(self) -> Dict[str, object]:
        return {"doc_id": self.doc_id, "score": float(self.score), "rank": int(self.rank)}


def sort_and_rank(scored: Dict[str, float], top_k: Optional[int] = None) -> List[Hit]:
    """Convert ``{doc_id: score}`` into ranked :class:`Hit` objects.

    Sorting is deterministic: score descending, then doc_id ascending for
    ties, so identical inputs always produce identical output.
    """
    ordered = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        ordered = ordered[:top_k]
    return [Hit(doc_id=doc_id, score=float(score), rank=index + 1)
            for index, (doc_id, score) in enumerate(ordered)]


@runtime_checkable
class Retriever(Protocol):
    """Minimal interface shared by all retrieval systems."""

    def retrieve(
        self,
        query_text: str,
        top_k: int = 100,
        allowed_doc_ids: Optional[Sequence[str]] = None,
    ) -> List[Hit]:
        """Return the top ``top_k`` hits for ``query_text``.

        Args:
            query_text: Raw query string (preprocessed internally).
            top_k: Maximum number of hits to return.
            allowed_doc_ids: When given, restrict candidates to this set
                (used by ``scope: per_word``).
        """
        ...
