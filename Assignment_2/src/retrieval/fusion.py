"""Rank-fusion strategies.

Fusion operates exclusively on RANKS (never on raw scores), because BM25
scores and cosine similarities are incompatible scales.

Currently implemented: Reciprocal Rank Fusion (RRF), Cormack et al., SIGIR 2009:

    RRF_score(d) = sum over systems i of 1 / (k + rank_i(d))

New fusion methods can be added here and selected via
``hybrid_config.yaml -> fusion.method`` without touching retrieval code.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from .base import Hit

DEFAULT_RRF_K = 60  # standard smoothing constant from Cormack et al. (2009)

FUSION_METHODS = ("rrf",)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hit]],
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> List[Hit]:
    """Fuse multiple ranked lists with RRF.

    Args:
        rankings: One ranked list per retrieval system (rank must be 1-based
            and contiguous; only ``Hit.rank`` is used, never ``Hit.score``).
        k: RRF smoothing constant (larger k dampens the contribution of
            top ranks).
        top_k: Optionally truncate the fused list.

    Returns:
        Fused hits ranked by RRF score descending, ties broken by doc_id
        for determinism. Ranks are reassigned 1..N.

    Notes:
        - Documents missing from a system's ranking simply contribute 0
          from that system (handled naturally by the sum).
        - Duplicate doc_ids inside a single ranking are collapsed to their
          best (first) rank.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")

    scores: Dict[str, float] = {}
    for ranking in rankings:
        seen_in_this_ranking: set[str] = set()
        for hit in ranking:
            doc_id = hit.doc_id
            if doc_id in seen_in_this_ranking:
                continue  # keep the best (first) occurrence only
            seen_in_this_ranking.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + hit.rank)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        ordered = ordered[:top_k]
    return [
        Hit(doc_id=doc_id, score=float(score), rank=index + 1)
        for index, (doc_id, score) in enumerate(ordered)
    ]


def fuse(
    rankings: Sequence[Sequence[Hit]],
    method: str = "rrf",
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> List[Hit]:
    """Dispatch to a registered fusion method (extensibility point)."""
    if method == "rrf":
        return reciprocal_rank_fusion(rankings, k=k, top_k=top_k)
    raise ValueError(
        f"Unknown fusion method '{method}'. Available: {FUSION_METHODS}"
    )
