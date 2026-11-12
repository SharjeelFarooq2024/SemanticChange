"""Tests for RRF fusion and the hybrid retriever."""
from __future__ import annotations

import math

import pytest

from Assignment_2.src.retrieval.base import Hit
from Assignment_2.src.retrieval.fusion import (
    DEFAULT_RRF_K,
    fuse,
    reciprocal_rank_fusion,
)
from Assignment_2.src.retrieval.hybrid_retriever import HybridRetriever


def _ranking(*doc_ids: str) -> list[Hit]:
    return [Hit(doc_id=doc_id, score=100.0 - i, rank=i + 1)
            for i, doc_id in enumerate(doc_ids)]


# --------------------------------------------------------------------------- #
# RRF mathematics
# --------------------------------------------------------------------------- #
def test_rrf_score_is_sum_of_inverse_rank_offsets():
    # d1: rank 1 in A, rank 2 in B -> 1/(60+1) + 1/(60+2)
    a = _ranking("d1", "d2")
    b = _ranking("d2", "d1")
    fused = {hit.doc_id: hit.score for hit in reciprocal_rank_fusion([a, b], k=60)}
    expected = 1 / (60 + 1) + 1 / (60 + 2)
    assert math.isclose(fused["d1"], expected, rel_tol=1e-12)
    # d2: rank 2 in A, rank 1 in B -> same total (symmetric example)
    assert math.isclose(fused["d2"], fused["d1"], rel_tol=1e-12)


def test_rrf_uses_only_ranks_not_raw_scores():
    # Scores are wildly different across systems but ranks are what matter.
    a = [Hit(doc_id="d1", score=0.001, rank=1), Hit(doc_id="d2", score=0.0001, rank=2)]
    b = [Hit(doc_id="d3", score=50.0, rank=1), Hit(doc_id="d1", score=40.0, rank=2)]
    fused = reciprocal_rank_fusion([a, b], k=60)
    # d1 and d3 both have one rank-1 and (for d1) one rank-2 -> d1 leads
    assert fused[0].doc_id == "d1"
    scores = {hit.doc_id: hit.score for hit in fused}
    assert math.isclose(scores["d1"], 1 / 61 + 1 / 62, rel_tol=1e-12)
    assert math.isclose(scores["d3"], 1 / 61, rel_tol=1e-12)


def test_rrf_keeps_documents_from_a_single_ranking():
    a = _ranking("d1", "d2")
    b = _ranking("d2", "d3")
    fused_ids = {hit.doc_id for hit in reciprocal_rank_fusion([a, b])}
    assert fused_ids == {"d1", "d2", "d3"}


def test_rrf_fusion_is_deterministic_with_score_ties():
    a = _ranking("x", "y")
    b = _ranking("y", "x")
    first = reciprocal_rank_fusion([a, b])
    second = reciprocal_rank_fusion([a, b])
    assert [(h.doc_id, h.score, h.rank) for h in first] == [
        (h.doc_id, h.score, h.rank) for h in second
    ]
    # exact tie -> doc_id ascending breaks it
    assert first[0].doc_id == "x" and first[1].doc_id == "y"


def test_rrf_handles_duplicate_doc_ids_within_one_ranking():
    a = [Hit(doc_id="d1", score=5.0, rank=1),
         Hit(doc_id="d1", score=4.0, rank=2),
         Hit(doc_id="d2", score=3.0, rank=3)]
    fused = reciprocal_rank_fusion([a], k=60)
    scores = {hit.doc_id: hit.score for hit in fused}
    assert math.isclose(scores["d1"], 1 / 61, rel_tol=1e-12)  # best rank only
    assert math.isclose(scores["d2"], 1 / 63, rel_tol=1e-12)


def test_rrf_ranks_are_reassigned_from_one():
    fused = reciprocal_rank_fusion([_ranking("a", "b"), _ranking("b", "a")])
    assert [hit.rank for hit in fused] == [1, 2]


def test_rrf_top_k_truncates():
    a = _ranking("d1", "d2", "d3", "d4")
    fused = reciprocal_rank_fusion([a], top_k=2)
    assert len(fused) == 2
    assert fused[0].doc_id == "d1"


def test_rrf_rejects_non_positive_k():
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion([_ranking("d1")], k=0)


def test_fuse_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown fusion method"):
        fuse([_ranking("d1")], method="weighted_sum")


def test_default_rrf_k_is_standard_constant():
    assert DEFAULT_RRF_K == 60


# --------------------------------------------------------------------------- #
# HybridRetriever integration (duck-typed component retrievers)
# --------------------------------------------------------------------------- #
class _StaticRetriever:
    """Returns a fixed ranking; records call arguments."""

    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.calls: list[dict] = []

    def retrieve(self, query_text, top_k=100, allowed_doc_ids=None):
        self.calls.append(
            {"query_text": query_text, "top_k": top_k,
             "allowed_doc_ids": allowed_doc_ids}
        )
        return self.hits[:top_k]


CONFIG = {
    "fusion": {"method": "rrf", "rrf_k": 60},
    "candidates": {"bm25": 3, "dense": 2},
}


def test_hybrid_requests_configured_candidate_counts_and_fuses():
    bm25 = _StaticRetriever(_ranking("d1", "d2", "d3"))
    dense = _StaticRetriever(_ranking("d3", "d4"))
    hybrid = HybridRetriever(bm25, dense, CONFIG)

    hits = hybrid.retrieve("en fru", top_k=10)

    assert bm25.calls[0]["top_k"] == 3
    assert dense.calls[0]["top_k"] == 2
    ids = [h.doc_id for h in hits]
    # d3: 1/61(dense)+1/63(bm25); d1: 1/62; d2/d4: 1/62 tie -> doc_id order
    assert ids == ["d3", "d1", "d2", "d4"]
    assert hits[0].doc_id == "d3"
    assert hits[0].rank == 1


def test_hybrid_passes_candidate_restriction_to_both_systems():
    bm25 = _StaticRetriever(_ranking("d1", "d2"))
    dense = _StaticRetriever(_ranking("d2", "d1"))
    hybrid = HybridRetriever(bm25, dense, CONFIG)
    hybrid.retrieve("q", top_k=5, allowed_doc_ids=["d1", "d2"])
    assert bm25.calls[0]["allowed_doc_ids"] == ["d1", "d2"]
    assert dense.calls[0]["allowed_doc_ids"] == ["d1", "d2"]


def test_hybrid_top_k_truncates_fused_output():
    bm25 = _StaticRetriever(_ranking("d1", "d2", "d3"))
    dense = _StaticRetriever(_ranking("d4", "d5"))
    hybrid = HybridRetriever(bm25, dense, CONFIG)
    hits = hybrid.retrieve("q", top_k=2)
    assert len(hits) == 2
    assert [h.rank for h in hits] == [1, 2]


def test_hybrid_uses_rrf_k_from_config():
    bm25 = _StaticRetriever([Hit(doc_id="d1", score=10.0, rank=1)])
    dense = _StaticRetriever([Hit(doc_id="d1", score=0.9, rank=1)])
    config = {"fusion": {"method": "rrf", "rrf_k": 10}, "candidates": {}}
    hybrid = HybridRetriever(bm25, dense, config)
    hits = hybrid.retrieve("q", top_k=1)
    assert math.isclose(hits[0].score, 1 / 11 + 1 / 11, rel_tol=1e-12)


def test_hybrid_retrieve_batch_uses_query_ids():
    bm25 = _StaticRetriever(_ranking("d1", "d2"))
    dense = _StaticRetriever(_ranking("d2", "d1"))
    hybrid = HybridRetriever(bm25, dense, CONFIG)

    class Q:
        def __init__(self, qid, text):
            self.query_id = qid
            self.text = text

    results = hybrid.retrieve_batch(
        [Q("w1", "aaa"), Q("w2", "bbb")],
        top_k=3,
        allowed_doc_ids_by_query={"w1": ["d1", "d2"], "w2": ["d2"]},
    )
    assert set(results) == {"w1", "w2"}
    assert all(isinstance(hits[0], Hit) for hits in results.values())
    assert bm25.calls[1]["allowed_doc_ids"] == ["d2"]
