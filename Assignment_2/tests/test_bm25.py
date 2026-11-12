"""Tests for the BM25 retriever (small synthetic corpus, no full runs)."""
from __future__ import annotations

from pathlib import Path

import pytest

from Assignment_2.src.data.data_loader import Document
from Assignment_2.src.retrieval.bm25_retriever import BM25IndexError, BM25Retriever

CONFIG = {"k1": 1.5, "b": 0.75, "tokenizer": {"lowercase": True}}


def _corpus() -> list[Document]:
    return [
        Document(doc_id="d1", text="the cat sat on the mat", word="cat"),
        Document(doc_id="d2", text="a dog barked at the cat", word="cat"),
        Document(doc_id="d3", text="dogs and cats are pets", word="dog"),
        Document(doc_id="d4", text="the mat was blue and soft", word="mat"),
        Document(doc_id="d5", text="birds sing in the garden", word="bird"),
        Document(doc_id="d6", text="cats chase dogs outside", word="cat"),
    ]


def _retriever() -> BM25Retriever:
    retriever = BM25Retriever(CONFIG)
    retriever.build_index(_corpus())
    return retriever


def test_returns_ranked_hits_with_descending_scores():
    hits = _retriever().retrieve("cat", top_k=3)
    assert len(hits) == 3
    assert [h.rank for h in hits] == [1, 2, 3]
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))
    assert hits[0].doc_id in {"d1", "d2", "d6"}


def test_top_k_is_respected():
    hits = _retriever().retrieve("cat", top_k=2)
    assert len(hits) == 2


def test_scores_are_query_dependent():
    retriever = _retriever()
    cat_hits = retriever.retrieve("cat", top_k=6)
    bird_hits = retriever.retrieve("bird", top_k=6)
    assert cat_hits[0].doc_id != bird_hits[0].doc_id or cat_hits[0].score != bird_hits[0].score


def test_allowed_doc_ids_restrict_candidates():
    hits = _retriever().retrieve("cat", top_k=10, allowed_doc_ids=["d4", "d5"])
    assert {h.doc_id for h in hits} <= {"d4", "d5"}


def test_allowed_doc_ids_with_unknown_id_raises():
    with pytest.raises(BM25IndexError, match="missing from the index"):
        _retriever().retrieve("cat", allowed_doc_ids=["ghost"])


def test_retrieve_before_build_raises():
    with pytest.raises(BM25IndexError, match="not built"):
        BM25Retriever(CONFIG).retrieve("cat")


def test_empty_corpus_build_raises():
    with pytest.raises(BM25IndexError, match="empty collection"):
        BM25Retriever(CONFIG).build_index([])


def test_results_are_deterministic():
    first = _retriever().retrieve("the cat", top_k=6)
    second = _retriever().retrieve("the cat", top_k=6)
    assert [(h.doc_id, h.score, h.rank) for h in first] == [
        (h.doc_id, h.score, h.rank) for h in second
    ]


def test_save_and_load_roundtrip_preserves_ranking(tmp_path: Path):
    path = tmp_path / "index.pkl"
    original = _retriever()
    original.save(path)
    loaded = BM25Retriever.load(path)
    assert loaded.k1 == original.k1
    assert loaded.b == original.b
    assert loaded.doc_ids == original.doc_ids
    before = original.retrieve("cat", top_k=6)
    after = loaded.retrieve("cat", top_k=6)
    assert [(h.doc_id, h.score) for h in before] == [(h.doc_id, h.score) for h in after]


def test_load_or_build_reuses_consistent_index(tmp_path: Path):
    path = tmp_path / "index.pkl"
    first = BM25Retriever(CONFIG)
    first.load_or_build(_corpus(), path)
    assert path.exists()
    second = BM25Retriever(CONFIG)
    second.load_or_build(_corpus(), path)
    assert second.doc_ids == first.doc_ids


def test_load_or_build_rebuilds_when_corpus_changes(tmp_path: Path):
    path = tmp_path / "index.pkl"
    first = BM25Retriever(CONFIG)
    first.load_or_build(_corpus(), path)
    changed = _corpus()[:-1] + [Document(doc_id="d9", text="new doc", word="new")]
    second = BM25Retriever(CONFIG)
    second.load_or_build(changed, path)
    assert second.doc_ids == [d.doc_id for d in changed]


def test_k1_and_b_are_configurable():
    tuned = BM25Retriever({"k1": 2.0, "b": 0.0, "tokenizer": {"lowercase": True}})
    tuned.build_index(_corpus())
    assert tuned.k1 == 2.0
    assert tuned.b == 0.0
    assert tuned.retrieve("cat", top_k=1)


def test_tokenize_keeps_temporal_information():
    doc = Document(doc_id="t1", text="Året 1883 bodde frun i staden", word="fru")
    retriever = BM25Retriever(CONFIG)
    retriever.build_index([doc])
    hits = retriever.retrieve("1883", top_k=1)
    assert hits[0].doc_id == "t1"
