"""Tests for the dense encoder cache and FAISS retriever.

A deterministic FakeEncoder is injected so unit tests never download
model weights. The real-encoder smoke test is opt-in via A2_RUN_SLOW=1.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from Assignment_2.src.data.data_loader import Document, Query
from Assignment_2.src.embeddings.embedding_manager import (
    EmbeddingCache,
    fingerprint_documents,
)
from Assignment_2.src.embeddings.encoder import DenseEncoder, EncoderError
from Assignment_2.src.retrieval.dense_retriever import DenseIndexError, DenseRetriever


# --------------------------------------------------------------------------- #
# Fake encoder (deterministic, no downloads)
# --------------------------------------------------------------------------- #
class FakeEncoder:
    """Bag-of-token-hashes encoder implementing the DenseEncoder interface.

    Uses a stable hash (md5) — Python's builtin ``hash()`` is randomised per
    process and would make retrieval results seed-dependent.
    """

    def __init__(self, dimension: int = 64, name: str = "fake") -> None:
        self._dimension = dimension
        self._name = name

    @property
    def encoder_name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dimension

    def identity(self, doc_fingerprint=None) -> dict:
        identity = {"encoder_key": self._name, "model_name": "fake-model",
                    "max_length": 128, "pooling": "mean", "normalize": True,
                    "query_prefix": "", "passage_prefix": ""}
        if doc_fingerprint is not None:
            identity["doc_fingerprint"] = doc_fingerprint
        return identity

    @staticmethod
    def _bucket(token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def encode(self, texts, kind: str = "passage", batch_size=None) -> np.ndarray:
        out = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in str(text).lower().split():
                out[row, self._bucket(token) % self._dimension] += 1.0
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


def _docs() -> list[Document]:
    return [
        Document(doc_id="d1", text="the cat sat on the mat", word="cat"),
        Document(doc_id="d2", text="a dog barked loudly", word="dog"),
        Document(doc_id="d3", text="cats and dogs are friends", word="cat"),
        Document(doc_id="d4", text="birds sing in the garden", word="bird"),
    ]


def _config(tmp_path: Path) -> dict:
    return {
        "encoder": {"name": "fake", "device": "cpu",
                    "models": {"fake": {"model_name": "fake-model"}}},
        "cache": {
            "model_cache_dir": str(tmp_path / "models"),
            "embeddings_dir": str(tmp_path / "emb"),
            "index_dir": str(tmp_path / "emb"),
        },
        "retrieval": {"top_k": 10, "scope": "per_word"},
        "binary_export": {"strategy": "top_k", "top_k": 5},
    }


def _retriever(tmp_path: Path, encoder: FakeEncoder | None = None) -> DenseRetriever:
    return DenseRetriever(_config(tmp_path), encoder=encoder or FakeEncoder())


# --------------------------------------------------------------------------- #
# Encoder configuration (no model loading)
# --------------------------------------------------------------------------- #
def test_encoder_identity_contains_cache_relevant_fields():
    encoder = DenseEncoder(
        "e5_small",
        models_config={
            "e5_small": {
                "model_name": "intfloat/multilingual-e5-small",
                "max_length": 512, "batch_size": 64, "pooling": "mean",
                "normalize": True, "query_prefix": "query: ",
                "passage_prefix": "passage: ",
            }
        },
    )
    identity = encoder.identity(doc_fingerprint="abc")
    assert identity["model_name"] == "intfloat/multilingual-e5-small"
    assert identity["query_prefix"] == "query: "
    assert identity["normalize"] is True
    assert identity["doc_fingerprint"] == "abc"
    assert not encoder.is_loaded  # lazy: nothing downloaded at construction


def test_unknown_encoder_raises():
    with pytest.raises(EncoderError, match="Unknown encoder"):
        DenseEncoder("nope", models_config={"e5_small": {}})


def test_encode_rejects_bad_kind():
    encoder = DenseEncoder(
        "e5_small",
        models_config={"e5_small": {"model_name": "x", "max_length": 8,
                                    "pooling": "mean", "normalize": True}},
    )
    with pytest.raises(EncoderError, match="kind"):
        encoder.encode(["text"], kind="document")


# --------------------------------------------------------------------------- #
# Embedding cache
# --------------------------------------------------------------------------- #
def test_fingerprint_changes_with_content_or_order():
    fp1 = fingerprint_documents(["a", "b"], ["x", "y"])
    fp2 = fingerprint_documents(["a", "b"], ["x", "z"])
    fp3 = fingerprint_documents(["b", "a"], ["y", "x"])
    fp4 = fingerprint_documents(["a", "b"], ["x", "y"])
    assert fp1 != fp2
    assert fp1 != fp3
    assert fp1 == fp4


def test_cache_save_and_load_roundtrip(tmp_path: Path):
    cache = EmbeddingCache(tmp_path / "emb")
    identity = FakeEncoder().identity(doc_fingerprint="fp1")
    ids = ["d1", "d2", "d3"]
    matrix = np.random.RandomState(0).rand(3, 16).astype(np.float32)
    cache.save(identity, ids, matrix)

    loaded, reason = cache.load(identity, ids)
    assert reason == "validated"
    assert loaded is not None
    np.testing.assert_array_equal(loaded, matrix)


def test_cache_rejected_on_identity_mismatch(tmp_path: Path):
    cache = EmbeddingCache(tmp_path / "emb")
    identity = FakeEncoder().identity(doc_fingerprint="fp1")
    ids = ["d1"]
    cache.save(identity, ids, np.zeros((1, 16), dtype=np.float32))

    changed = dict(identity, model_name="other-model")
    loaded, reason = cache.load(changed, ids)
    assert loaded is None
    assert "identity_mismatch" in reason


def test_cache_rejected_on_doc_set_change(tmp_path: Path):
    cache = EmbeddingCache(tmp_path / "emb")
    identity = FakeEncoder().identity(doc_fingerprint="fp1")
    cache.save(identity, ["d1", "d2"], np.zeros((2, 16), dtype=np.float32))
    loaded, reason = cache.load(identity, ["d1", "d3"])
    assert loaded is None
    assert reason == "doc_id_mismatch"


def test_cache_missing_returns_reason(tmp_path: Path):
    cache = EmbeddingCache(tmp_path / "emb")
    loaded, reason = cache.load(FakeEncoder().identity(doc_fingerprint="x"), ["d1"])
    assert loaded is None
    assert reason == "cache_missing"


# --------------------------------------------------------------------------- #
# Dense retrieval
# --------------------------------------------------------------------------- #
def test_build_index_and_retrieve_returns_valid_ranked_hits(tmp_path: Path):
    retriever = _retriever(tmp_path)
    retriever.build_index(_docs())
    assert retriever.is_built
    assert retriever.dimension == 64

    hits = retriever.retrieve("cat sat", top_k=3)
    assert 1 <= len(hits) <= 3
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert {h.doc_id for h in hits} <= {"d1", "d2", "d3", "d4"}
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))
    assert hits[0].doc_id == "d1"


def test_allowed_doc_ids_restrict_dense_candidates(tmp_path: Path):
    retriever = _retriever(tmp_path)
    retriever.build_index(_docs())
    hits = retriever.retrieve("cat dog", top_k=10, allowed_doc_ids=["d3", "d4"])
    assert {h.doc_id for h in hits} <= {"d3", "d4"}


def test_retrieve_before_build_raises(tmp_path: Path):
    with pytest.raises(DenseIndexError, match="not built"):
        _retriever(tmp_path).retrieve("query")


def test_embeddings_are_cached_and_reused(tmp_path: Path):
    retriever = _retriever(tmp_path)
    retriever.build_index(_docs())
    npz_files = list((tmp_path / "emb").glob("*.npz"))
    assert len(npz_files) == 1

    second = _retriever(tmp_path)
    second.build_index(_docs())  # must hit cache, not re-encode
    assert second.dimension == 64


def test_faiss_index_persists_and_reloads(tmp_path: Path):
    first = _retriever(tmp_path)
    first.build_index(_docs())
    before = first.retrieve("dogs friends", top_k=4)

    second = _retriever(tmp_path)
    second.build_index(_docs())
    assert second.doc_ids == first.doc_ids
    after = second.retrieve("dogs friends", top_k=4)
    assert [(h.doc_id, round(h.score, 5)) for h in before] == [
        (h.doc_id, round(h.score, 5)) for h in after
    ]


def test_rebuild_when_document_text_changes(tmp_path: Path):
    first = _retriever(tmp_path)
    first.build_index(_docs())

    changed = _docs()
    changed[0] = Document(doc_id="d1", text="totally different content", word="cat")
    second = _retriever(tmp_path)
    second.build_index(changed)
    hits = second.retrieve("totally different", top_k=1)
    assert hits[0].doc_id == "d1"


def test_retrieve_batch_keys_are_query_ids(tmp_path: Path):
    retriever = _retriever(tmp_path)
    retriever.build_index(_docs())
    queries = [Query(query_id="cat", text="cat sat"), Query(query_id="dog", text="dog barked")]
    results = retriever.retrieve_batch(queries, top_k=2)
    assert set(results) == {"cat", "dog"}


# --------------------------------------------------------------------------- #
# Optional real-encoder smoke test (downloads ~470 MB on first run)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.environ.get("A2_RUN_SLOW") != "1",
    reason="set A2_RUN_SLOW=1 to run the real-encoder smoke test",
)
def test_real_e5_encoder_smoke():
    from Assignment_2.src.utils.paths import load_config, resolve_a2_path

    config = load_config("dense_config")
    encoder_cfg = config["encoder"]
    encoder = DenseEncoder(
        "e5_small",
        models_config=encoder_cfg["models"],
        device=encoder_cfg.get("device", "auto"),
        model_cache_dir=resolve_a2_path(config["cache"]["model_cache_dir"]),
    )
    vectors = encoder.encode(["query: en fru", "passage: frun i staden"], kind="query")
    assert vectors.shape[0] == 2
    assert vectors.shape[1] == encoder.dimension
    assert vectors.dtype == np.float32
    # normalized rows (e5 profile has normalize: true)
    np.testing.assert_allclose(
        np.linalg.norm(vectors, axis=1), np.ones(2), rtol=1e-4, atol=1e-4
    )
