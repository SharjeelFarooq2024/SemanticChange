"""Tests for Assignment 2 data loading and validation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from Assignment_2.src.data.data_loader import (
    DataLoadError,
    Document,
    Qrel,
    Query,
    RetrievalBundle,
    load_bundle_validated,
    load_documents,
    load_qrels,
    load_queries,
    read_jsonl_strict,
)
from Assignment_2.src.data.data_validator import ValidationError, validate_bundle


# --------------------------------------------------------------------------- #
# Real Assignment 1 dataset
# --------------------------------------------------------------------------- #
def test_real_dataset_loads_with_expected_counts():
    documents = load_documents()
    queries = load_queries()
    qrels = load_qrels()
    assert len(documents) == 1983
    assert len(queries) == 10
    assert len(qrels) == 1983


def test_document_and_query_ids_are_preserved():
    documents = load_documents()
    queries = load_queries()
    assert len({doc.doc_id for doc in documents}) == len(documents)
    assert {q.query_id for q in queries} == {
        "kvinna", "skär", "fröken", "dumpa", "exportera",
        "fru", "herre", "förort", "suga", "motiv",
    }
    # doc IDs look like the raw sentence_ids (no transformation)
    assert all("_" in doc.doc_id for doc in documents[:20])


def test_qrels_join_to_documents_and_are_binary():
    bundle = load_bundle_validated()
    doc_ids = {doc.doc_id for doc in bundle.documents}
    assert all(qrel.doc_id in doc_ids for qrel in bundle.qrels)
    assert {qrel.relevance for qrel in bundle.qrels} <= {0, 1}
    assert bundle.diagnostics["n_relevant_qrels"] == 705
    assert bundle.diagnostics["warnings"] == []


def test_documents_by_query_word_partitions_the_corpus():
    bundle = load_bundle_validated()
    by_word = bundle.documents_by_query_word()
    assert sum(len(ids) for ids in by_word.values()) == len(bundle.documents)
    assert len(by_word["fru"]) == 274


# --------------------------------------------------------------------------- #
# Strict JSONL behaviour
# --------------------------------------------------------------------------- #
def test_read_jsonl_strict_reports_malformed_line_number(tmp_path: Path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"sentence_id": "a", "text": "ok"}\nnot json\n', encoding="utf-8")
    with pytest.raises(DataLoadError, match="line 2"):
        read_jsonl_strict(path)


def test_read_jsonl_strict_missing_file():
    with pytest.raises(DataLoadError, match="not found"):
        read_jsonl_strict(Path("definitely/missing/file.jsonl"))


def test_loader_rejects_missing_required_field(tmp_path: Path):
    path = tmp_path / "usages.jsonl"
    path.write_text('{"text": "no id here"}\n', encoding="utf-8")
    config = {"data": {"root": str(tmp_path), "usages": "usages.jsonl"}}
    with pytest.raises(DataLoadError, match="sentence_id"):
        load_documents(config)


# --------------------------------------------------------------------------- #
# Validator failure modes (synthetic bundles)
# --------------------------------------------------------------------------- #
def _bundle(documents, queries, qrels) -> RetrievalBundle:
    return RetrievalBundle(documents=documents, queries=queries, qrels=qrels)


def _ok_fixture() -> tuple[list, list, list]:
    documents = [
        Document(doc_id="d1", text="en mening", word="fru"),
        Document(doc_id="d2", text="annan mening", word="fru"),
    ]
    queries = [Query(query_id="fru", text="definition", word="fru")]
    qrels = [Qrel(query_id="fru", doc_id="d1", relevance=1, word="fru")]
    return documents, queries, qrels


def test_validator_accepts_consistent_bundle():
    documents, queries, qrels = _ok_fixture()
    diagnostics = validate_bundle(_bundle(documents, queries, qrels))
    assert diagnostics["n_documents"] == 2
    assert diagnostics["n_relevant_qrels"] == 1


def test_validator_rejects_duplicate_document_ids():
    documents, queries, qrels = _ok_fixture()
    documents.append(Document(doc_id="d1", text="duplicate id", word="fru"))
    with pytest.raises(ValidationError, match="Duplicate document IDs"):
        validate_bundle(_bundle(documents, queries, qrels))


def test_validator_rejects_empty_document_text():
    documents, queries, qrels = _ok_fixture()
    documents.append(Document(doc_id="d3", text="   ", word="fru"))
    with pytest.raises(ValidationError, match="empty text"):
        validate_bundle(_bundle(documents, queries, qrels))


def test_validator_rejects_qrel_for_unknown_document():
    documents, queries, qrels = _ok_fixture()
    qrels.append(Qrel(query_id="fru", doc_id="ghost", relevance=1))
    with pytest.raises(ValidationError, match="unknown documents"):
        validate_bundle(_bundle(documents, queries, qrels))


def test_validator_rejects_query_without_candidates():
    documents, queries, qrels = _ok_fixture()
    queries.append(Query(query_id="motiv", text="def", word="motiv"))
    with pytest.raises(ValidationError, match="no usage documents"):
        validate_bundle(_bundle(documents, queries, qrels))


def test_validator_rejects_empty_query_set():
    documents, queries, qrels = _ok_fixture()
    with pytest.raises(ValidationError, match="No queries loaded"):
        validate_bundle(_bundle(documents, [], qrels))
