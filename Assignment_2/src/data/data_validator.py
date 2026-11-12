"""Validation of the retrieval bundle (documents / queries / qrels).

Failures raise :class:`ValidationError` with an actionable message; nothing
is silently dropped. Warnings are collected into a diagnostics dict so that
run logs can surface them without aborting.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:  # pragma: no cover
    from .data_loader import RetrievalBundle


class ValidationError(ValueError):
    """Raised when the retrieval inputs fail integrity checks."""


def validate_bundle(bundle: "RetrievalBundle") -> Dict[str, Any]:
    """Validate cross-file integrity of a loaded bundle.

    Checks:
      1. Non-empty documents, queries, qrels.
      2. Unique, non-empty document IDs and query IDs.
      3. No empty document text or query text.
      4. Every qrel references an existing document.
      5. Every qrel query_id matches an existing query.
      6. Every query word has at least one usage document (candidate pool > 0).
      7. Every usage word has a definition (otherwise its usages are
         unreachable as queries).

    Returns:
        A diagnostics dict with counts and any non-fatal warnings.

    Raises:
        ValidationError: listing *all* problems found (not just the first).
    """
    problems: List[str] = []
    warnings: List[str] = []

    documents, queries, qrels = bundle.documents, bundle.queries, bundle.qrels

    if not documents:
        problems.append("No documents loaded (usages.jsonl is empty).")
    if not queries:
        problems.append("No queries loaded (definitions.jsonl is empty).")
    if not qrels:
        problems.append("No qrels loaded (subtask2.jsonl is empty).")

    # --- ID uniqueness / emptiness -------------------------------------- #
    seen_doc_ids: set[str] = set()
    duplicate_doc_ids: List[str] = []
    empty_text_doc_ids: List[str] = []
    for doc in documents:
        if doc.doc_id in seen_doc_ids:
            duplicate_doc_ids.append(doc.doc_id)
        seen_doc_ids.add(doc.doc_id)
        if not doc.text.strip():
            empty_text_doc_ids.append(doc.doc_id)
    if duplicate_doc_ids:
        problems.append(
            f"Duplicate document IDs ({len(duplicate_doc_ids)}): "
            f"{_preview(duplicate_doc_ids)}"
        )
    if empty_text_doc_ids:
        problems.append(
            f"Documents with empty text ({len(empty_text_doc_ids)}): "
            f"{_preview(empty_text_doc_ids)}"
        )

    seen_query_ids: set[str] = set()
    duplicate_query_ids: List[str] = []
    empty_query_ids: List[str] = []
    for query in queries:
        if query.query_id in seen_query_ids:
            duplicate_query_ids.append(query.query_id)
        seen_query_ids.add(query.query_id)
        if not query.text.strip():
            empty_query_ids.append(query.query_id)
    if duplicate_query_ids:
        problems.append(f"Duplicate query IDs: {_preview(duplicate_query_ids)}")
    if empty_query_ids:
        problems.append(f"Queries with empty definition text: {_preview(empty_query_ids)}")

    # --- Qrel referential integrity -------------------------------------- #
    dangling_docs: List[str] = []
    dangling_queries: List[str] = []
    for qrel in qrels:
        if qrel.doc_id not in seen_doc_ids:
            dangling_docs.append(qrel.doc_id)
        if qrel.query_id not in seen_query_ids:
            dangling_queries.append(qrel.query_id)
    if dangling_docs:
        problems.append(
            f"Qrels reference unknown documents ({len(dangling_docs)}): "
            f"{_preview(sorted(set(dangling_docs)))}"
        )
    if dangling_queries:
        problems.append(
            f"Qrels reference unknown queries ({len(dangling_queries)}): "
            f"{_preview(sorted(set(dangling_queries)))}"
        )

    # --- Candidate pools --------------------------------------------------- #
    doc_words = {doc.word for doc in documents}
    query_words = {query.word for query in queries}
    queries_without_candidates = sorted(query_words - doc_words)
    if queries_without_candidates:
        problems.append(
            "Queries with no usage documents to retrieve from: "
            f"{_preview(queries_without_candidates)}"
        )
    words_without_definitions = sorted(doc_words - query_words)
    if words_without_definitions:
        warnings.append(
            "Usage words without a definition (unreachable as queries): "
            f"{_preview(words_without_definitions)}"
        )

    if problems:
        raise ValidationError(
            "Retrieval bundle failed validation:\n  - " + "\n  - ".join(problems)
        )

    qrels_per_query: Dict[str, int] = {}
    for qrel in qrels:
        qrels_per_query[qrel.query_id] = qrels_per_query.get(qrel.query_id, 0) + 1
    relevant_per_query: Dict[str, int] = {}
    for qrel in qrels:
        if qrel.relevance == 1:
            relevant_per_query[qrel.query_id] = relevant_per_query.get(qrel.query_id, 0) + 1

    diagnostics: Dict[str, Any] = {
        "n_documents": len(documents),
        "n_queries": len(queries),
        "n_qrels": len(qrels),
        "n_relevant_qrels": sum(1 for q in qrels if q.relevance == 1),
        "qrels_per_query": qrels_per_query,
        "relevant_per_query": relevant_per_query,
        "warnings": warnings,
    }
    return diagnostics


def _preview(values: List[str], limit: int = 10) -> str:
    shown = ", ".join(values[:limit])
    suffix = f", ... (+{len(values) - limit} more)" if len(values) > limit else ""
    return f"[{shown}{suffix}]"
