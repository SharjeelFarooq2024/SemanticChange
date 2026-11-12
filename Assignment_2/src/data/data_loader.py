"""Data loading for Assignment 2 retrieval.

Retrieval formulation (agreed with the group):

- documents  = usages from ``usages.jsonl``         (doc_id  = sentence_id)
- queries    = target-sense definitions             (query_id = word)
- qrels      = Subtask 2 binary labels              (word -> sentence_id, rel in {0,1})

The Assignment 1 raw files are referenced in place and never modified.
Reuses Assignment 1's ``load_jsonl`` for standard files while adding a
strict loader that reports malformed lines with line numbers instead of
silently dropping records.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..utils.paths import REPO_ROOT, load_config, resolve_repo_path


class DataLoadError(ValueError):
    """Raised when a required input file is missing or a record is malformed."""


# --------------------------------------------------------------------------- #
# Record types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Document:
    """A retrievable document (one usage sentence)."""

    doc_id: str
    text: str
    word: str = ""
    period_label: str = ""
    year: str = ""
    pos: str = ""

    @property
    def metadata(self) -> Dict[str, str]:
        return {
            "word": self.word,
            "period_label": self.period_label,
            "year": self.year,
            "pos": self.pos,
        }


@dataclass(frozen=True)
class Query:
    """A retrieval query (one target-sense definition)."""

    query_id: str
    text: str
    word: str = ""

    @property
    def metadata(self) -> Dict[str, str]:
        return {"word": self.word}


@dataclass(frozen=True)
class Qrel:
    """One relevance judgment: does ``doc_id`` express the sense of ``query_id``?"""

    query_id: str
    doc_id: str
    relevance: int
    word: str = ""
    period_label: str = ""


@dataclass
class RetrievalBundle:
    """Validated collection of documents, queries, and qrels."""

    documents: List[Document]
    queries: List[Query]
    qrels: List[Qrel]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def documents_by_query_word(self) -> Dict[str, List[str]]:
        """Map each query's target word to the doc_ids of its usages."""
        mapping: Dict[str, List[str]] = {}
        for doc in self.documents:
            mapping.setdefault(doc.word, []).append(doc.doc_id)
        return mapping

    def qrels_by_query(self) -> Dict[str, Dict[str, int]]:
        """Map query_id -> {doc_id: relevance}."""
        mapping: Dict[str, Dict[str, int]] = {}
        for qrel in self.qrels:
            mapping.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
        return mapping


# --------------------------------------------------------------------------- #
# Low-level JSONL reading
# --------------------------------------------------------------------------- #
def read_jsonl_strict(path: Path | str) -> List[Dict[str, Any]]:
    """Read a JSONL file, raising :class:`DataLoadError` with line numbers
    for malformed JSON instead of silently dropping the record."""
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"Required file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DataLoadError(
                    f"Malformed JSON in {path.name} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise DataLoadError(
                    f"Expected a JSON object in {path.name} at line {line_number}, "
                    f"got {type(row).__name__}"
                )
            rows.append(row)
    return rows


def _require_fields(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    """Raise if any record lacks one of the required fields."""
    for index, row in enumerate(rows, start=1):
        missing = [name for name in fields if name not in row]
        if missing:
            raise DataLoadError(
                f"{path.name} record #{index} is missing required field(s): {', '.join(missing)}"
            )


# --------------------------------------------------------------------------- #
# Public loaders
# --------------------------------------------------------------------------- #
def load_documents(config: Optional[Dict[str, Any]] = None) -> List[Document]:
    """Load all usage documents, preserving original ``sentence_id`` values."""
    config = config or load_config("data_config")
    data_root = resolve_repo_path(config["data"]["root"])
    path = data_root / config["data"]["usages"]
    rows = read_jsonl_strict(path)
    _require_fields(path, rows, ["sentence_id", "text"])

    documents: List[Document] = []
    for index, row in enumerate(rows, start=1):
        doc_id = str(row.get("sentence_id", "")).strip()
        if not doc_id:
            raise DataLoadError(f"{path.name} record #{index} has an empty sentence_id")
        documents.append(
            Document(
                doc_id=doc_id,
                text=str(row.get("text", "") or ""),
                word=str(row.get("word", "") or ""),
                period_label=str(row.get("period_label", "") or ""),
                year=str(row.get("year", "") or ""),
                pos=str(row.get("pos", "") or ""),
            )
        )
    return documents


def load_queries(config: Optional[Dict[str, Any]] = None) -> List[Query]:
    """Load target-sense definitions as queries.

    ``query_id`` is the target word (stable, unique, and directly joinable
    with qrels built from ``subtask2.jsonl``).
    """
    config = config or load_config("data_config")
    data_root = resolve_repo_path(config["data"]["root"])
    path = data_root / config["data"]["definitions"]
    rows = read_jsonl_strict(path)
    _require_fields(path, rows, ["word", "definition"])

    queries: List[Query] = []
    for index, row in enumerate(rows, start=1):
        word = str(row.get("word", "")).strip()
        if not word:
            raise DataLoadError(f"{path.name} record #{index} has an empty word")
        queries.append(
            Query(query_id=word, text=str(row.get("definition", "") or ""), word=word)
        )
    return queries


def load_qrels(config: Optional[Dict[str, Any]] = None) -> List[Qrel]:
    """Load Subtask 2 labels as relevance judgments.

    Relevance: ``1`` = usage expresses the target sense, ``0`` = it does not.
    """
    config = config or load_config("data_config")
    data_root = resolve_repo_path(config["data"]["root"])
    path = data_root / config["data"]["subtask2"]
    rows = read_jsonl_strict(path)
    _require_fields(path, rows, ["word", "sentence_id", "label"])

    qrels: List[Qrel] = []
    for index, row in enumerate(rows, start=1):
        word = str(row.get("word", "")).strip()
        doc_id = str(row.get("sentence_id", "")).strip()
        label = row.get("label")
        if label not in (0, 1):
            raise DataLoadError(
                f"{path.name} record #{index} has label {label!r}; expected 0 or 1"
            )
        if not word or not doc_id:
            raise DataLoadError(
                f"{path.name} record #{index} has empty word or sentence_id"
            )
        qrels.append(
            Qrel(
                query_id=word,
                doc_id=doc_id,
                relevance=int(label),
                word=word,
                period_label=str(row.get("period_label", "") or ""),
            )
        )
    return qrels


def load_bundle(config: Optional[Dict[str, Any]] = None) -> RetrievalBundle:
    """Load documents, queries, and qrels together (unvalidated)."""
    config = config or load_config("data_config")
    return RetrievalBundle(
        documents=load_documents(config),
        queries=load_queries(config),
        qrels=load_qrels(config),
    )


def load_bundle_validated(config: Optional[Dict[str, Any]] = None) -> RetrievalBundle:
    """Load and validate the retrieval bundle; raises on any integrity error."""
    from .data_validator import validate_bundle

    bundle = load_bundle(config)
    bundle.diagnostics = validate_bundle(bundle)
    return bundle
