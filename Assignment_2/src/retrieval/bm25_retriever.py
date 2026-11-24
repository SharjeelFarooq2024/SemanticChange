"""BM25 (Okapi) lexical retrieval built on the ``rank_bm25`` library.

- Index is built over the full document collection once and can be
  pickled to disk for reuse (``save`` / ``load``).
- Original ``sentence_id`` document IDs are preserved end to end.
- Candidate restriction (per-word scope) is applied at query time without
  rebuilding the index.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# rank_bm25 is imported lazily inside build_index

from ..preprocessing.tokenizer import Tokenizer
from ..utils.paths import ensure_dir
from .base import Hit, sort_and_rank


class BM25IndexError(RuntimeError):
    """Raised when retrieval is attempted before an index exists."""


class BM25Retriever:
    """Okapi BM25 over a document collection with configurable k1 / b."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = dict(config or {})
        self.k1 = float(self.config.get("k1", 1.5))
        self.b = float(self.config.get("b", 0.75))
        self.tokenizer = Tokenizer(self.config.get("tokenizer"))
        self._index: Optional[Any] = None
        self._doc_ids: List[str] = []
        self._doc_id_set: set[str] = set()

    # ------------------------------------------------------------------ #
    # Index construction
    # ------------------------------------------------------------------ #
    def build_index(self, documents: Iterable[Any]) -> None:
        """Build the BM25 index from objects with ``doc_id`` and ``text``."""
        doc_ids: List[str] = []
        tokenized_corpus: List[List[str]] = []
        for doc in documents:
            doc_id = str(doc.doc_id)
            tokens = self.tokenizer.tokenize(doc.text)
            doc_ids.append(doc_id)
            tokenized_corpus.append(tokens)

        if not tokenized_corpus:
            raise BM25IndexError("Cannot build a BM25 index over an empty collection")

        from rank_bm25 import BM25Okapi
        self._index = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self._doc_ids = doc_ids
        self._doc_id_set = set(doc_ids)

    @property
    def is_built(self) -> bool:
        return self._index is not None

    @property
    def doc_ids(self) -> List[str]:
        return list(self._doc_ids)

    def _require_index(self) -> Any:
        if self._index is None:
            raise BM25IndexError(
                "BM25 index is not built. Call build_index() or load() first."
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
        """Rank the collection for ``query_text`` and return the top hits.

        When ``allowed_doc_ids`` is provided, only those documents are
        eligible (the underlying BM25 scores are computed over the full
        corpus, which keeps scores comparable across scopes).
        """
        index = self._require_index()
        query_tokens = self.tokenizer.tokenize(query_text)
        scores = index.get_scores(query_tokens)  # aligned with self._doc_ids

        if allowed_doc_ids is None:
            candidates = zip(self._doc_ids, scores)
        else:
            allowed = set(str(doc_id) for doc_id in allowed_doc_ids)
            unknown = allowed - self._doc_id_set
            if unknown:
                raise BM25IndexError(
                    f"allowed_doc_ids contains IDs missing from the index: "
                    f"{sorted(unknown)[:5]}"
                )
            candidates = (
                (doc_id, score)
                for doc_id, score in zip(self._doc_ids, scores)
                if doc_id in allowed
            )

        scored: Dict[str, float] = {doc_id: float(score) for doc_id, score in candidates}
        return sort_and_rank(scored, top_k=top_k)

    def retrieve_batch(
        self,
        queries: Iterable[Any],
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

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Path | str) -> None:
        """Pickle the index, document-ID order, and tokenizer settings."""
        self._require_index()
        path = Path(path)
        ensure_dir(path.parent)
        payload = {
            "k1": self.k1,
            "b": self.b,
            "tokenizer_config": self.tokenizer.describe(),
            "doc_ids": self._doc_ids,
            "index": self._index,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path | str, config: Optional[Dict[str, Any]] = None) -> "BM25Retriever":
        """Reload a previously saved index; raises FileNotFoundError if absent."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found: {path}")
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        retriever = cls(config)
        retriever.k1 = float(payload["k1"])
        retriever.b = float(payload["b"])
        retriever.tokenizer = Tokenizer(payload.get("tokenizer_config"))
        retriever._index = payload["index"]
        retriever._doc_ids = list(payload["doc_ids"])
        retriever._doc_id_set = set(retriever._doc_ids)
        return retriever

    def load_or_build(self, documents: Iterable[Any], path: Path | str) -> None:
        """Load the saved index if present and consistent, otherwise build+save.

        Consistency check: the saved document-ID list must exactly match the
        IDs of ``documents`` (order included), otherwise the index is rebuilt.
        """
        path = Path(path)
        expected_ids = [str(doc.doc_id) for doc in documents]
        if path.exists():
            try:
                existing = self.load(path)
            except (OSError, pickle.PickleError, EOFError, KeyError):
                existing = None
            if existing is not None and existing.doc_ids == expected_ids:
                self._index = existing._index
                self._doc_ids = existing._doc_ids
                self._doc_id_set = existing._doc_id_set
                self.k1 = existing.k1
                self.b = existing.b
                self.tokenizer = existing.tokenizer
                return
        self.build_index(documents)
        self.save(path)
