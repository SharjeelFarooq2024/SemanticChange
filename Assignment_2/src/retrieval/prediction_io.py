"""Prediction output writers.

Two artifacts are written per system, per the agreed interface:

1. ``ranked.csv`` — the primary ranked output:
   columns ``query_id,doc_id,rank,score`` (one row per hit).

2. ``binary_predictions.csv`` — convenience export for the official-style
   Subtask 2 evaluation:
   columns ``query_id,sentence_id,word,label,score,rank`` covering the
   query word's *entire* candidate pool (label = 1 iff rank <= binary_k).

Threshold / operating-point selection is an evaluation decision; the ranked
CSV contains full scores so any cutoff can be recomputed downstream.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..data.data_loader import Qrel, Query, RetrievalBundle
from ..utils.paths import ensure_dir
from .base import Hit

RANKED_COLUMNS = ["query_id", "doc_id", "rank", "score"]
BINARY_COLUMNS = ["query_id", "sentence_id", "word", "label", "score", "rank"]


def save_ranked_predictions(
    results: Mapping[str, Sequence[Hit]],
    path: Path | str,
) -> Path:
    """Write all ranked lists to a single CSV (sorted by query_id, then rank)."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANKED_COLUMNS)
        writer.writeheader()
        for query_id in sorted(results):
            for hit in results[query_id]:
                writer.writerow(
                    {
                        "query_id": query_id,
                        "doc_id": hit.doc_id,
                        "rank": hit.rank,
                        "score": f"{hit.score:.6f}",
                    }
                )
    return path


def save_binary_predictions(
    results: Mapping[str, Sequence[Hit]],
    bundle: RetrievalBundle,
    path: Path | str,
    binary_top_k: int,
) -> Path:
    """Export per-usage binary labels for the full candidate pool of each query.

    A candidate is labelled ``1`` iff its rank in ``results`` is
    ``<= binary_top_k``; candidates absent from the ranking receive ``0``.
    """
    path = Path(path)
    ensure_dir(path.parent)

    candidates_by_word = bundle.documents_by_query_word()
    doc_meta = {doc.doc_id: doc for doc in bundle.documents}

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BINARY_COLUMNS)
        writer.writeheader()
        for query in sorted(bundle.queries, key=lambda q: q.query_id):
            hits = results.get(query.query_id, [])
            hit_by_doc = {hit.doc_id: hit for hit in hits}
            for doc_id in candidates_by_word.get(query.word, []):
                hit = hit_by_doc.get(doc_id)
                label = 1 if hit is not None and hit.rank <= binary_top_k else 0
                writer.writerow(
                    {
                        "query_id": query.query_id,
                        "sentence_id": doc_id,
                        "word": query.word,
                        "label": label,
                        "score": f"{hit.score:.6f}" if hit is not None else "",
                        "rank": hit.rank if hit is not None else "",
                    }
                )
    return path


def save_run_metadata(path: Path | str, payload: Dict[str, Any]) -> Path:
    """Write a JSON run summary (config snapshot, counts, timestamp)."""
    path = Path(path)
    ensure_dir(path.parent)
    envelope = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    path.write_text(
        json.dumps(envelope, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def save_qrels_reference(bundle: RetrievalBundle, path: Path | str) -> Path:
    """Write the qrels used by this run (``query_id,doc_id,relevance``).

    Convenience copy for downstream evaluation; the authoritative source
    remains the Assignment 1 raw file.
    """
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "doc_id", "relevance"])
        writer.writeheader()
        for qrel in sorted(bundle.qrels, key=lambda q: (q.query_id, q.doc_id)):
            writer.writerow(
                {
                    "query_id": qrel.query_id,
                    "doc_id": qrel.doc_id,
                    "relevance": qrel.relevance,
                }
            )
    return path
