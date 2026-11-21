"""End-to-end pipeline orchestration for the three retrieval systems.

One entry point per run:

    pipeline.run_system("bm25", scope="per_word", top_k=100)

Steps: load config -> load + validate data -> preprocess/index (cached) ->
retrieve -> write predictions -> return a summary dict.

Dense and hybrid modules are imported lazily so BM25-only usage never
requires FAISS or model weights.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .data.data_loader import Query, RetrievalBundle, load_bundle_validated
from .retrieval.base import Hit
from .retrieval.prediction_io import (
    save_binary_predictions,
    save_qrels_reference,
    save_ranked_predictions,
    save_run_metadata,
)
from .utils.logging_utils import get_logger
from .utils.paths import PREDICTIONS_DIR, load_config, resolve_a2_path
from .utils.reproducibility import set_seed

VALID_SYSTEMS = ("bm25", "dense", "hybrid")


def _select_queries(queries: Sequence[Query], limit: Optional[int]) -> List[Query]:
    """Deterministically select a subset of queries (sorted by query_id)."""
    ordered = sorted(queries, key=lambda q: q.query_id)
    if limit is None or limit >= len(ordered):
        return list(ordered)
    return ordered[:limit]


def _resolve_allowed(
    bundle: RetrievalBundle, queries: Sequence[Query], scope: str
) -> Optional[Dict[str, Sequence[str]]]:
    """Return per-query candidate restrictions for ``per_word`` scope.

    ``global`` scope returns None (no restriction).
    """
    if scope == "global":
        return None
    if scope != "per_word":
        raise ValueError(f"Unknown scope '{scope}'; expected 'per_word' or 'global'")
    by_word = bundle.documents_by_query_word()
    allowed: Dict[str, Sequence[str]] = {}
    for query in queries:
        pool = by_word.get(query.word, [])
        if not pool:
            raise ValueError(
                f"No candidate documents for query '{query.query_id}' (word '{query.word}')"
            )
        allowed[query.query_id] = pool
    return allowed


def _run_bm25(
    bundle: RetrievalBundle,
    queries: Sequence[Query],
    config: Dict[str, Any],
    top_k: int,
    allowed: Optional[Dict[str, Sequence[str]]],
    log,
) -> Dict[str, List[Hit]]:
    from .retrieval.bm25_retriever import BM25Retriever

    index_path = resolve_a2_path(config["index"]["path"])
    retriever = BM25Retriever(config)
    log.info("Building/loading BM25 index (%s)", index_path)
    retriever.load_or_build(bundle.documents, index_path)
    log.info("BM25 retrieval: %d queries, top_k=%d", len(queries), top_k)
    return retriever.retrieve_batch(queries, top_k=top_k, allowed_doc_ids_by_query=allowed)


def _run_dense(
    bundle: RetrievalBundle,
    queries: Sequence[Query],
    config: Dict[str, Any],
    top_k: int,
    allowed: Optional[Dict[str, Sequence[str]]],
    encoder_name: Optional[str],
    log,
) -> Dict[str, List[Hit]]:
    from .retrieval.dense_retriever import DenseRetriever

    retriever = DenseRetriever(config, encoder_name=encoder_name, log=log)
    log.info("Building/loading dense index")
    retriever.build_index(bundle.documents)
    log.info("Dense retrieval: %d queries, top_k=%d", len(queries), top_k)
    return retriever.retrieve_batch(queries, top_k=top_k, allowed_doc_ids_by_query=allowed)


def _run_hybrid(
    bundle: RetrievalBundle,
    queries: Sequence[Query],
    data_config: Dict[str, Any],
    config: Dict[str, Any],
    top_k: int,
    allowed: Optional[Dict[str, Sequence[str]]],
    encoder_name: Optional[str],
    log,
) -> Dict[str, List[Hit]]:
    from .retrieval.dense_retriever import DenseRetriever
    from .retrieval.bm25_retriever import BM25Retriever
    from .retrieval.hybrid_retriever import HybridRetriever

    bm25_config = load_config("bm25_config")
    dense_config = load_config("dense_config")

    bm25 = BM25Retriever(bm25_config)
    bm25.load_or_build(bundle.documents, resolve_a2_path(bm25_config["index"]["path"]))

    dense = DenseRetriever(dense_config, encoder_name=encoder_name, log=log)
    dense.build_index(bundle.documents)

    hybrid = HybridRetriever(bm25, dense, config)
    log.info("Hybrid (RRF) retrieval: %d queries, top_k=%d", len(queries), top_k)
    return hybrid.retrieve_batch(queries, top_k=top_k, allowed_doc_ids_by_query=allowed)


def run_system(
    system: str,
    scope: Optional[str] = None,
    top_k: Optional[int] = None,
    encoder: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one retrieval system end to end and write its predictions.

    Args:
        system: ``bm25`` | ``dense`` | ``hybrid``.
        scope: ``per_word`` | ``global`` (defaults to the system config).
        top_k: Retrieval cutoff (defaults to the system config).
        encoder: Dense encoder key for dense/hybrid (defaults to config).
        limit: If set, only run the first N queries (sorted by query_id).

    Returns:
        Summary dict (also written to ``predictions/<system>/run_metadata.json``).
    """
    if system not in VALID_SYSTEMS:
        raise ValueError(f"Unknown system '{system}'; expected one of {VALID_SYSTEMS}")

    log = get_logger(system, log_file=True)
    data_config = load_config("data_config")
    system_config = load_config(f"{system}_config")
    set_seed(int(data_config.get("seed", 42)))

    scope = scope or system_config.get("retrieval", {}).get("scope", "per_word")
    top_k = int(top_k or system_config.get("retrieval", {}).get("top_k", 100))
    binary_top_k = int(system_config.get("binary_export", {}).get("top_k", 50))
    if binary_top_k > top_k:
        log.warning(
            "binary_export.top_k (%d) exceeds retrieval top_k (%d); "
            "candidates beyond top_k will be labelled 0",
            binary_top_k,
            top_k,
        )

    log.info("Loading data ...")
    bundle = load_bundle_validated(data_config)
    log.info(
        "Loaded %d documents, %d queries, %d qrels (%d relevant)",
        len(bundle.documents),
        len(bundle.queries),
        len(bundle.qrels),
        bundle.diagnostics.get("n_relevant_qrels", 0),
    )
    for warning in bundle.diagnostics.get("warnings", []):
        log.warning("data: %s", warning)

    queries = _select_queries(bundle.queries, limit)
    allowed = _resolve_allowed(bundle, queries, scope)
    log.info("Scope=%s | running %d/%d queries | top_k=%d",
             scope, len(queries), len(bundle.queries), top_k)

    if system == "bm25":
        results = _run_bm25(bundle, queries, system_config, top_k, allowed, log)
    elif system == "dense":
        results = _run_dense(bundle, queries, system_config, top_k, allowed, encoder, log)
    else:
        results = _run_hybrid(
            bundle, queries, data_config, system_config, top_k, allowed, encoder, log
        )

    # ------------------------------------------------------------------ #
    # Write predictions
    # ------------------------------------------------------------------ #
    # Non-default runs (limit, non-standard scope, or ablation encoder) get
    # their own dedicated directory so they never clobber full canonical results.
    default_encoder = system_config.get("encoder", {}).get("name")
    if limit is not None:
        dir_name = f"{system}_limit{len(queries)}"
    elif scope != "per_word":
        dir_name = f"{system}_{scope}"
    elif encoder and encoder != default_encoder:
        dir_name = f"{system}_{encoder}"
    else:
        dir_name = system

    out_dir = PREDICTIONS_DIR / dir_name
    ranked_path = save_ranked_predictions(results, out_dir / "ranked.csv")
    binary_path = save_binary_predictions(
        results, bundle, out_dir / "binary_predictions.csv", binary_top_k=binary_top_k
    )
    qrels_path = save_qrels_reference(bundle, PREDICTIONS_DIR / "qrels.csv")
    n_hits = sum(len(hits) for hits in results.values())

    summary: Dict[str, Any] = {
        "system": system,
        "scope": scope,
        "top_k": top_k,
        "encoder": encoder or system_config.get("encoder", {}).get("name"),
        "binary_top_k": binary_top_k,
        "n_queries_run": len(queries),
        "n_queries_total": len(bundle.queries),
        "n_hits": n_hits,
        "n_documents": len(bundle.documents),
        "outputs": {
            "ranked": str(ranked_path),
            "binary": str(binary_path),
            "qrels": str(qrels_path),
        },
        "config": system_config,
    }
    save_run_metadata(out_dir / "run_metadata.json", summary)
    log.info("Wrote %d ranked hits -> %s", n_hits, ranked_path)
    log.info("Wrote binary predictions -> %s", binary_path)
    return summary
