"""Build (or refresh) the BM25 index without running retrieval.

Usage (from the repository root):
    python Assignment_2/scripts/build_bm25_index.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (path bootstrap)

from Assignment_2.src.data.data_loader import load_bundle_validated
from Assignment_2.src.retrieval.bm25_retriever import BM25Retriever
from Assignment_2.src.utils.logging_utils import get_logger
from Assignment_2.src.utils.paths import load_config, resolve_a2_path
from Assignment_2.src.utils.reproducibility import set_seed


def main() -> None:
    log = get_logger("build_bm25_index", log_file=True)
    data_config = load_config("data_config")
    bm25_config = load_config("bm25_config")
    set_seed(int(data_config.get("seed", 42)))

    bundle = load_bundle_validated(data_config)
    log.info("Loaded %d documents", len(bundle.documents))

    index_path = resolve_a2_path(bm25_config["index"]["path"])
    retriever = BM25Retriever(bm25_config)
    if index_path.exists():
        log.info("Index already exists at %s — verifying consistency", index_path)
    retriever.load_or_build(bundle.documents, index_path)
    log.info(
        "BM25 index ready: %d documents, k1=%.2f, b=%.2f -> %s",
        len(retriever.doc_ids),
        retriever.k1,
        retriever.b,
        index_path,
    )


if __name__ == "__main__":
    main()
