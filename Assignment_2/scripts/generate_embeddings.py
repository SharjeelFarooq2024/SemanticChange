"""Generate and cache document embeddings + the FAISS index (no retrieval).

Usage (from the repository root):
    python Assignment_2/scripts/generate_embeddings.py [--encoder e5_small|xlmr_base]
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from Assignment_2.src.data.data_loader import load_bundle_validated
from Assignment_2.src.retrieval.dense_retriever import DenseRetriever
from Assignment_2.src.utils.logging_utils import get_logger
from Assignment_2.src.utils.paths import load_config
from Assignment_2.src.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cached document embeddings")
    parser.add_argument(
        "--encoder",
        choices=["e5_small", "xlmr_base"],
        default=None,
        help="Encoder profile (defaults to dense_config encoder.name)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild even if a valid cache exists"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log = get_logger("generate_embeddings", log_file=True)
    data_config = load_config("data_config")
    dense_config = load_config("dense_config")
    set_seed(int(data_config.get("seed", 42)))

    bundle = load_bundle_validated(data_config)
    log.info("Loaded %d documents", len(bundle.documents))

    retriever = DenseRetriever(dense_config, encoder_name=args.encoder, log=log)
    retriever.build_index(bundle.documents, force_rebuild=args.force)
    log.info(
        "Embeddings ready: encoder=%s dim=%s docs=%d",
        retriever.encoder.encoder_name,
        retriever.dimension,
        len(retriever.doc_ids),
    )


if __name__ == "__main__":
    main()
