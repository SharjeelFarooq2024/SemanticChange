"""Run dense retrieval and write predictions/dense/.

Usage (from the repository root):
    python Assignment_2/scripts/run_dense.py [--encoder e5_small|xlmr_base]
                                             [--scope per_word|global]
                                             [--top-k 100] [--limit N]
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from Assignment_2.src.pipeline import run_system


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense retrieval")
    parser.add_argument("--encoder", choices=["e5_small", "xlmr_base"], default=None)
    parser.add_argument("--scope", choices=["per_word", "global"], default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N queries (sorted by query_id)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_system(
        "dense",
        scope=args.scope,
        top_k=args.top_k,
        encoder=args.encoder,
        limit=args.limit,
    )
    print(
        f"[dense] {summary['n_queries_run']} queries, {summary['n_hits']} hits "
        f"-> {summary['outputs']['ranked']}"
    )


if __name__ == "__main__":
    main()
