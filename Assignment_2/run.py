#!/usr/bin/env python
"""Assignment 2 — unified entry point for the core retrieval pipeline.

Usage (from the repository root):

    python Assignment_2/run.py --system bm25
    python Assignment_2/run.py --system dense
    python Assignment_2/run.py --system hybrid
    python Assignment_2/run.py --system all

Optional:
    --scope per_word|global     candidate restriction (default from config)
    --top-k N                   retrieval cutoff (default from config)
    --encoder e5_small|xlmr_base  dense encoder profile
    --limit N                   run only the first N queries (sorted by id)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python Assignment_2/run.py` from any directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Assignment_2.src.pipeline import VALID_SYSTEMS, run_system  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Assignment 2 retrieval pipeline (BM25 / dense / hybrid)"
    )
    parser.add_argument(
        "--system",
        required=True,
        choices=[*VALID_SYSTEMS, "all"],
        help="Retrieval system to run ('all' runs the three systems in order)",
    )
    parser.add_argument("--scope", choices=["per_word", "global"], default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--encoder", choices=["e5_small", "xlmr_base"], default=None,
                        help="Dense encoder profile (dense and hybrid only)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N queries (sorted by query_id)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    systems = list(VALID_SYSTEMS) if args.system == "all" else [args.system]

    summaries = []
    for system in systems:
        summary = run_system(
            system,
            scope=args.scope,
            top_k=args.top_k,
            encoder=args.encoder,
            limit=args.limit,
        )
        summaries.append(summary)
        print(
            f"[{system}] {summary['n_queries_run']} queries | "
            f"{summary['n_hits']} hits | "
            f"ranked -> {summary['outputs']['ranked']}"
        )

    print("\nDone. Prediction files:")
    for summary in summaries:
        print(f"  {summary['system']:6s} {summary['outputs']['ranked']}")
        print(f"  {summary['system']:6s} {summary['outputs']['binary']}")


if __name__ == "__main__":
    main()
