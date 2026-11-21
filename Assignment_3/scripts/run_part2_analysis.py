"""Compatibility entry point for code-only analysis of the completed experiments."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from Assignment_3.src.analysis import category as classify_hybrid_errors, exact_sign_flip, holm_adjust


if __name__ == "__main__":
    from Assignment_3.scripts.finalize_results import main
    main()
