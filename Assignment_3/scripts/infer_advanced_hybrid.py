"""Alias for inference with the selected final model."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT))
from Assignment_3.scripts.infer import main


if __name__ == "__main__":
    main()
