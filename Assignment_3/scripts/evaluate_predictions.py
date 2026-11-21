"""Score official-format JSONL predictions with the pinned upstream evaluator."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from Assignment_3.src.evaluation import official_module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"official_f1": official_module().evaluate_subtask2(args.gold, args.predictions),
                      "metric": "mean of within-word two-class macro-F1"}, indent=2))
