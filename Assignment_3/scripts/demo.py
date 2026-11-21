"""Live retrieval or explicitly labeled saved-ranking fallback. JSON output."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from Assignment_3.src.inference import retrieval_demo
from Assignment_3.src.runtime import dump_json, load_config
from Assignment_3.scripts.infer import DEFAULT_CHECKPOINT, predict_inputs
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["checkpoint", "saved", "live"], default="checkpoint")
    parser.add_argument("--input", type=Path, default=ROOT / "Assignment_3/runs/completed/demo_inputs.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--system", choices=["bm25", "dense", "hybrid", "dense_xlmr"], default="hybrid")
    parser.add_argument("--word", default="motiv")
    parser.add_argument("--definition")
    parser.add_argument("--top-k", default=5, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config()
    if args.mode == "checkpoint":
        predictions = predict_inputs(pd.read_json(args.input, lines=True), args.checkpoint)
        result = {"system": "advanced_hybrid", "mode": "fitted_checkpoint_inference", "predictions": predictions.to_dict(orient="records")}
    else:
        result = retrieval_demo(config, ROOT / config["output_dir"], args.word, args.system, args.mode, args.definition, args.top_k)
    if args.output:
        dump_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
