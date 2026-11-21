"""Predict JSONL usages using a saved fitted checkpoint."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from Assignment_3.src.inference import predict_checkpoint
from Assignment_3.src.runtime import load_config, sha256
from Assignment_3.src.evaluation import export_jsonl


DEFAULT_CHECKPOINT = ROOT / "Assignment_3/runs/advanced_hybrid/checkpoints/seed_42.joblib"


def predict_inputs(inputs, checkpoint=DEFAULT_CHECKPOINT):
    checkpoint = Path(checkpoint)
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    if "input_kind" in metadata:
        config = load_config()
        return predict_checkpoint(checkpoint, inputs, ROOT / config["output_dir"])
    if sha256(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint integrity check failed")
    if not {"word", "text"}.issubset(inputs) or inputs[["word", "text"]].isna().any().any():
        raise ValueError("Inputs require nonmissing word and text")
    data = inputs.copy()
    if "sentence_id" not in data:
        data["sentence_id"] = [f"input_{i+1}" for i in range(len(data))]
    if data.duplicated(["word", "sentence_id"]).any():
        raise ValueError("Duplicate inference identity")
    data["prediction"] = joblib.load(checkpoint).predict(data)
    return data[["word", "sentence_id", "prediction"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = predict_inputs(pd.read_json(args.input, lines=True), args.checkpoint)
    export_jsonl(result, args.output, "prediction")
    print(f"Saved {len(result)} checkpoint predictions to {args.output}")


if __name__ == "__main__":
    main()
