"""Predict usages using the supervised hybrid checkpoint, without refitting."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from Assignment_3.src.evaluation import export_jsonl
from Assignment_3.src.runtime import sha256


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "Assignment_3/runs/improved_hybrid/checkpoints/seed_42/improved_hybrid.joblib")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    if sha256(args.checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint integrity failed")
    frame = pd.read_json(args.input, lines=True)
    if not {"word", "text"}.issubset(frame) or frame[["word", "text"]].isna().any().any():
        raise ValueError("Inputs require nonmissing word and text")
    if "sentence_id" not in frame:
        frame["sentence_id"] = [f"input_{i+1}" for i in range(len(frame))]
    if frame.duplicated(["word", "sentence_id"]).any():
        raise ValueError("Duplicate prediction identity")
    model = joblib.load(args.checkpoint)
    frame["prediction"] = model.predict(frame)
    export_jsonl(frame, args.output, "prediction")
    print(f"Saved {len(frame)} improved-hybrid predictions to {args.output}")
