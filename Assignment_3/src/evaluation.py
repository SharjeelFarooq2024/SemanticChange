"""Strict adapter around the unmodified, commit-pinned official Subtask 2 scorer."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "Assignment_3/vendor/official"
sys.path.insert(0, str(ROOT / "Assignment_3/.deps"))
sys.path.insert(0, str(VENDOR))
_OFFICIAL = None


def official_module():
    global _OFFICIAL
    if _OFFICIAL is None:
        provenance = json.loads((VENDOR / "provenance.json").read_text(encoding="utf-8"))
        for name in ["evaluation.py", "evaluation_utils.py"]:
            expected = provenance["files"][name]["sha256"]
            if hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Official evaluator was modified: {name}")
        spec = importlib.util.spec_from_file_location("semantic_change_official", VENDOR / "evaluation.py")
        _OFFICIAL = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_OFFICIAL)
    return _OFFICIAL


def fast_official_f1(words, gold, prediction):
    """Vectorized equivalent, including sklearn's observed-class behavior.

    Used for cutoff sweeps; every saved run is also scored by the actual upstream
    function. A perfectly predicted single-class word scores 1, not 0.5.
    """
    words, gold, prediction = map(np.asarray, (words, gold, prediction))
    if not len(gold) or not (len(words) == len(gold) == len(prediction)):
        raise ValueError("Expected equal, nonempty arrays")
    if not set(np.unique(gold)) <= {0, 1} or not set(np.unique(prediction)) <= {0, 1}:
        raise ValueError("Labels must be binary")
    _, index = np.unique(words, return_inverse=True)
    size = index.max() + 1
    tp = np.bincount(index, weights=(gold == 1) & (prediction == 1), minlength=size)
    tn = np.bincount(index, weights=(gold == 0) & (prediction == 0), minlength=size)
    fp = np.bincount(index, weights=(gold == 0) & (prediction == 1), minlength=size)
    fn = np.bincount(index, weights=(gold == 1) & (prediction == 0), minlength=size)
    pden, nden = 2*tp + fp + fn, 2*tn + fp + fn
    pf = np.divide(2*tp, pden, out=np.zeros(size), where=pden > 0)
    nf = np.divide(2*tn, nden, out=np.zeros(size), where=nden > 0)
    n_classes = (pden > 0).astype(int) + (nden > 0).astype(int)
    return float(np.mean((pf+nf)/n_classes))


def score_predictions(frame):
    if frame.duplicated(["word", "sentence_id"]).any():
        raise ValueError("Duplicate prediction identity")
    if frame[["word", "sentence_id", "label_value", "prediction"]].isna().any().any():
        raise ValueError("Missing prediction fields")
    if not set(frame.label_value) <= {0, 1} or not set(frame.prediction) <= {0, 1}:
        raise ValueError("Labels must be binary")
    keys = frame.word + ";" + frame.sentence_id
    gold = dict(zip(keys, map(int, frame.label_value)))
    pred = dict(zip(keys, map(int, frame.prediction)))
    official = official_module().compute_f1_score(gold, pred, "in-memory validated predictions")
    if not np.isclose(official, fast_official_f1(frame.word, frame.label_value, frame.prediction), atol=1e-12, rtol=0):
        raise AssertionError("Fast scorer disagrees with official source")
    y, p = frame.label_value, frame.prediction
    positive_by_word = [f1_score(g.label_value, g.prediction, average="binary", zero_division=0)
                        for _, g in frame.groupby("word")]
    return {"official_f1": float(official),
            "pooled_macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "positive_f1_per_word": float(np.mean(positive_by_word)),
            "accuracy": float(accuracy_score(y, p)),
            # Same observed-gold-class definition as sklearn, also well-defined
            # for one-class lemma slices without confusion-matrix warnings.
            "balanced_accuracy": float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)])),
            "macro_precision": float(precision_score(y, p, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y, p, average="macro", zero_division=0))}


def export_jsonl(frame, path, prediction_column):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in frame[["word", "sentence_id", prediction_column]].itertuples(index=False, name=None):
            handle.write(json.dumps({"word": row[0], "sentence_id": row[1], "label": int(row[2])}, ensure_ascii=False) + "\n")
    return path
