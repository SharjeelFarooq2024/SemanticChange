"""Audit the pilot's scope, official score and saved adapted checkpoint."""
import json

import numpy as np
import pandas as pd
import pytest

from Assignment_3.src.evaluation import score_predictions
from Assignment_3.src.runtime import ROOT, sha256


RUN = ROOT / "Assignment_3/runs/neural_pilot"
pytestmark = pytest.mark.skipif(not (RUN / "pilot_summary.json").exists(), reason="Pilot not run")


def test_neural_pilot_validation_has_no_outer_test_or_shared_text():
    split = pd.read_csv(RUN / "splits.csv")
    outer = pd.read_csv(ROOT / "Assignment_3/runs/advanced_hybrid/splits/outer_seed_42.csv")
    assert not set(split.sentence_id) & set(outer[outer.role == "outer_test"].sentence_id)
    fit, validation = split[split.role == "fit"], split[split.role == "validation"]
    assert not set(fit.text_group) & set(validation.text_group)
    predictions = pd.read_json(RUN / "best_validation_predictions.jsonl", lines=True)
    assert set(predictions.sentence_id) == set(validation.sentence_id)


def test_neural_pilot_checkpoint_and_official_score():
    summary = json.loads((RUN / "pilot_summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((RUN / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    assert not summary["test_evaluation_performed"]
    assert metadata["checkpoint_replayed"] and metadata["encoder_update_l2"] > 0
    assert sha256(RUN / "best_pilot.pt") == metadata["checkpoint_sha256"]
    gold = pd.read_json(RUN / "validation_gold.jsonl", lines=True).rename(columns={"label": "label_value"})
    pred = pd.read_json(RUN / "best_validation_predictions.jsonl", lines=True).rename(columns={"label": "prediction"})
    scored = score_predictions(gold.merge(pred, on=["word", "sentence_id"], validate="one_to_one"))
    assert np.isclose(scored["official_f1"], summary["best_neural_validation_f1"], atol=1e-12)
    epochs = pd.read_csv(RUN / "validation_epochs.csv")
    assert np.isclose(epochs.official_f1.max(), summary["best_neural_validation_f1"])
