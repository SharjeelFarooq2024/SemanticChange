"""Regression checks for metric identity, leakage controls and complete run artifacts."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Assignment_3.src.evaluation import export_jsonl, fast_official_f1, official_module, score_predictions
from Assignment_3.src.experiments import make_split, rank_predictions, tune_cutoff
from Assignment_3.src.runtime import ROOT, load_data, sha256


def test_official_metric_counts_both_classes_within_each_word():
    frame = pd.DataFrame({"word": ["a", "a", "a", "a", "b"], "sentence_id": list("12345"),
                          "label_value": [0, 0, 0, 1, 1], "prediction": [0, 0, 0, 0, 1]})
    score = score_predictions(frame)
    # Word a has (negative F1=6/7, positive F1=0)/2. Word b scores 1.
    assert score["official_f1"] == pytest.approx(((6/7)/2+1)/2)
    assert score["official_f1"] != score["positive_f1_per_word"]


@pytest.mark.parametrize("seed", [1, 9, 42])
def test_fast_calibration_matches_actual_upstream(seed):
    rng = np.random.default_rng(seed)
    words = np.array(["a"]*13+["b"]*7+["c"])
    for _ in range(10):
        frame = pd.DataFrame({"word": words, "sentence_id": [str(i) for i in range(len(words))],
                              "label_value": rng.integers(0, 2, len(words)), "prediction": rng.integers(0, 2, len(words))})
        score = score_predictions(frame)
        assert fast_official_f1(words, frame.label_value, frame.prediction) == pytest.approx(score["official_f1"], abs=1e-12)


def test_official_loader_rejects_missing_or_extra_identities(tmp_path):
    frame = pd.DataFrame({"word": ["a", "a"], "sentence_id": ["1", "2"], "label_value": [0, 1], "prediction": [0, 1]})
    gold = export_jsonl(frame, tmp_path/"gold.jsonl", "label_value")
    missing = export_jsonl(frame.iloc[:1], tmp_path/"missing.jsonl", "prediction")
    with pytest.raises(ValueError, match="Missing expected"):
        official_module().evaluate_subtask2(gold, missing)
    extra = frame.copy()
    extra.loc[0, "sentence_id"] = "unknown"
    path = export_jsonl(extra, tmp_path/"extra.jsonl", "prediction")
    with pytest.raises(ValueError, match="Unknown usage"):
        official_module().evaluate_subtask2(gold, path)


def test_exact_duplicate_contexts_stay_in_one_group_for_every_seed():
    data, _ = load_data()
    for seed in [42, 100, 2023, 2026, 777]:
        train, test = make_split(data, "grouped_text", seed)
        assert not set(data.iloc[train].text_group) & set(data.iloc[test].text_group)
        assert not set(train) & set(test)
        assert len(train)+len(test) == len(data)


def test_cutoff_sweep_matches_official_file_metric_and_handles_missing_ranks():
    train = pd.DataFrame({"word": ["a", "a", "a", "b", "b", "b"], "sentence_id": list("123456"),
                          "label_value": [1, 0, 0, 0, 1, 1]})
    ranked = pd.DataFrame({"query_id": ["a", "a", "b", "b", "b"], "doc_id": list("12456"), "rank": [1, 2, 1, 2, 3]})
    k, score, choices = tune_cutoff(train, ranked, [1, 2, 3])
    official = []
    for cutoff in [1, 2, 3]:
        frame = train.assign(prediction=rank_predictions(train, ranked, cutoff))
        official.append(score_predictions(frame)["official_f1"])
    assert score == pytest.approx(max(official))
    assert k == int(np.argmax(official))+1
    assert rank_predictions(train, ranked, 3)[2] == 0


COMPLETED = ROOT / "Assignment_3/runs/completed"


@pytest.mark.skipif(not (COMPLETED/"artifact_manifest.json").exists(), reason="Full experiment artifacts not generated yet")
def test_completed_package_has_real_models_official_scores_and_clean_grouped_splits():
    manifest = json.loads((COMPLETED/"artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_seed_configurations"] == 210
    assert manifest["classifier_checkpoint_count"] == 100
    assert manifest["retrieval_checkpoint_count"] == 110
    for path, digest in manifest["files_sha256"].items():
        assert sha256(ROOT/path) == digest, path
    splits = pd.read_csv(COMPLETED/"tables/split_audit.csv")
    assert (splits[splits.protocol == "grouped_text"].shared_text_groups == 0).all()
    with zipfile.ZipFile(ROOT/"Assignment_3/final_package.zip") as archive:
        assert sum(n.endswith(".joblib") for n in archive.namelist()) == 105
        assert not any("runs/improved_hybrid/" in n or "runs/neural_pilot/" in n for n in archive.namelist())
        assert archive.testzip() is None
