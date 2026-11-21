"""Checks for objective weighting, label-independent inference and split isolation."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from Assignment_3.src.advanced_hybrid import AdvancedHead, metadata_frame, word_class_weights
from Assignment_3.src.runtime import ROOT, sha256


def test_word_class_weights_equalize_supported_word_classes():
    words = np.array(["a"]*8+["b"]*2)
    labels = np.array([0]*4+[1]*4+[0,1])
    weights = word_class_weights(words, labels)
    totals = [weights[(words==word)&(labels==label)].sum() for word in ["a","b"] for label in [0,1]]
    np.testing.assert_allclose(totals, np.full(4, 2.5))


def test_metadata_never_includes_gold_labels():
    first = pd.DataFrame({"word":["a"], "pos":["NN"], "period_label":["1900"], "label_value":[1]})
    second = first.assign(label_value=0)
    pd.testing.assert_frame_equal(metadata_frame(first), metadata_frame(second))
    assert "label_value" not in metadata_frame(first)


def test_expert_predictions_do_not_read_evaluation_labels():
    frame = pd.DataFrame({"word":["a","a","b","b"], "label_value":[0,1,0,1]})
    x = np.array([[-1,0],[1,0],[0,-1],[0,1]], dtype=float)
    views = {"linear": sparse.csr_matrix(x), "dense":x}
    head = AdvancedHead({"kind":"experts", "C":1., "global_mix":.5}, 42).fit(views, frame)
    np.testing.assert_array_equal(head.decision_function(views,frame), head.decision_function(views,frame.assign(label_value=1-frame.label_value)))


RUN = ROOT / "Assignment_3/runs/advanced_hybrid"


@pytest.mark.skipif(not (RUN/"artifact_manifest.json").exists(), reason="Advanced run not complete")
def test_advanced_artifacts_and_inner_oof_are_isolated():
    frozen = json.loads((RUN/"frozen_selections.json").read_text(encoding="utf-8"))
    assert frozen["all_seeds_selected_before_outer_test_scoring"]
    oof = pd.read_csv(RUN/"predictions/inner_oof_margins.csv")
    for seed in [42,100,2023,2026,777]:
        split = pd.read_csv(RUN/f"splits/outer_seed_{seed}.csv")
        assert set(oof[oof.seed==seed].sentence_id) == set(split[split.role=="outer_train"].sentence_id)
        assert not set(oof[oof.seed==seed].sentence_id) & set(split[split.role=="outer_test"].sentence_id)
        inner = pd.read_csv(RUN/f"splits/inner_seed_{seed}.csv")
        for _, fold in inner.groupby("fold"):
            assert not set(fold[fold.role=="fit"].text_group) & set(fold[fold.role=="validation"].text_group)
    manifest = json.loads((RUN/"artifact_manifest.json").read_text(encoding="utf-8"))
    for path,digest in manifest["files_sha256"].items():
        assert sha256(ROOT/path)==digest, path


def test_final_outputs_contain_only_the_selected_model_and_required_comparisons():
    results = ROOT / "Assignment_3/results"
    findings = json.loads((results/"findings.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(results/"seed_metrics.csv")
    selected = scores[scores.role=="selected_model"]
    assert set(selected.system)=={"advanced_hybrid"}
    assert len(selected)==5
    assert selected.official_f1.mean()==pytest.approx(findings["official_f1_mean"])
    assert not set(scores.system) & {"improved_hybrid", "context_only_control", "retrieval_only_control", "neural_pilot"}
    for name in ["improved_hybrid", "neural_pilot"]:
        assert not (ROOT/f"Assignment_3/runs/{name}").exists()
    manifest = json.loads((results/"artifact_manifest.json").read_text(encoding="utf-8"))
    for path,digest in manifest["files_sha256"].items():
        assert sha256(ROOT/path)==digest, path
