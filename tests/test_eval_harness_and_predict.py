"""Unit and integration tests for eval_harness.py and scripts/predict.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Assignment_3/.deps"))
sys.path.insert(0, str(ROOT / "Assignment_3/vendor/official"))

import eval_harness as eh
import evaluation as vendor_eval


def test_official_f1_and_fast_match_vendored_scorer_with_edge_cases():
    rng = np.random.default_rng(123)
    words = np.array(["a"] * 12 + ["b"] * 10 + ["single_pos"] * 5 + ["single_neg"] * 5)

    for _ in range(10):
        gold = np.concatenate([
            rng.integers(0, 2, 22),
            np.ones(5, dtype=int),
            np.zeros(5, dtype=int),
        ])
        pred = np.concatenate([
            rng.integers(0, 2, 22),
            np.ones(5, dtype=int),      # perfectly predicted single class
            rng.integers(0, 2, 5),      # imperfect single class
        ])
        keys = [f"{w};id_{i}" for i, w in enumerate(words)]
        gold_dict = dict(zip(keys, map(int, gold)))
        pred_dict = dict(zip(keys, map(int, pred)))

        expected = vendor_eval.compute_f1_score(gold_dict, pred_dict, "test")
        actual = eh.official_f1(words, gold, pred)
        fast = eh.fast_official_f1(words, gold, pred)

        assert actual == pytest.approx(expected, abs=1e-12)
        assert fast == pytest.approx(expected, abs=1e-12)


def test_near_dup_groups_clusters_punctuation_and_minor_variants():
    texts = [
        "En gammal kvinna gick på gatan i Stockholm år 1883.",
        "En gammal kvinna, gick på gatan i Stockholm år 1883!",     # punctuation variant (Jaccard = 1.0)
        "En gammal kvinna gick på gatan i Stockholm 1883",         # 1-token drop (Jaccard = 9/10 >= 0.80)
        "Helt annan mening om exportera varor till utlandet idag.",  # distinct
    ]
    groups = eh.near_dup_groups(texts, threshold=0.80)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_best_threshold_shrinkage_and_nondegenerate_safeguard():
    # Skewed lemma where naive argmax might predict all-0 without safeguard
    words = np.array(["skewed"] * 20 + ["balanced"] * 20)
    y_true = np.array([0] * 18 + [1, 1] + [0] * 10 + [1] * 10)
    y_prob = np.array(
        list(np.linspace(0.05, 0.40, 18)) + [0.45, 0.85] +
        list(np.linspace(0.10, 0.48, 10)) + list(np.linspace(0.55, 0.90, 10))
    )

    cal_full_shrink = eh.best_threshold(y_true, y_prob, words, shrinkage=1.0)
    assert cal_full_shrink["per_word"]["skewed"] == pytest.approx(cal_full_shrink["global_threshold"])
    assert cal_full_shrink["per_word"]["balanced"] == pytest.approx(cal_full_shrink["global_threshold"])

    cal_local = eh.best_threshold(y_true, y_prob, words, shrinkage=0.5)
    preds = eh.apply_calibrated_thresholds(words, y_prob, cal_local)
    # Safeguard ensures minority class on skewed word isn't collapsed to all-0
    assert preds[:20].sum() >= 1


def test_diagnostics_flags_unstable_minority_and_computes_threshold_loss():
    df = pd.DataFrame({
        "word": ["rare"] * 12 + ["common"] * 20,
        "label": [1] * 9 + [0] * 3 + [1] * 10 + [0] * 10,
    })
    # Rare predicts all 1 (collapsed minority class 0), probabilities actually separate them well
    preds = np.array([1] * 12 + [1] * 10 + [0] * 10)
    probs = np.array([0.8] * 9 + [0.3] * 3 + [0.9] * 10 + [0.1] * 10)
    diag = eh.diagnostics(df, preds, probs).set_index("word")

    assert bool(diag.loc["rare", "unstable_flag"]) is True
    assert diag.loc["rare", "minority_support"] == 3
    assert diag.loc["rare", "minority_recall"] == pytest.approx(0.0)
    assert diag.loc["rare", "oracle_f1"] == pytest.approx(1.0)
    assert diag.loc["rare", "threshold_loss"] > 0.3

    assert bool(diag.loc["common", "unstable_flag"]) is False
    assert diag.loc["common", "actual_f1"] == pytest.approx(1.0)


def test_paired_bootstrap_adoption_rule():
    df = pd.DataFrame({
        "word": ["w1"] * 30 + ["w2"] * 30,
        "label": ([0] * 15 + [1] * 15) * 2,
    })
    y = df["label"].to_numpy()
    preds_a = np.tile(y, (2, 1))
    # Flip 20% in system B
    preds_b = preds_a.copy()
    preds_b[:, ::5] = 1 - preds_b[:, ::5]

    res = eh.paired_bootstrap(df, preds_a, preds_b, n_boot=500, seed=42)
    assert res["point_diff"] > 0
    assert res["ci95_lo"] > 0
    assert res["adopted"] is True

    res_tie = eh.paired_bootstrap(df, preds_a, preds_a, n_boot=200, seed=42)
    assert res_tie["point_diff"] == pytest.approx(0.0)
    assert res_tie["adopted"] is False


def test_run_cv_enforces_group_isolation_and_returns_valid_oof():
    rows = []
    for i in range(40):
        w = "w1" if i < 20 else "w2"
        lbl = 0 if (i % 2 == 0) else 1
        # Pair every two items into same group
        rows.append({"word": w, "label": lbl, "group": i // 2, "x": float(lbl)})
    df = pd.DataFrame(rows)

    seen_group_overlaps = []

    class DummyThresh:
        def fit(self, tr):
            self.tr_groups = set(tr["group"])
            self.majority = int(tr["label"].mean() >= 0.5)
            return self

        def predict_proba(self, te):
            overlap = self.tr_groups & set(te["group"])
            seen_group_overlaps.append(len(overlap))
            return np.where(te["x"] > 0.5, 0.8, 0.2)

        def predict(self, te):
            return (self.predict_proba(te) >= 0.5).astype(int)

    res = eh.run_cv(df, lambda s: DummyThresh(), n_splits=5, n_repeats=2, base_seed=42, group_col="group")
    assert all(ov == 0 for ov in seen_group_overlaps)
    assert res["oof_predictions"].shape == (2, 40)
    assert res["mean_official_f1"] == pytest.approx(1.0)


def test_lolo_and_lopo_never_leak_held_out_slices():
    df = pd.DataFrame({
        "word": ["w1"] * 6 + ["w2"] * 6,
        "period_label": ["1880-1883", "2022-2023"] * 6,
        "label": [0, 1, 0, 1, 0, 1] * 2,
    })

    seen_lolo_words = []
    seen_lopo_periods = []

    class RecordingModel:
        def __init__(self, collector, col):
            self.collector = collector
            self.col = col

        def fit(self, tr):
            self.collector.append(set(tr[self.col]))
            return self

        def predict(self, te):
            return np.zeros(len(te), dtype=int)

    eh.leave_one_lemma_out(df, lambda s: RecordingModel(seen_lolo_words, "word"))
    assert seen_lolo_words == [{"w2"}, {"w1"}]

    eh.leave_one_period_out(df, lambda s: RecordingModel(seen_lopo_periods, "period_label"))
    assert seen_lopo_periods == [{"2022-2023"}, {"1880-1883"}]


def test_predict_script_handles_unlabeled_unseen_data(tmp_path):
    from scripts.predict import run_prediction

    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test_unseen"
    train_dir.mkdir()
    test_dir.mkdir()
    out_file = tmp_path / "preds.jsonl"

    # Build minimal synthetic train dir with labels
    train_usages = []
    train_labels = []
    for i in range(40):
        w = "dumpa" if i < 20 else "kvinna"
        lbl = 1 if (i % 2 == 0) else 0
        txt = f"{w} positiva ord bra {i}" if lbl == 1 else f"{w} negativa ord dålig {i}"
        sid = f"tr_{i}"
        train_usages.append({"word": w, "sentence_id": sid, "text": txt, "pos": "NN", "period_label": "2022-2023"})
        train_labels.append({"word": w, "sentence_id": sid, "label": lbl})

    (train_dir / "usages.jsonl").write_text("\n".join(json.dumps(r) for r in train_usages), encoding="utf-8")
    (train_dir / "subtask2.jsonl").write_text("\n".join(json.dumps(r) for r in train_labels), encoding="utf-8")

    # Build unseen test dir WITHOUT subtask2.jsonl (unlabeled)
    test_usages = [
        {"word": "dumpa", "sentence_id": "te_1", "text": "dumpa positiva ord bra"},
        {"word": "kvinna", "sentence_id": "te_2", "text": "kvinna negativa ord dålig"},
    ]
    (test_dir / "usages.jsonl").write_text("\n".join(json.dumps(r) for r in test_usages), encoding="utf-8")

    run_prediction(
        data_dir=test_dir,
        output_path=out_file,
        train_dir=train_dir,
        model_type="auto",
        shrinkage=0.5,
    )

    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    assert [(r["word"], r["sentence_id"]) for r in lines] == [("dumpa", "te_1"), ("kvinna", "te_2")]
    assert all(r["label"] in (0, 1) for r in lines)
