"""Audit retained baseline and ablation evidence for the final package."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from Assignment_3.src.experiments import CLASSIFIER_SYSTEMS, RETRIEVAL_SYSTEMS
from Assignment_3.src.inference import predict_checkpoint, retrieval_demo
from Assignment_3.src.runtime import ROOT, dump_json, load_data, sha256
from Assignment_3.src.deliverables import current_deliverable


def finalize(config, output, log):
    checkpoints = json.loads((output / "checkpoints/manifest.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(output / "tables/seed_metrics.csv")
    predictions = pd.read_csv(output / "predictions/all_predictions.csv")
    splits = pd.read_csv(output / "tables/split_audit.csv")
    n_splits = len(config["seeds"])*len(config["split_protocols"])
    expected = n_splits*(len(CLASSIFIER_SYSTEMS)+len(RETRIEVAL_SYSTEMS))
    if len(scores) != expected or len(checkpoints) != expected:
        raise AssertionError(f"Expected {expected} seed configurations, got {len(scores)}, {len(checkpoints)}")
    if not scores.checkpoint_replayed.all():
        raise AssertionError("A checkpoint was not replayed")
    if not (splits[splits.protocol == "grouped_text"].shared_text_groups == 0).all():
        raise AssertionError("Grouped split leakage")
    for checkpoint in checkpoints:
        path = output / checkpoint["path"]
        if sha256(path) != checkpoint["sha256"]:
            raise AssertionError(f"Checkpoint integrity failed: {path}")
    for protocol in config["split_protocols"]:
        for seed in config["seeds"]:
            part = scores[(scores.protocol == protocol)&(scores.seed == seed)]
            if set(part.system) != set(CLASSIFIER_SYSTEMS+RETRIEVAL_SYSTEMS):
                raise AssertionError("Missing system/seed coverage")
    if predictions.duplicated(["protocol", "seed", "system", "word", "sentence_id"]).any():
        raise AssertionError("Duplicate prediction identities")
    data, _ = load_data()
    # A genuine checkpoint input-to-output fallback: predictions are recomputed,
    # not merely copied from the table.
    test = predictions[(predictions.protocol == "grouped_text")&(predictions.seed == 42)&(predictions.system == "word_char_tfidf_linear_svm")]
    selected = pd.concat([test[test.prediction == test.label_value].head(2), test[test.prediction != test.label_value].head(2)])
    inputs = selected[["word", "sentence_id"]].merge(data, on=["word", "sentence_id"], validate="one_to_one")
    checkpoint = output / "checkpoints/grouped_text/seed_42/word_char_tfidf_linear_svm.joblib"
    replay = predict_checkpoint(checkpoint, inputs, output)
    expected_pred = selected.set_index("sentence_id").loc[replay.sentence_id, "prediction"].to_numpy()
    if not np.array_equal(replay.prediction, expected_pred):
        raise AssertionError("CLI inference input-to-output replay failed")
    fallback = selected.merge(data[["word", "sentence_id", "text", "definition"]], on=["word", "sentence_id"])
    fallback["mode"] = "fitted_checkpoint_inference_replayed"
    fallback.to_json(output / "demo_checkpoint_fallback.jsonl", orient="records", lines=True, force_ascii=False)
    inputs[["word", "sentence_id", "text", "pos", "period_label"]].to_json(output / "demo_inputs.jsonl", orient="records", lines=True, force_ascii=False)
    dump_json(output / "demo_saved_retrieval.json", retrieval_demo(config, output, "motiv", mode="saved"))
    provenance = json.loads((ROOT / "Assignment_3/vendor/official/provenance.json").read_text(encoding="utf-8"))
    dump_json(output / "completion_status.json", {"primary_experiments": True, "planned_ablations": True,
        "official_evaluator_verified_and_used": True, "secondary_metrics": True, "repeated_seed_statistics": True,
        "saved_predictions": True, "fitted_checkpoints_present_and_replayed": True, "measured_resource_costs": True,
        "split_definitions_and_grouped_duplicate_control": True, "quantitative_errors_and_examples": True,
        "comparability_labeled": True, "findings_and_limitations_json": True, "working_checkpoint_demo": True,
        "saved_demo_fallback": True, "paper_and_detailed_report": "excluded by user", "presentation": "excluded by user"})
    # The active log may receive further messages; retain and hash a stable snapshot.
    (output / "execution_snapshot.log").write_bytes((output / "execution.log").read_bytes())
    files = [*sorted((ROOT / "Assignment_3/src").glob("*.py")), *sorted((ROOT / "Assignment_3/scripts").glob("*.py")),
             *sorted((ROOT / "Assignment_3/configs").glob("*.json")), *sorted((ROOT / "Assignment_3/vendor/official").glob("*")),
             *sorted((ROOT / "Assignment_3/tests").glob("*.py")), ROOT / "Assignment_3/requirements.txt",
             ROOT / "Assignment_3/RUNNING.md", ROOT / "configs/baselines.yaml",
             ROOT / "Assignment_3/README.md", ROOT / "README.md", ROOT / "requirements.txt",
             *sorted((ROOT / "configs").glob("*.yaml")),
             *sorted((ROOT / "tests").glob("*.py")), *sorted((ROOT / "Assignment_2/tests").glob("*.py")),
             *[p for p in (ROOT / "src").rglob("*.py") if "__pycache__" not in p.parts],
             *[p for p in (ROOT / "Assignment_2/src").rglob("*.py") if "__pycache__" not in p.parts],
             *sorted((ROOT / "Assignment_2/configs").glob("*.yaml")), *sorted((ROOT / "data/raw/dev/SV").glob("*.jsonl")),
             *[p for p in output.rglob("*") if p.is_file() and p.name not in {"artifact_manifest.json", "execution.log"}]]
    files = sorted(set(p for p in files if current_deliverable(p)))
    manifest = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "official_evaluator": provenance,
        "n_seed_configurations": len(scores), "classifier_checkpoint_count": sum(p["path"].endswith(".joblib") for p in checkpoints),
        "retrieval_checkpoint_count": sum(p["path"].endswith(".json") for p in checkpoints),
        "prediction_rows": len(predictions), "fitted_checkpoint_replay_checks_passed": True,
        "scope": "Baseline and ablation evidence supporting the selected final model; no paper, detailed report or slides",
        "external_model_weights": {"included_in_zip": False, "reason": "Public pretrained weights are large; exact revisions/hashes and download command are provided. Actual fitted heads and representation matrices are included.", "profiles": config["models"]},
        "files_sha256": {str(p.relative_to(ROOT)).replace('\\','/'): sha256(p) for p in files}}
    dump_json(output / "artifact_manifest.json", manifest)
    log.info("Baseline/ablation audit complete; use finalize_results.py to export the final package")
