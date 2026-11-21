"""Publish only the selected model plus required baseline/ablation evidence."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from Assignment_3.src.analysis import holm_adjust, interval, paired
from Assignment_3.src.evaluation import official_module, score_predictions
from Assignment_3.src.runtime import dump_json, load_data, logger, sha256
from Assignment_3.src.deliverables import current_deliverable

BASE = ROOT / "Assignment_3/runs/completed"
MODEL = ROOT / "Assignment_3/runs/advanced_hybrid"
RESULTS = ROOT / "Assignment_3/results"
METRICS = ["official_f1", "pooled_macro_f1", "accuracy", "balanced_accuracy"]
# Categories are defined before aggregation. They describe observable decisions,
# not inferred linguistic causes. Comparisons use the matched lexical baseline.
ERROR_CATEGORIES = ["FN model-only miss", "FN shared miss", "FP model-only match", "FP shared match"]


def summarize_final(replay=True):
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "Assignment_3/configs/advanced_hybrid.json").read_text(encoding="utf-8"))
    saved_config = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    if "previous_improvement" in saved_config:
        saved_config.pop("previous_improvement")
        if saved_config != config:
            raise ValueError("Cleanup must not change the model-selection configuration")
        dump_json(MODEL / "config.json", config)
        frozen = json.loads((MODEL / "frozen_selections.json").read_text(encoding="utf-8"))
        frozen["training_config_sha256"] = frozen["config_sha256"]
        frozen["config_sha256"] = sha256(ROOT / "Assignment_3/configs/advanced_hybrid.json")
        frozen["post_training_config_cleanup"] = "Removed only an unused discarded-comparison location; model selection and fitted parameters are unchanged. Original training config digest is retained."
        dump_json(MODEL / "frozen_selections.json", frozen)
        for checkpoint in (MODEL / "checkpoints").glob("seed_*.json"):
            metadata = json.loads(checkpoint.read_text(encoding="utf-8"))
            metadata["selection_sha256"] = sha256(MODEL / "frozen_selections.json")
            dump_json(checkpoint, metadata)
    data, definitions = load_data()
    frames, records = [], []
    for seed in config["seeds"]:
        print(f"Checking selected checkpoint and official predictions: seed {seed}", flush=True)
        frame = pd.read_csv(MODEL / f"predictions/seed_{seed}.csv")
        if replay:
            checkpoint = MODEL / f"checkpoints/seed_{seed}.joblib"
            meta = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
            if sha256(checkpoint) != meta["checkpoint_sha256"]:
                raise ValueError("Final checkpoint integrity failed")
            inputs = frame[["word", "sentence_id"]].merge(data, on=["word", "sentence_id"], validate="one_to_one")
            model = joblib.load(checkpoint)
            if not np.array_equal(model.predict(inputs), frame.prediction):
                raise AssertionError("Final checkpoint predictions changed during cleanup")
            del model
            gc.collect()
        metrics = score_predictions(frame)
        official = official_module().evaluate_subtask2(
            MODEL / f"official/seed_{seed}/gold/SV/subtask2.jsonl",
            MODEL / f"official/seed_{seed}/SV_subtask2.jsonl")
        if not np.isclose(metrics["official_f1"], official, atol=1e-12):
            raise AssertionError("Official file score differs")
        records.append({"seed": seed, "system": "advanced_hybrid", **metrics})
        frames.append(frame)
        print(f"Verified seed {seed}: official F1 {metrics['official_f1']:.6f}", flush=True)
    final_scores = pd.DataFrame(records)
    baselines = pd.read_csv(BASE / "tables/seed_metrics.csv")
    baselines = baselines[baselines.protocol == config["protocol"]]
    scores = pd.concat([final_scores, baselines[["seed", "system", *METRICS]]], ignore_index=True)
    scores["role"] = np.where(scores.system == "advanced_hybrid", "selected_model", "baseline_or_ablation")
    scores.to_csv(RESULTS / "seed_metrics.csv", index=False)
    rng = np.random.default_rng(config["statistics_seed"])
    summaries = []
    for (system, role), group in scores.groupby(["system", "role"]):
        for metric in METRICS:
            lo, hi = interval(group[metric], rng, config["bootstrap_samples"])
            summaries.append({"system": system, "role": role, "metric": metric, "n_seeds": len(group),
                "mean": group[metric].mean(), "std": group[metric].std(ddof=1), "ci95_lo": lo, "ci95_hi": hi})
    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    table = summary[summary.metric == "official_f1"].sort_values(["role", "mean"], ascending=[False, False])
    table.to_csv(RESULTS / "model_scores.csv", index=False)
    comparisons = []
    pivot = scores.pivot(index="seed", columns="system", values="official_f1").sort_index()
    for baseline in baselines.system.unique():
        comparisons.append({"system_a": "advanced_hybrid", "system_b": baseline, "comparison_type": "direct",
            **paired(pivot.advanced_hybrid, pivot[baseline], rng, config["bootstrap_samples"])})
    pairs = pd.DataFrame(comparisons)
    pairs["holm_p"] = holm_adjust(pairs.exact_p)
    pairs.to_csv(RESULTS / "paired_tests.csv", index=False)
    # Remove discarded trial rows from the selected model's own tables too.
    from Assignment_3.scripts.run_advanced_hybrid import summarize
    retained = scores[scores.system.isin(["advanced_hybrid", "word_char_tfidf_linear_svm", "hybrid_train_tuned"])]
    retained.to_csv(MODEL / "tables/seed_metrics.csv", index=False)
    summarize(config, MODEL, retained, logger(MODEL))
    final_pred = pd.concat(frames, ignore_index=True)
    reference_pred = pd.read_csv(BASE / "predictions/all_predictions.csv")
    reference_pred = reference_pred[(reference_pred.protocol == config["protocol"]) &
                                    (reference_pred.system == "word_char_tfidf_linear_svm")]
    joined = final_pred.merge(reference_pred[["seed", "word", "sentence_id", "prediction"]].rename(
        columns={"prediction": "baseline_prediction"}), on=["seed", "word", "sentence_id"], validate="one_to_one")
    y, p, b = joined.label_value, joined.prediction, joined.baseline_prediction
    joined["error_category"] = np.select([(y==1)&(p==0)&(b==1), (y==1)&(p==0)&(b==0),
        (y==0)&(p==1)&(b==0), (y==0)&(p==1)&(b==1)], ERROR_CATEGORIES, default="correct")
    joined.to_csv(RESULTS / "annotated_predictions.csv", index=False)
    counts = pd.crosstab(joined.seed, joined.error_category).reindex(columns=[*ERROR_CATEGORIES, "correct"], fill_value=0)
    counts.to_csv(RESULTS / "error_counts_by_seed.csv")
    counts.agg(["mean", "std"]).T.to_csv(RESULTS / "error_summary.csv")
    per_word = [{"seed": seed, "word": word, "n": len(g), "errors": int((g.label_value != g.prediction).sum()),
                 **score_predictions(g)} for (seed, word), g in joined.groupby(["seed", "word"])]
    pd.DataFrame(per_word).to_csv(RESULTS / "per_word_metrics.csv", index=False)
    examples = joined[joined.error_category != "correct"].sort_values(["seed", "word", "sentence_id"]).groupby("error_category").head(2)
    examples = examples.merge(data[["word", "sentence_id", "text"]], on=["word", "sentence_id"], validate="many_to_one")
    examples = examples.merge(definitions[["word", "definition"]], on="word", validate="many_to_one")
    explanations = {"FN model-only miss": "Target sense was missed; lexical baseline recovered it.",
        "FN shared miss": "Both systems missed the target sense.", "FP model-only match": "Model assigned the target sense; lexical baseline rejected it.",
        "FP shared match": "Both systems incorrectly assigned the target sense."}
    examples["explanation"] = examples.error_category.map(explanations)
    examples.to_csv(RESULTS / "representative_errors.csv", index=False)
    findings = {"selected_system": "advanced_hybrid", "checkpoint": str((MODEL / "checkpoints/seed_42.joblib").relative_to(ROOT)).replace("\\", "/"),
        "protocol": config["protocol"], "seeds": config["seeds"], "official_f1_mean": float(final_scores.official_f1.mean()),
        "official_f1_std": float(final_scores.official_f1.std(ddof=1)), "pooled_macro_f1_mean": float(final_scores.pooled_macro_f1.mean()),
        "accuracy_mean": float(final_scores.accuracy.mean()), "target_official_f1": .8,
        "target_reached": bool(final_scores.official_f1.mean() >= .8), "error_categories": ERROR_CATEGORIES,
        "error_count_unit": "Test appearances across overlapping splits, not independent usages.",
        "baseline_and_ablation_evidence": "Assignment_3/runs/completed",
        "limitations": ["Development splits were inspected during earlier model development; this is not an untouched hidden test.",
            "Five overlapping splits describe split sensitivity; paired tests do not prove superiority over the lexical SVM.",
            "Exact duplicates are grouped, while source and near-duplicate leakage are not ruled out.",
            "Rare word-class combinations, domain metadata priors and unseen-lemma generalization remain limitations."],
        "next_steps": ["Evaluate the frozen system on untouched task test data.", "Audit minority-sense errors and collect additional independently labeled examples."],
        "reports_generated": False}
    dump_json(RESULTS / "findings.json", findings)
    published = json.loads((ROOT / "Assignment_3/configs/published_comparisons.json").read_text(encoding="utf-8"))
    dump_json(RESULTS / "published_comparisons.json", {"scope": "Methodological comparisons only; published work does not report this fitted final model on these splits.", "comparisons": published})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = summary[(summary.metric == "official_f1") & summary.system.isin(
        ["advanced_hybrid", "word_char_tfidf_linear_svm", "frozen_xlmr_logistic_regression", "bm25_train_tuned", "dense_train_tuned", "hybrid_train_tuned"])].sort_values("mean")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(selected.system, selected["mean"], xerr=selected["std"], capsize=3)
    ax.set_xlabel("Official F1: five-split mean +/- sample standard deviation")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(RESULTS / f"model_comparison.{extension}", dpi=180)
    plt.close(fig)
    return findings


def refresh_manifests():
    from Assignment_3.src.package import finalize
    config = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
    finalize(config, BASE, logger(BASE))
    # Core fitted checkpoints and raw run predictions are unchanged. The source
    # and output manifest is refreshed after removing obsolete trial dependencies.
    files = [ROOT / "Assignment_3/configs/advanced_hybrid.json", ROOT / "Assignment_3/scripts/run_advanced_hybrid.py",
             ROOT / "Assignment_3/src/advanced_hybrid.py", *[p for p in MODEL.rglob("*")
             if p.is_file() and p.name not in {"artifact_manifest.json", "execution.log"}]]
    dump_json(MODEL / "artifact_manifest.json", {"files_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in files},
        "official_metric_unchanged": True, "all_seeds_selected_before_test_scoring": True,
        "checkpoints_replayed": True, "cleanup_scope": "Comparison/reporting dependencies only; fitted parameters and held-out predictions unchanged."})


def package_final():
    roots = [ROOT / "Assignment_3/src", ROOT / "Assignment_3/scripts", ROOT / "Assignment_3/configs",
        ROOT / "Assignment_3/tests", ROOT / "Assignment_3/vendor/official", BASE, MODEL, RESULTS,
        ROOT / "src", ROOT / "tests", ROOT / "configs", ROOT / "Assignment_2/src",
        ROOT / "Assignment_2/tests", ROOT / "Assignment_2/configs", ROOT / "data/raw/dev/SV"]
    files = [p for directory in roots for p in directory.rglob("*") if p.is_file() and
        "__pycache__" not in p.parts and p.name not in {"execution.log", "artifact_manifest.json"}]
    files += [BASE / "artifact_manifest.json", MODEL / "artifact_manifest.json", ROOT / "README.md",
              ROOT / "Assignment_3/README.md", ROOT / "Assignment_3/RUNNING.md", ROOT / "requirements.txt",
              ROOT / "Assignment_3/requirements.txt", ROOT / "Assignment_2/evaluate.py"]
    files = sorted(set(p for p in files if current_deliverable(p)))
    manifest = RESULTS / "artifact_manifest.json"
    dump_json(manifest, {"selected_system": "advanced_hybrid", "fitted_final_checkpoints": 5,
        "baseline_and_ablation_classifier_checkpoints": 100, "discarded_trials_included": False,
        "pretrained_weights_included": False, "pretrained_weights": "Exact revisions and model hashes are in runs/completed/artifacts/*.json and configs/complete_experiments.json.",
        "files_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in files}})
    archive = ROOT / "Assignment_3/final_package.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as package:
        for path in files+[manifest]:
            package.write(path, str(path.relative_to(ROOT)).replace("\\", "/"))
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None:
            raise AssertionError("Final ZIP failed its integrity check")
    dump_json(archive.with_suffix(".json"), {"archive": archive.name, "bytes": archive.stat().st_size,
        "sha256": sha256(archive), "selected_model": "advanced_hybrid", "discarded_trials_included": False})
    print(f"Final package verified: {archive.stat().st_size / 2**20:.1f} MiB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-replay", action="store_true", help="Only for repackaging after a verified replay")
    args = parser.parse_args()
    findings = summarize_final(replay=not args.skip_replay)
    refresh_manifests()
    package_final()
    print(f"Selected model official F1: {findings['official_f1_mean']:.6f} +/- {findings['official_f1_std']:.6f}")


if __name__ == "__main__":
    main()
