"""Nested-validation hybrid improvement; keeps completed baseline results intact."""
from __future__ import annotations

import gc
import json
import sys
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from Assignment_3.src.analysis import holm_adjust, interval, paired
from Assignment_3.src.evaluation import export_jsonl, fast_official_f1, official_module, score_predictions
from Assignment_3.src.experiments import make_split
from Assignment_3.src.improved_hybrid import (HybridFeatureBuilder, ImprovedHybridModel,
    RETRIEVAL_FEATURE_NAMES, apply_thresholds, combine_blocks, fit_thresholds, make_head)
from Assignment_3.src.runtime import Measurements, dump_json, hardware, load_data, logger, sha256

CONFIG = ROOT / "Assignment_3/configs/hybrid_improvement.json"


def select_models(config, data, output, log, measurements):
    """Finish selection for every seed before any new outer-test evaluation."""
    selections, selection_rows, margins_rows = {}, [], []
    for seed in config["seeds"]:
        train_index, test_index = make_split(data, config["protocol"], seed)
        train, test = data.iloc[train_index], data.iloc[test_index]
        oof = np.full((len(train), len(config["candidates"])), np.nan)
        folds = np.full(len(train), -1)
        memberships = []
        cv = StratifiedGroupKFold(n_splits=config["inner_folds"], shuffle=True, random_state=seed+17)
        for fold, (fit_index, val_index) in enumerate(cv.split(train, train.label_value, train.text_group)):
            fit, val = train.iloc[fit_index], train.iloc[val_index]
            if set(fit.text_group) & set(val.text_group) or set(train.sentence_id) & set(test.sentence_id):
                raise AssertionError("Nested split identity or exact-text leakage")
            folds[val_index] = fold
            memberships.extend({"fold": fold, "role": role, "sentence_id": sid, "text_group": text_hash}
                for role, part in [("fit", fit), ("validation", val)] for sid, text_hash in zip(part.sentence_id, part.text_group))
            builder = HybridFeatureBuilder(config)
            with measurements.measure("inner_features_fit", seed=seed, fold=fold, train_n=len(fit), validation_n=len(val)):
                fit_blocks = builder.fit_transform(fit)
                val_blocks = builder.transform(val)
            for candidate_id, setting in enumerate(config["candidates"]):
                head = make_head(config, setting, seed+fold)
                with measurements.measure("inner_head_fit_predict", seed=seed, fold=fold, candidate_id=candidate_id, family=setting["family"]):
                    head.fit(combine_blocks(fit_blocks, setting), fit.label_value)
                    oof[val_index, candidate_id] = head.decision_function(combine_blocks(val_blocks, setting))
            log.info("Selection only: seed=%d inner fold %d/%d complete", seed, fold+1, config["inner_folds"])
            del builder, fit_blocks, val_blocks, head
            gc.collect()
        if not np.isfinite(oof).all() or (folds < 0).any():
            raise AssertionError("Missing OOF prediction coverage")
        pd.DataFrame(memberships).to_csv(output / f"splits/inner_seed_{seed}.csv", index=False)
        data.iloc[np.r_[train_index, test_index]][["word", "sentence_id", "text_group"]].assign(
            role=["outer_train"]*len(train_index)+["outer_test"]*len(test_index)).to_csv(output / f"splits/outer_seed_{seed}.csv", index=False)
        best = {}
        for candidate_id, setting in enumerate(config["candidates"]):
            margins = oof[:, candidate_id]
            for shrinkage in config["word_threshold_shrinkage"]:
                calibration = fit_thresholds(train.word, train.label_value, margins, shrinkage, config["threshold_quantiles"])
                prediction = apply_thresholds(train.word, margins, calibration)
                score = fast_official_f1(train.word, train.label_value, prediction)
                selection_rows.append({"seed": seed, "candidate_id": candidate_id, **setting,
                    "word_threshold_shrinkage": shrinkage, "oof_selection_official_f1": score,
                    "score_scope": "training OOF model/threshold-selection score, not final test performance"})
                family = setting["family"]
                if family not in best or score > best[family]["oof_selection_official_f1"]+1e-12:
                    best[family] = {"candidate_id": candidate_id, "setting": setting, "calibration": calibration,
                                    "oof_selection_official_f1": score}
            margins_rows.extend({"seed": seed, "sentence_id": sid, "word": word, "inner_fold": int(fold),
                                 "candidate_id": candidate_id, "margin": float(margin)}
                                for sid, word, fold, margin in zip(train.sentence_id, train.word, folds, margins))
        selections[str(seed)] = best
        for family, selection in best.items():
            log.info("Selected on training OOF only: seed=%d family=%s candidate=%d score=%.4f shrinkage=%.1f",
                     seed, family, selection["candidate_id"], selection["oof_selection_official_f1"], selection["calibration"]["word_shrinkage"])
        dump_json(output / "selections_in_progress.json", selections)
        pd.DataFrame(selection_rows).to_csv(output / "tables/inner_selection.csv", index=False)
    pd.DataFrame(margins_rows).to_csv(output / "predictions/inner_oof_margins.csv", index=False)
    dump_json(output / "frozen_selections.json", {"all_seeds_selected_before_test_scoring": True,
        "config_sha256": sha256(CONFIG), "selections": selections,
        "caution": "These are optimized training-selection scores, not independent validation/test estimates."})
    return selections


def evaluate_frozen_selections(config, data, output, selections, log, measurements):
    reference = ROOT / config["reference_run"]
    rows, predictions, coefficients = [], [], []
    for seed in config["seeds"]:
        train_index, test_index = make_split(data, config["protocol"], seed)
        train, test = data.iloc[train_index], data.iloc[test_index]
        builder = HybridFeatureBuilder(config)
        with measurements.measure("outer_features_fit", seed=seed, train_n=len(train), test_n=len(test)):
            fit_blocks = builder.fit_transform(train)
            test_blocks = builder.transform(test)
        gold_path = export_jsonl(test, output / f"official/seed_{seed}/gold/SV/subtask2.jsonl", "label_value")
        for family, selection in selections[str(seed)].items():
            setting, calibration = selection["setting"], selection["calibration"]
            head = make_head(config, setting, seed)
            with measurements.measure("outer_head_fit", seed=seed, family=family, n=len(train)):
                head.fit(combine_blocks(fit_blocks, setting), train.label_value)
            with measurements.measure("outer_head_predict", seed=seed, family=family, n=len(test), frozen_encoding_included=False):
                margins = head.decision_function(combine_blocks(test_blocks, setting))
                prediction = apply_thresholds(test.word, margins, calibration)
            model = ImprovedHybridModel(builder, head, setting, calibration)
            checkpoint = output / f"checkpoints/seed_{seed}/{family}.joblib"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, checkpoint, compress=3)
            restored = joblib.load(checkpoint)
            if not np.array_equal(restored.predict(test), prediction):
                raise AssertionError(f"Checkpoint replay failed: seed={seed} family={family}")
            frame = test[["word", "sentence_id", "label_value", "text_group", "period_label"]].copy()
            frame["seed"], frame["system"], frame["prediction"], frame["margin"] = seed, family, prediction, margins
            pred_path = output / f"predictions/seed_{seed}/{family}.csv"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(pred_path, index=False)
            official_path = export_jsonl(frame, output / f"official/seed_{seed}/{family}/SV_subtask2.jsonl", "prediction")
            metrics = score_predictions(frame)
            assert np.isclose(metrics["official_f1"], official_module().evaluate_subtask2(gold_path, official_path), atol=1e-12)
            rows.append({"seed": seed, "system": family, "test_n": len(test), **metrics})
            predictions.append(frame)
            dimensions = {name: block.shape[1] for name, block in fit_blocks.items()}
            offset = 0
            for name in ["context", "retrieval", "embedding"]:
                if setting[f"{name}_weight"] > 0:
                    part = head.coef_[0, offset:offset+dimensions[name]]
                    coefficients.append({"seed": seed, "system": family, "block": name,
                        "dimension": len(part), "coefficient_l2": float(np.linalg.norm(part)),
                        "weight": setting[f"{name}_weight"]})
                    if name == "retrieval":
                        pd.DataFrame({"feature": RETRIEVAL_FEATURE_NAMES, "scaled_coefficient": part}).to_csv(
                            checkpoint.with_name(f"{family}_retrieval_coefficients.csv"), index=False)
                    offset += dimensions[name]
            dump_json(checkpoint.with_suffix(".json"), {"system": family, "seed": seed, "setting": setting,
                "calibration": calibration, "checkpoint_sha256": sha256(checkpoint), "checkpoint_replayed": True,
                "selection_sha256": sha256(output / "frozen_selections.json"), "feature_dimensions": dimensions,
                "reference_run": config["reference_run"], "reference_e5_sha256": sha256(reference / "artifacts/e5_small.npz"),
                "supervision": "Locally trained linear head plus OOF threshold calibration; frozen encoders unchanged"})
            log.info("Outer test: seed=%d %s official=%.4f pooled=%.4f", seed, family, metrics["official_f1"], metrics["pooled_macro_f1"])
            del model, restored, head
            gc.collect()
        del builder, fit_blocks, test_blocks
        gc.collect()
        # References are joined by identity rather than positional assumptions.
        for baseline in ["hybrid_train_tuned", "dense_train_tuned", "word_char_tfidf_linear_svm", "all_positive"]:
            base = pd.read_csv(reference / f"predictions/{config['protocol']}/seed_{seed}/{baseline}.csv")
            if set(base.sentence_id) != set(test.sentence_id):
                raise AssertionError("Reference and improved model test identities differ")
            base["system"] = baseline
            base["margin"] = np.nan
            predictions.append(base[frame.columns])
            rows.append({"seed": seed, "system": baseline, "test_n": len(test), **score_predictions(base)})
    scores = pd.DataFrame(rows)
    scores.to_csv(output / "tables/seed_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions/all_test_predictions.csv", index=False)
    pd.DataFrame(coefficients).to_csv(output / "tables/feature_block_coefficients.csv", index=False)
    return scores


def summarize(config, output, scores, log):
    rng = np.random.default_rng(config["statistics_seed"])
    records = []
    for system, group in scores.groupby("system"):
        for metric in ["official_f1", "pooled_macro_f1", "accuracy", "balanced_accuracy"]:
            lo, hi = interval(group[metric], rng, config["bootstrap_samples"])
            records.append({"system": system, "metric": metric, "mean": group[metric].mean(),
                "std": group[metric].std(ddof=1), "ci95_lo": lo, "ci95_hi": hi, "n_seeds": len(group)})
    summary = pd.DataFrame(records)
    summary.to_csv(output / "tables/summary.csv", index=False)
    comparisons = []
    for reference in ["hybrid_train_tuned", "dense_train_tuned", "word_char_tfidf_linear_svm", "context_only_control", "retrieval_only_control"]:
        for metric in ["official_f1", "pooled_macro_f1"]:
            pivot = scores.pivot(index="seed", columns="system", values=metric).sort_index()
            comparisons.append({"system_a": "improved_hybrid", "system_b": reference, "metric": metric,
                **paired(pivot.improved_hybrid, pivot[reference], rng, config["bootstrap_samples"])})
    comparisons = pd.DataFrame(comparisons)
    comparisons["holm_p"] = holm_adjust(comparisons.exact_p)
    comparisons.to_csv(output / "tables/paired_comparisons.csv", index=False)
    pred = pd.read_csv(output / "predictions/all_test_predictions.csv")
    perword, errors = [], []
    for (seed, system), group in pred.groupby(["seed", "system"]):
        errors.append({"seed": seed, "system": system, "fp": int(((group.label_value == 0)&(group.prediction == 1)).sum()),
                       "fn": int(((group.label_value == 1)&(group.prediction == 0)).sum()),
                       "error_rate": float((group.label_value != group.prediction).mean())})
        for word, part in group.groupby("word"):
            perword.append({"seed": seed, "system": system, "word": word, "n": len(part), **score_predictions(part)})
    pd.DataFrame(perword).to_csv(output / "tables/per_word_metrics.csv", index=False)
    pd.DataFrame(errors).to_csv(output / "tables/error_counts.csv", index=False)
    official = summary[summary.metric == "official_f1"].set_index("system")
    new, old = official.loc["improved_hybrid"], official.loc["hybrid_train_tuned"]
    context_delta = comparisons[(comparisons.system_b == "context_only_control")&(comparisons.metric == "official_f1")].iloc[0]
    dump_json(output / "improvement_summary.json", {"official_metric": "Pinned official mean of within-word two-class macro-F1",
        "original_hybrid": {"mean": old["mean"], "std": old["std"]},
        "improved_hybrid": {"mean": new["mean"], "std": new["std"]},
        "absolute_change": float(new["mean"]-old["mean"]),
        "retrieval_contribution_against_matched_context_control": {"mean_difference": context_delta.mean_diff, "holm_p": context_delta.holm_p},
        "selection_frozen_before_test": True, "new_supervised_head": True,
        "evidence_scope": "Five paired, overlapping, previously inspected Swedish development splits; not hidden-test superiority.",
        "statistical_limit": "With five nonzero pairs, minimum two-sided exact sign-flip p is 0.0625. Split-bootstrap intervals are descriptive.",
        "limitations": config["notes"]})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    part = official.sort_values("mean")
    ax.barh(part.index, part["mean"], xerr=np.maximum(0, np.vstack([part["mean"]-part.ci95_lo, part.ci95_hi-part["mean"]])), capsize=4)
    ax.set_xlabel("Official F1: five-split mean and descriptive 95% split-bootstrap interval")
    fig.tight_layout()
    fig.savefig(output / "figures/model_comparison.png", dpi=220)
    fig.savefig(output / "figures/model_comparison.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    log.info("Final paired development comparison: original hybrid %.4f +/- %.4f; improved %.4f +/- %.4f",
             old["mean"], old["std"], new["mean"], new["std"])


def package(config, output, log):
    base_zip = ROOT / "Assignment_3/reproducibility_package.zip"
    base_metadata = json.loads((base_zip.with_suffix(".json")).read_text(encoding="utf-8"))
    if sha256(base_zip) != base_metadata["sha256"]:
        raise AssertionError("Baseline archive changed")
    new_files = [CONFIG, ROOT / "Assignment_3/src/improved_hybrid.py", Path(__file__).resolve(),
                 ROOT / "Assignment_3/scripts/infer_improved_hybrid.py", ROOT / "Assignment_3/tests/test_improved_hybrid.py",
                 ROOT / "Assignment_3/RUNNING_IMPROVEMENT.md",
                 *[p for p in output.rglob("*") if p.is_file() and p.name not in {"artifact_manifest.json", "execution.log"}]]
    new_files = sorted(set(p for p in new_files if p.is_file()))
    manifest = {"config_sha256": sha256(CONFIG), "base_archive_sha256": base_metadata["sha256"],
        "new_files_sha256": {str(p.relative_to(ROOT)).replace('\\','/'): sha256(p) for p in new_files},
        "all_checkpoints_replayed": True, "selection_frozen_before_any_outer_test_score": True,
        "training_scope": "outer-train only, grouped inner OOF selection", "reports_generated": False}
    dump_json(output / "artifact_manifest.json", manifest)
    destination = ROOT / "Assignment_3/improved_hybrid_package.zip"
    with zipfile.ZipFile(base_zip) as source, zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        extra_names = {str(p.relative_to(ROOT)).replace('\\','/') for p in new_files}
        for item in source.infolist():
            if item.filename not in extra_names:
                archive.writestr(item, source.read(item.filename))
        for path in new_files+[output / "artifact_manifest.json"]:
            archive.write(path, str(path.relative_to(ROOT)).replace('\\','/'))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
    dump_json(destination.with_suffix(".json"), {"bytes": destination.stat().st_size, "sha256": sha256(destination), "baseline_reference_preserved": True})
    log.info("Verified upgraded reproducibility package: %.1f MB", destination.stat().st_size/2**20)


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    for name in ["tables", "predictions", "splits", "checkpoints", "official", "figures"]:
        (output/name).mkdir(parents=True, exist_ok=True)
    log = logger(output)
    measurements = Measurements(output)
    dump_json(output / "config.json", config)
    dump_json(output / "hardware.json", hardware())
    data, _ = load_data()
    start = time.perf_counter()
    selections = select_models(config, data, output, log, measurements)
    scores = evaluate_frozen_selections(config, data, output, selections, log, measurements)
    summarize(config, output, scores, log)
    pd.read_json(output / "resource_measurements.jsonl", lines=True).to_csv(output / "tables/measured_resources.csv", index=False)
    dump_json(output / "execution_summary.json", {"wall_seconds": time.perf_counter()-start, "refitted_frozen_encoders": False,
        "new_fitted_models": len(config["seeds"])*3, "reports_generated": False})
    package(config, output, log)


if __name__ == "__main__":
    main()
