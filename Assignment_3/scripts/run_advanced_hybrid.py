"""Bounded, training-only search for stronger word-aware hybrid models."""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from Assignment_3.src.advanced_hybrid import AdvancedBuilder, AdvancedHead, AdvancedHybridModel
from Assignment_3.src.analysis import holm_adjust, interval, paired
from Assignment_3.src.evaluation import export_jsonl, fast_official_f1, official_module, score_predictions
from Assignment_3.src.experiments import make_split
from Assignment_3.src.improved_hybrid import apply_thresholds, fit_thresholds
from Assignment_3.src.runtime import Measurements, dump_json, hardware, load_data, logger, sha256

CONFIG = ROOT / "Assignment_3/configs/advanced_hybrid.json"


def select(config, output, data, log, measurements):
    selections, rows, oof_rows = {}, [], []
    for seed in config["seeds"]:
        train_id, test_id = make_split(data, config["protocol"], seed)
        train = data.iloc[train_id]
        cv = StratifiedGroupKFold(n_splits=config["inner_folds"], shuffle=True, random_state=seed+17)
        margins = {c["name"]: np.full(len(train), np.nan) for c in config["candidates"]}
        folds = np.full(len(train), -1)
        split_rows = []
        for fold, (fit_id, val_id) in enumerate(cv.split(train, train.label_value, train.text_group)):
            fit, val = train.iloc[fit_id], train.iloc[val_id]
            if set(fit.text_group) & set(val.text_group):
                raise AssertionError("Inner duplicate overlap")
            folds[val_id] = fold
            split_rows.extend({"fold": fold, "role": role, "sentence_id": sid, "text_group": group}
                              for role,frame in [("fit",fit),("validation",val)] for sid,group in zip(frame.sentence_id, frame.text_group))
            builder = AdvancedBuilder(config)
            with measurements.measure("inner_advanced_features", seed=seed, fold=fold):
                fit_views = builder.fit_transform(fit, seed+fold)
                val_views = builder.transform(val)
            for setting in config["candidates"]:
                with measurements.measure("inner_advanced_head", seed=seed, fold=fold, candidate=setting["name"]):
                    head = AdvancedHead(setting, seed+fold).fit(fit_views, fit)
                    margins[setting["name"]][val_id] = head.decision_function(val_views, val)
                del head
            log.info("Training-only selection seed=%d fold=%d/%d finished", seed, fold+1, config["inner_folds"])
            del builder, fit_views, val_views
            gc.collect()
        if any(not np.isfinite(values).all() for values in margins.values()):
            raise AssertionError("Incomplete OOF predictions")
        pd.DataFrame(split_rows).to_csv(output / f"splits/inner_seed_{seed}.csv", index=False)
        pd.concat([data.iloc[train_id].assign(role="outer_train"), data.iloc[test_id].assign(role="outer_test")])[
            ["word", "sentence_id", "text_group", "role"]].to_csv(output / f"splits/outer_seed_{seed}.csv", index=False)
        scales = {name: float(max(values.std(), 1e-8)) for name,values in margins.items()}
        recipes = [{"name": name, "components": [name], "scales": [1.]} for name in margins]
        for i, components in enumerate(config["ensemble_components"]):
            recipes.append({"name": f"ensemble_{i}", "components": components,
                            "scales": [scales[name] for name in components]})
        best = None
        for recipe in recipes:
            values = np.mean([margins[name]/scale for name,scale in zip(recipe["components"], recipe["scales"])], axis=0)
            for shrink in config["word_threshold_shrinkage"]:
                calibration = fit_thresholds(train.word, train.label_value, values, shrink, config["threshold_quantiles"])
                score = fast_official_f1(train.word, train.label_value, apply_thresholds(train.word, values, calibration))
                row = {"seed": seed, "candidate": recipe["name"], "shrinkage": shrink, "oof_selection_f1": score}
                rows.append(row)
                if best is None or score > best["oof_selection_f1"]+1e-12:
                    best = {**row, "recipe": recipe, "calibration": calibration}
        selections[str(seed)] = best
        for name, values in margins.items():
            oof_rows.extend({"seed": seed, "sentence_id": sid, "word": word, "fold": int(fold),
                             "candidate": name, "margin": float(value)}
                            for sid,word,fold,value in zip(train.sentence_id, train.word, folds, values))
        pd.DataFrame(rows).to_csv(output / "tables/inner_selection.csv", index=False)
        log.info("FROZEN training selection seed=%d candidate=%s OOF selection F1=%.4f", seed, best["candidate"], best["oof_selection_f1"])
    pd.DataFrame(oof_rows).to_csv(output / "predictions/inner_oof_margins.csv", index=False)
    dump_json(output / "frozen_selections.json", {"all_seeds_selected_before_outer_test_scoring": True,
        "config_sha256": sha256(CONFIG), "selections": selections,
        "warning": "OOF selection scores include hyperparameter/threshold selection and are not final performance estimates."})
    return selections


def evaluate(config, output, data, selections, log, measurements):
    records, predictions = [], []
    settings = {c["name"]:c for c in config["candidates"]}
    for seed in config["seeds"]:
        train_id, test_id = make_split(data, config["protocol"], seed)
        train, test = data.iloc[train_id], data.iloc[test_id]
        selected = selections[str(seed)]
        builder = AdvancedBuilder(config)
        with measurements.measure("outer_advanced_features", seed=seed):
            fit_views = builder.fit_transform(train, seed)
            test_views = builder.transform(test)
        heads = []
        for name in selected["recipe"]["components"]:
            with measurements.measure("outer_advanced_head", seed=seed, candidate=name):
                heads.append(AdvancedHead(settings[name], seed).fit(fit_views, train))
        margins = np.mean([head.decision_function(test_views, test)/scale for head,scale in zip(heads, selected["recipe"]["scales"])], axis=0)
        prediction = apply_thresholds(test.word, margins, selected["calibration"])
        model = AdvancedHybridModel(builder, heads, selected["recipe"]["scales"], selected["calibration"])
        path = output / f"checkpoints/seed_{seed}.joblib"
        joblib.dump(model, path, compress=3)
        restored = joblib.load(path)
        if not np.array_equal(restored.predict(test), prediction):
            raise AssertionError("Advanced checkpoint replay failed")
        dump_json(path.with_suffix(".json"), {"checkpoint_sha256": sha256(path), "checkpoint_replayed": True,
            "selected": selected, "selection_sha256": sha256(output / "frozen_selections.json")})
        frame = test[["word", "sentence_id", "label_value", "text_group", "period_label"]].copy()
        frame["seed"], frame["system"], frame["prediction"], frame["margin"] = seed, "advanced_hybrid", prediction, margins
        frame.to_csv(output / f"predictions/seed_{seed}.csv", index=False)
        gold = export_jsonl(frame, output / f"official/seed_{seed}/gold/SV/subtask2.jsonl", "label_value")
        pred = export_jsonl(frame, output / f"official/seed_{seed}/SV_subtask2.jsonl", "prediction")
        metrics = score_predictions(frame)
        assert np.isclose(metrics["official_f1"], official_module().evaluate_subtask2(gold, pred), atol=1e-12)
        records.append({"seed": seed, "system": "advanced_hybrid", **metrics})
        predictions.append(frame)
        log.info("Held-out development seed=%d official=%.4f accuracy=%.4f", seed, metrics["official_f1"], metrics["accuracy"])
        del builder, fit_views, test_views, heads, model, restored
        gc.collect()
    reference = pd.read_csv(ROOT / config["reference_run"] / "tables/seed_metrics.csv")
    reference = reference[(reference.protocol == config["protocol"]) & reference.system.isin(["word_char_tfidf_linear_svm", "hybrid_train_tuned"])]
    for _,row in reference.iterrows():
        records.append({key:row[key] for key in ["seed", "system", "official_f1", "pooled_macro_f1", "accuracy", "balanced_accuracy"]})
    scores = pd.DataFrame(records)
    scores.to_csv(output / "tables/seed_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions/all_predictions.csv", index=False)
    return scores


def summarize(config, output, scores, log):
    rng = np.random.default_rng(config["statistics_seed"])
    rows, comparisons = [], []
    for system, group in scores.groupby("system"):
        for metric in ["official_f1", "pooled_macro_f1", "accuracy", "balanced_accuracy"]:
            lo,hi = interval(group[metric], rng, config["bootstrap_samples"])
            rows.append({"system": system, "metric": metric, "mean": group[metric].mean(), "std": group[metric].std(ddof=1), "ci95_lo":lo, "ci95_hi":hi})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "tables/summary.csv", index=False)
    for baseline in ["word_char_tfidf_linear_svm", "hybrid_train_tuned"]:
        p = scores.pivot(index="seed", columns="system", values="official_f1").sort_index()
        comparisons.append({"system_a":"advanced_hybrid", "system_b":baseline, **paired(p.advanced_hybrid, p[baseline], rng, config["bootstrap_samples"])})
    comparisons = pd.DataFrame(comparisons)
    comparisons["holm_p"] = holm_adjust(comparisons.exact_p)
    comparisons.to_csv(output / "tables/paired_tests.csv", index=False)
    official = summary[summary.metric == "official_f1"].set_index("system")
    current = official.loc["advanced_hybrid"]
    dump_json(output / "result_summary.json", {"advanced_official_f1_mean":current["mean"], "advanced_official_f1_std":current["std"],
        "target_official_f1":0.8, "target_reached":bool(current["mean"]>=.8), "metric_unchanged":True,
        "claims": "Five reused, overlapping development splits; no hidden-test or statistically confirmed superiority claim.", "protocol_notes":config["notes"]})
    log.info("Advanced official F1 %.4f +/- %.4f; target 0.8 reached=%s", current["mean"], current["std"], current["mean"]>=.8)


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    for folder in ["tables", "splits", "predictions", "checkpoints", "official"]:
        (output/folder).mkdir(parents=True, exist_ok=True)
    log = logger(output)
    measurements = Measurements(output)
    dump_json(output / "config.json", config)
    dump_json(output / "hardware.json", hardware())
    data,_ = load_data()
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        selections = select(config, output, data, log, measurements)
        scores = evaluate(config, output, data, selections, log, measurements)
    summarize(config, output, scores, log)
    pd.read_json(output / "resource_measurements.jsonl", lines=True).to_csv(output / "tables/resources.csv", index=False)
    dump_json(output / "execution_summary.json", {"wall_seconds":time.perf_counter()-start, "fitted_checkpoints":5, "reports_generated":False})
    files = [CONFIG, Path(__file__).resolve(), ROOT/"Assignment_3/src/advanced_hybrid.py",
             *[p for p in output.rglob("*") if p.is_file() and p.name not in {"artifact_manifest.json", "execution.log"}]]
    dump_json(output / "artifact_manifest.json", {"files_sha256":{str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in files},
        "official_metric_unchanged":True, "all_seeds_selected_before_test_scoring":True, "checkpoints_replayed":True})
    from Assignment_3.scripts.finalize_results import main as publish_final
    publish_final()


if __name__ == "__main__":
    main()
