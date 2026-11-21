"""Complete primary experiments, planned ablations and checkpoint replay validation."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from Assignment_2.src.data.data_loader import Document
from Assignment_2.src.evaluation.evaluator import evaluate_retrieval_run
from Assignment_2.src.retrieval.base import Hit, sort_and_rank
from Assignment_2.src.retrieval.bm25_retriever import BM25Retriever
from Assignment_2.src.retrieval.fusion import reciprocal_rank_fusion
from Assignment_3.scripts.run_part1_experiments import build_classical_models
from Assignment_3.src.evaluation import export_jsonl, fast_official_f1, official_module, score_predictions
from Assignment_3.src.runtime import ROOT, dump_json, fingerprint, load_data, sha256
from src.baselines.classical_features import build_ablated_preprocessor, build_feature_frame, build_interpretable_preprocessor
from src.baselines.common import load_config as baseline_config

CLASSIFIER_SYSTEMS = ["majority", "all_positive", "tfidf_logistic_regression", "tfidf_complement_nb",
    "word_char_tfidf_linear_svm", "eda_features_logistic_regression", "eda_features_random_forest",
    "ablated_no_word_id_logistic_regression", "frozen_xlmr_logistic_regression", "frozen_xlmr_linear_svm"]
RETRIEVAL_BASES = ["bm25", "dense", "hybrid", "dense_xlmr"]
RETRIEVAL_SYSTEMS = [f"{s}_{v}" for s in RETRIEVAL_BASES for v in ["naive_k50", "train_tuned"]] + [
    f"{s}_full_depth_tuned" for s in ["bm25", "dense", "hybrid"]]


def make_split(data, protocol, seed, fraction=.2):
    index = np.arange(len(data))
    if protocol == "stratified":
        train, test = train_test_split(index, test_size=fraction, random_state=seed, stratify=data.label_value)
    elif protocol == "grouped_text":
        splitter = StratifiedGroupKFold(n_splits=round(1/fraction), shuffle=True, random_state=seed)
        train, test = next(splitter.split(index, data.label_value, data.text_group))
    else:
        raise ValueError(f"Unknown split protocol: {protocol}")
    if set(train) & set(test) or len(train)+len(test) != len(data):
        raise ValueError("Invalid split identities")
    if set(data.iloc[train].word) != set(data.word) or set(data.iloc[test].word) != set(data.word):
        raise ValueError("A split lost a target lemma")
    overlap = set(data.iloc[train].text_group) & set(data.iloc[test].text_group)
    if protocol == "grouped_text" and overlap:
        raise ValueError("Duplicate text leakage in grouped split")
    return np.asarray(train), np.asarray(test)


def ranked_frame(results):
    return pd.DataFrame([{"query_id": word, **hit.as_dict()} for word, hits in sorted(results.items()) for hit in hits])


def fuse_frames(bm25, dense, k=60, depth=100):
    results = {}
    for word in sorted(bm25.query_id.unique()):
        lists = []
        for frame in [bm25, dense]:
            part = frame[(frame.query_id == word) & (frame["rank"] <= depth)].sort_values("rank")
            lists.append([Hit(str(r.doc_id), float(r.score), int(r.rank)) for r in part.itertuples()])
        results[word] = reciprocal_rank_fusion(lists, k=k, top_k=depth)
    return ranked_frame(results)


def build_rankings(config, output, data, definitions, measurements, log):
    folder = output / "rankings"
    folder.mkdir(parents=True, exist_ok=True)
    documents = [Document(doc_id=r.sentence_id, text=r.text, word=r.word) for r in data.itertuples()]
    bm25 = BM25Retriever({"k1": 1.5, "b": .75, "tokenizer": {"pattern": r"[\w]+", "lowercase": True}})
    with measurements.measure("index_build", system="bm25", cache_status="fresh", n_documents=len(data)):
        bm25.build_index(documents)
    bm25.save(output / "artifacts/bm25_index.pkl")
    bm_results, global_results = {}, {}
    for q in definitions.itertuples():
        allowed = data.loc[data.word == q.word, "sentence_id"].tolist()
        with measurements.measure("query_retrieve", system="bm25", word=q.word, depth=300):
            bm_results[q.word] = bm25.retrieve(q.definition, top_k=config["extended_retrieval_depth"], allowed_doc_ids=allowed)
        with measurements.measure("query_retrieve", system="bm25_global", word=q.word, depth=100):
            global_results[q.word] = bm25.retrieve(q.definition, top_k=100)
    rankings = {"bm25_full": ranked_frame(bm_results), "bm25_global": ranked_frame(global_results)}
    for encoder, system in [("e5_small", "dense"), ("xlmr_base", "dense_xlmr")]:
        with measurements.measure("index_load", system=system, cache_status="prepared_frozen_vectors"):
            cache = np.load(output / f"artifacts/{encoder}.npz")
            if list(cache["doc_ids"]) != data.sentence_id.tolist() or list(cache["query_words"]) != definitions.word.tolist():
                raise ValueError("Embedding cache identities mismatch")
            vectors, queries = cache["documents"], cache["queries"]
        results = {}
        for i, q in enumerate(definitions.itertuples()):
            allowed = data.word.to_numpy() == q.word
            with measurements.measure("query_retrieve", system=system, word=q.word, depth=300, encoding_included=False):
                scores = vectors[allowed] @ queries[i]
                results[q.word] = sort_and_rank(dict(zip(data.loc[allowed, "sentence_id"], scores)), top_k=300)
        rankings[f"{system}_full"] = ranked_frame(results)
        del cache, vectors, queries
    for system in ["bm25", "dense", "dense_xlmr"]:
        rankings[system] = rankings[f"{system}_full"].loc[lambda x: x["rank"] <= config["retrieval_depth"]].copy()
    with measurements.measure("rank_fusion", system="hybrid", depth=100, k=config["rrf_k"]):
        rankings["hybrid"] = fuse_frames(rankings["bm25"], rankings["dense"], k=config["rrf_k"], depth=100)
    rankings["hybrid_full"] = fuse_frames(rankings["bm25_full"], rankings["dense_full"], k=config["rrf_k"], depth=300)
    for k in config["rrf_k_grid"]:
        rankings[f"hybrid_k{k}"] = fuse_frames(rankings["bm25"], rankings["dense"], k=k, depth=100)
    for system, frame in rankings.items():
        frame.to_csv(folder / f"{system}.csv", index=False)
    qrels = data[["word", "sentence_id", "label_value"]].rename(columns={"word": "query_id", "sentence_id": "doc_id", "label_value": "relevance"})
    qrels.to_csv(folder / "qrels.csv", index=False)
    query_rows = []
    for system, frame in rankings.items():
        for word in sorted(data.word.unique()):
            metrics = evaluate_retrieval_run(frame[frame.query_id == word], qrels[qrels.query_id == word])
            query_rows.append({"system": system, "word": word, **metrics})
    pd.DataFrame(query_rows).to_csv(output / "tables/query_metrics.csv", index=False)
    log.info("Fresh rankings and all candidate/encoder/fusion/depth variants saved: %d", len(rankings))
    return rankings


def rank_predictions(rows, ranked, cutoff):
    index = pd.MultiIndex.from_frame(rows[["word", "sentence_id"]])
    ranks = ranked.set_index(["query_id", "doc_id"])["rank"].reindex(index).fillna(np.inf).to_numpy()
    return (ranks <= cutoff).astype(int)


def tune_cutoff(train, ranked, grid):
    choices = [(int(k), fast_official_f1(train.word, train.label_value, rank_predictions(train, ranked, k))) for k in grid]
    cutoff, score = max(choices, key=lambda pair: pair[1])
    return cutoff, score, choices


def classifiers(config, seed):
    models = build_classical_models(config, seed)
    models["all_positive"] = DummyClassifier(strategy="constant", constant=1)
    models["word_char_tfidf_linear_svm"].set_params(model__max_iter=20000)
    for name, preprocessor in [("eda_features_logistic_regression", build_interpretable_preprocessor(config)),
                               ("ablated_no_word_id_logistic_regression", build_ablated_preprocessor(config))]:
        models[name] = Pipeline([("features", preprocessor), ("model", LogisticRegression(
            max_iter=config["classical"]["max_iter"], class_weight="balanced", random_state=seed))])
    models["eda_features_random_forest"] = Pipeline([("features", build_interpretable_preprocessor(config)),
        ("model", RandomForestClassifier(n_estimators=100, max_depth=15, class_weight="balanced", random_state=seed, n_jobs=1))])
    models["frozen_xlmr_logistic_regression"] = Pipeline([("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))])
    models["frozen_xlmr_linear_svm"] = Pipeline([("scale", StandardScaler()),
        ("model", LinearSVC(max_iter=20000, class_weight="balanced", random_state=seed))])
    if set(models) != set(CLASSIFIER_SYSTEMS):
        raise ValueError("Missing classifier")
    return models


def run_experiments(config, output, log, measurements):
    data, definitions = load_data()
    for folder in ["tables", "predictions", "checkpoints", "splits", "official"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    dump_json(output / "config.json", config)
    baseline = baseline_config(ROOT)
    baseline.pop("project_root", None)
    dump_json(output / "baseline_config.json", baseline)
    rankings = build_rankings(config, output, data, definitions, measurements, log)
    cache = np.load(output / "artifacts/xlmr_base.npz")
    frozen = cache["classifier_contexts"]
    if list(cache["doc_ids"]) != data.sentence_id.tolist():
        raise ValueError("Classifier embedding identities mismatch")
    features = build_feature_frame(data, baseline)
    records, all_predictions, sweeps, split_records = [], [], [], []
    checkpoint_records = []

    def record(protocol, seed, system, test, pred, checkpoint, metadata):
        frame = test[["sentence_id", "word", "period_label", "label_value", "text_group"]].copy()
        frame["protocol"], frame["seed"], frame["system"], frame["prediction"] = protocol, seed, system, np.asarray(pred, dtype=int)
        metrics = score_predictions(frame)
        folder = output / f"predictions/{protocol}/seed_{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_csv(folder / f"{system}.csv", index=False)
        gold_path = output / f"official/{protocol}/seed_{seed}/gold/SV/subtask2.jsonl"
        pred_path = output / f"official/{protocol}/seed_{seed}/{system}/SV_subtask2.jsonl"
        export_jsonl(frame, gold_path, "label_value")
        export_jsonl(frame, pred_path, "prediction")
        # Exercise the official file loader as well as its scoring function.
        official_value = official_module().evaluate_subtask2(gold_path, pred_path)
        if not np.isclose(metrics["official_f1"], official_value, atol=1e-12, rtol=0):
            raise AssertionError("Official file evaluation mismatch")
        tn = int(((frame.label_value == 0) & (frame.prediction == 0)).sum())
        fp = int(((frame.label_value == 0) & (frame.prediction == 1)).sum())
        fn = int(((frame.label_value == 1) & (frame.prediction == 0)).sum())
        tp = int(((frame.label_value == 1) & (frame.prediction == 1)).sum())
        records.append({"protocol": protocol, "seed": seed, "system": system, "test_n": len(test),
                        "tn": tn, "fp": fp, "fn": fn, "tp": tp, **metadata, **metrics})
        all_predictions.append(frame)
        checkpoint_records.append({"protocol": protocol, "seed": seed, "system": system,
            "path": str(checkpoint.relative_to(output)).replace('\\', '/'),
            "bytes": checkpoint.stat().st_size, "sha256": sha256(checkpoint)})
        log.info("%s seed=%d %s official=%.4f pooled=%.4f", protocol, seed, system, metrics["official_f1"], metrics["pooled_macro_f1"])

    for protocol in config["split_protocols"]:
        for seed in config["seeds"]:
            train_index, test_index = make_split(data, protocol, seed, config["test_fraction"])
            train, test = data.iloc[train_index], data.iloc[test_index]
            shared = set(train.text_group) & set(test.text_group)
            split_rows = pd.concat([train.assign(split="train"), test.assign(split="test")])
            split_rows[["word", "sentence_id", "label_value", "text_group", "split"]].to_csv(output / f"splits/{protocol}_seed_{seed}.csv", index=False)
            split_records.append({"protocol": protocol, "seed": seed, "train_n": len(train), "test_n": len(test),
                "train_positive_n": int(train.label_value.sum()), "test_positive_n": int(test.label_value.sum()),
                "shared_text_groups": len(shared), "test_rows_with_train_text": int(test.text_group.isin(shared).sum()),
                "id_overlap": 0, "test_identity_sha256": fingerprint(sorted(test.sentence_id))})
            folder = output / f"checkpoints/{protocol}/seed_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            for system, model in classifiers(baseline, seed).items():
                if system.startswith("frozen_xlmr"):
                    x_train, x_test, input_kind = frozen[train_index], frozen[test_index], "frozen_xlmr_context"
                elif system.startswith("eda_features") or system.startswith("ablated"):
                    x_train, x_test, input_kind = features.iloc[train_index], features.iloc[test_index], "eda_frame"
                else:
                    x_train, x_test, input_kind = train.text, test.text, "text"
                with measurements.measure("classifier_fit", protocol=protocol, seed=seed, system=system, n=len(train)):
                    model.fit(x_train, train.label_value)
                with measurements.measure("classifier_predict", protocol=protocol, seed=seed, system=system,
                                          n=len(test), encoding_included=False):
                    predictions = model.predict(x_test)
                checkpoint = folder / f"{system}.joblib"
                with measurements.measure("checkpoint_save", protocol=protocol, seed=seed, system=system):
                    joblib.dump(model, checkpoint, compress=3)
                restored = joblib.load(checkpoint)
                if not np.array_equal(restored.predict(x_test), predictions):
                    raise AssertionError(f"Checkpoint replay mismatch: {system}")
                dump_json(checkpoint.with_suffix(".json"), {"system": system, "protocol": protocol, "seed": seed,
                    "input_kind": input_kind, "models": config["models"], "max_length": config["max_length"],
                    "checkpoint_sha256": sha256(checkpoint), "checkpoint_replayed": True,
                    "fit_rows": len(train), "test_rows": len(test),
                    "pipeline_parameters": {k: str(v) for k, v in model.get_params(deep=True).items()}})
                record(protocol, seed, system, test, predictions, checkpoint, {"cutoff": None, "checkpoint_replayed": True})
                del model, restored, x_train, x_test
                gc.collect()
            for system in RETRIEVAL_BASES:
                for variant in ["naive_k50", "train_tuned"] + (["full_depth_tuned"] if system != "dense_xlmr" else []):
                    base = f"{system}_full" if variant == "full_depth_tuned" else system
                    ranked = rankings[base]
                    if variant == "naive_k50":
                        cutoff, train_score = 50, None
                    else:
                        grid = config["extended_cutoff_grid"] if variant == "full_depth_tuned" else config["cutoff_grid"]
                        cutoff, train_score, choices = tune_cutoff(train, ranked, grid)
                        sweeps.extend({"protocol": protocol, "seed": seed, "system": system, "variant": variant,
                                       "cutoff": k, "train_official_f1": value} for k, value in choices)
                    name = f"{system}_{variant}"
                    with measurements.measure("retrieval_decision", protocol=protocol, seed=seed, system=name,
                                              n=len(test), ranking_and_encoding_included=False):
                        predictions = rank_predictions(test, ranked, cutoff)
                    checkpoint = folder / f"{name}.json"
                    dump_json(checkpoint, {"system": name, "ranking": f"rankings/{base}.csv", "cutoff": cutoff,
                        "selected_on": "fixed" if variant == "naive_k50" else "outer_train_labels_only",
                        "train_official_f1": train_score, "config_sha256": sha256(output / "config.json"),
                        "ranking_sha256": sha256(output / f"rankings/{base}.csv"), "frozen_encoder_profiles": config["models"]})
                    restored = json.loads(checkpoint.read_text(encoding="utf-8"))
                    replay = rank_predictions(test, pd.read_csv(output / restored["ranking"]), restored["cutoff"])
                    if not np.array_equal(predictions, replay):
                        raise AssertionError("Retrieval checkpoint replay mismatch")
                    record(protocol, seed, name, test, predictions, checkpoint,
                           {"cutoff": cutoff, "train_official_f1": train_score, "checkpoint_replayed": True})
            pd.DataFrame(records).to_csv(output / "tables/seed_metrics.csv", index=False)
            pd.concat(all_predictions, ignore_index=True).to_csv(output / "predictions/all_predictions.csv", index=False)
            dump_json(output / "checkpoints/manifest.json", checkpoint_records)
    pd.DataFrame(sweeps).to_csv(output / "tables/cutoff_sweeps_train_only.csv", index=False)
    pd.DataFrame(split_records).to_csv(output / "tables/split_audit.csv", index=False)
    log.info("All %d primary/classifier/ablation seed configurations executed and checkpoint-replayed", len(records))
    return pd.DataFrame(records)
