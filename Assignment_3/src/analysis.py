"""Machine-readable statistics, ablations, errors, comparisons and figures.

No research paper, presentation, LaTeX section or narrative report is generated.
"""
from __future__ import annotations

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from Assignment_3.src.evaluation import score_predictions
from Assignment_3.src.runtime import ROOT, dump_json, load_data, text_group


def exact_sign_flip(differences):
    d = np.asarray(differences, dtype=float)
    if not len(d) or not np.isfinite(d).all():
        raise ValueError("Expected nonempty finite paired values")
    assignments = np.arange(2**len(d), dtype=np.uint64)[:, None]
    signs = 2*((assignments >> np.arange(len(d), dtype=np.uint64)) & 1).astype(float)-1
    return float(np.mean(abs((signs*d).mean(axis=1)) >= abs(d.mean())-1e-12))


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    result = np.empty(len(p))
    result[order] = np.minimum(1, np.maximum.accumulate(p[order]*np.arange(len(p), 0, -1)))
    return result


def interval(values, rng, n_boot):
    v = np.asarray(values)
    samples = v[rng.integers(len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def paired(a, b, rng, n_boot):
    d = np.asarray(a)-np.asarray(b)
    lo, hi = interval(d, rng, n_boot)
    return {"n_units": len(d), "mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)),
            "ci95_lo": lo, "ci95_hi": hi, "exact_p": exact_sign_flip(d),
            "wins": int((d>1e-12).sum()), "losses": int((d < -1e-12).sum()), "ties": int((abs(d)<=1e-12).sum())}


def category(gold, hybrid, dense):
    gold, hybrid, dense = map(np.asarray, (gold, hybrid, dense))
    return np.select([(gold==1)&(hybrid==0)&(dense==1), (gold==1)&(hybrid==0)&(dense==0),
                      (gold==0)&(hybrid==1)&(dense==0), (gold==0)&(hybrid==1)&(dense==1)],
                     ["FN fusion loss", "FN shared miss", "FP fusion addition", "FP shared false match"], default="correct")


def analyze(config, output, log):
    tables = output / "tables"
    def save(name, frame):
        frame.to_csv(tables / f"{name}.csv", index=False)
        return frame
    rng = np.random.default_rng(config["statistics_seed"])
    data, definitions = load_data()
    scores = pd.read_csv(tables / "seed_metrics.csv")
    pred = pd.read_csv(output / "predictions/all_predictions.csv")
    query = pd.read_csv(tables / "query_metrics.csv")
    n_boot = config["bootstrap_samples"]
    metrics = ["official_f1", "pooled_macro_f1", "positive_f1_per_word", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall"]
    summaries = []
    for (protocol, system), g in scores.groupby(["protocol", "system"]):
        for metric in metrics:
            lo, hi = interval(g[metric], rng, n_boot)
            summaries.append({"protocol": protocol, "system": system, "metric": metric, "n_seeds": len(g),
                "mean": g[metric].mean(), "std": g[metric].std(ddof=1), "ci95_lo": lo, "ci95_hi": hi,
                "ci_unit": "overlapping outer split; percentile bootstrap", "selection_metric": "official_f1"})
    summary = save("benchmark_summary", pd.DataFrame(summaries))
    rank_summary = []
    for system, g in query.groupby("system"):
        for metric in ["mrr", "map", "ndcg@10", "p@5", "p@10", "recall@50", "recall@100"]:
            lo, hi = interval(g[metric], rng, n_boot)
            rank_summary.append({"system": system, "metric": metric, "n_queries": len(g), "mean": g[metric].mean(),
                "std": g[metric].std(ddof=1), "ci95_lo": lo, "ci95_hi": hi})
    save("ranking_summary", pd.DataFrame(rank_summary))
    pairs = [("hybrid_train_tuned", "dense_train_tuned"), ("hybrid_naive_k50", "dense_naive_k50"),
             ("hybrid_train_tuned", "bm25_train_tuned"), ("dense_train_tuned", "dense_xlmr_train_tuned"),
             ("ablated_no_word_id_logistic_regression", "eda_features_logistic_regression"),
             ("frozen_xlmr_logistic_regression", "word_char_tfidf_linear_svm"),
             ("hybrid_train_tuned", "all_positive"),
             *[(f"{s}_train_tuned", f"{s}_naive_k50") for s in ["bm25", "dense", "hybrid", "dense_xlmr"]],
             *[(f"{s}_full_depth_tuned", f"{s}_train_tuned") for s in ["bm25", "dense", "hybrid"]]]
    paired_rows = []
    for protocol, group in scores.groupby("protocol"):
        for metric in ["official_f1", "pooled_macro_f1"]:
            pivot = group.pivot(index="seed", columns="system", values=metric).sort_index()
            for a, b in pairs:
                paired_rows.append({"protocol": protocol, "system_a": a, "system_b": b, "metric": metric,
                                    **paired(pivot[a], pivot[b], rng, n_boot)})
    paired_rows = pd.DataFrame(paired_rows)
    paired_rows["holm_p"] = 1.0
    for protocol, g in paired_rows.groupby("protocol"):
        paired_rows.loc[g.index, "holm_p"] = holm_adjust(g.exact_p)
    save("paired_seed_tests", paired_rows)
    qtests = []
    for a, b in [("bm25", "bm25_global"), ("dense", "dense_xlmr"), ("hybrid", "dense"),
                 ("hybrid_k30", "hybrid_k60"), ("hybrid_k60", "hybrid_k10")]:
        for metric in ["mrr", "map", "ndcg@10"]:
            pivot = query.pivot(index="word", columns="system", values=metric).sort_index()
            qtests.append({"system_a": a, "system_b": b, "metric": metric, **paired(pivot[a], pivot[b], rng, n_boot)})
    qtests = pd.DataFrame(qtests)
    qtests["holm_p"] = holm_adjust(qtests.exact_p)
    save("paired_query_tests", qtests)

    ablation_rows = []
    hypotheses = [
        *[("candidate_scope", "Per-word candidate restriction improves BM25 retrieval.", "bm25", "bm25_global", m, "query") for m in ["mrr", "map", "ndcg@10"]],
        *[("encoder_backbone", "Contrastive E5 improves matched dense retrieval over XLM-R.", "dense", "dense_xlmr", m, "query") for m in ["mrr", "map", "ndcg@10"]],
        *[("rrf_smoothing", "Intermediate k=30-60 improves retrieval quality.", "hybrid_k30", "hybrid_k60", m, "query") for m in ["mrr", "map", "ndcg@10"]],
        ("word_identity", "Removing categorical word identity improves the official metric.", "ablated_no_word_id_logistic_regression", "eda_features_logistic_regression", "official_f1", "seed"),
        *[("decision_cutoff", "Training-selected K improves official F1 over fixed K=50.", f"{s}_train_tuned", f"{s}_naive_k50", "official_f1", "seed") for s in ["bm25", "dense", "hybrid"]],
        *[("candidate_depth", "Deeper rankings and an extended training-only cutoff grid improve coverage and official F1.", f"{s}_full_depth_tuned", f"{s}_train_tuned", "official_f1", "seed") for s in ["bm25", "dense", "hybrid"]],
    ]
    for axis, hypothesis, a, b, metric, unit in hypotheses:
        if unit == "query":
            value = qtests[(qtests.system_a == a) & (qtests.system_b == b) & (qtests.metric == metric)].iloc[0]
            protocols = [("full_development_queries", value)]
        else:
            protocols = [(p, g.iloc[0]) for p, g in paired_rows[(paired_rows.system_a == a) & (paired_rows.system_b == b) & (paired_rows.metric == metric)].groupby("protocol")]
        for protocol, value in protocols:
            delta = float(value.mean_diff)
            verdict = "positive observed effect" if delta > 1e-12 else "negative observed effect" if delta < -1e-12 else "no observed effect"
            ablation_rows.append({"axis": axis, "hypothesis": hypothesis, "protocol": protocol, "system_a": a, "system_b": b,
                "metric": metric, "mean_difference": delta, "wins": int(value.wins), "losses": int(value.losses),
                "ci95_lo": value.ci95_lo, "ci95_hi": value.ci95_hi, "exact_p": value.exact_p, "holm_p": value.holm_p,
                "result": verdict, "significant_under_exploratory_test": bool(value.holm_p < .05),
                "limits": "Overlapping splits; five-pair exact p cannot reach .05." if unit == "seed" else "Ten reused development queries; query bootstrap and sign tests are exploratory.",
                "coverage_or_interaction": "Candidate depth and K-search range both change." if axis == "candidate_depth" else "Encoder size, pretraining and prefixes differ." if axis == "encoder_backbone" else "No text masking; POS, period and original surface forms remain." if axis == "word_identity" else "Global scope increases pool size and includes unjudged cross-word pairs." if axis == "candidate_scope" else "Selection uses outer training labels only; test labels are never optimized."})
    save("ablation_hypotheses", pd.DataFrame(ablation_rows))
    # Complete subgroup scores with the same official scorer, not a relabeled binary F1.
    subgroup_rows = []
    for (protocol, seed, system), group in pred.groupby(["protocol", "seed", "system"]):
        for word, g in group.groupby("word"):
            subgroup_rows.append({"protocol": protocol, "seed": seed, "system": system, "word": word,
                                  "n": len(g), "positives": int(g.label_value.sum()), **score_predictions(g)})
    save("per_word_metrics", pd.DataFrame(subgroup_rows))
    error_summary = scores[["protocol", "seed", "system", "test_n", "tp", "tn", "fp", "fn"]].copy()
    error_summary["error_rate"] = (error_summary.fp+error_summary.fn)/error_summary.test_n
    save("confusion_counts", error_summary)
    wide = pred.pivot(index=["protocol", "seed", "word", "sentence_id"], columns="system", values="prediction").reset_index()
    wide.columns.name = None
    wide = wide.merge(data, on=["word", "sentence_id"], validate="many_to_one")
    wide["error_category"] = category(wide.label_value, wide.hybrid_train_tuned, wide.dense_train_tuned)
    if not np.array_equal(wide.error_category != "correct", wide.label_value != wide.hybrid_train_tuned):
        raise AssertionError("Error taxonomy does not partition errors")
    for system in ["bm25", "dense", "hybrid"]:
        rank = pd.read_csv(output / f"rankings/{system}.csv").rename(columns={"query_id": "word", "doc_id": "sentence_id", "rank": f"{system}_rank"})
        wide = wide.merge(rank[["word", "sentence_id", f"{system}_rank"]], on=["word", "sentence_id"], how="left", validate="many_to_one")
    taxonomy, transitions, diagnostics = [], [], []
    wide["historical_before_1950"] = wide.year.astype(int) < 1950
    stop = set("och i en ett att som av på för med till den det de är var om från inte eller har hade men så vid sin sitt sina han hon vi du jag".split())
    def tokens(text):
        return {t for t in re.findall(r"[^\W\d_]+", text_group(text)) if len(t)>=3 and t not in stop}
    wide["zero_definition_overlap"] = [not(tokens(t)&tokens(d)) for t,d in zip(wide.text, wide.definition)]
    wide["surface_noise_proxy"] = wide.text.map(lambda t: bool(re.search(r"[\ufffd<>]|\b(?=\w*\d)(?=\w*[^\W\d_])\w+\b", t)))
    for (protocol, seed), group in wide.groupby(["protocol", "seed"]):
        errors = int((group.error_category != "correct").sum())
        for rule in config["error_taxonomy"]:
            n = int((group.error_category == rule["category"]).sum())
            taxonomy.append({"protocol": protocol, "seed": seed, "category": rule["category"], "count": n,
                             "test_n": len(group), "error_n": errors, "error_share": n/errors if errors else 0})
        harm = int(((group.dense_train_tuned == group.label_value)&(group.hybrid_train_tuned != group.label_value)).sum())
        repair = int(((group.dense_train_tuned != group.label_value)&(group.hybrid_train_tuned == group.label_value)).sum())
        transitions.append({"protocol": protocol, "seed": seed, "new_errors": harm, "repaired_errors": repair,
                            "net_extra_errors": harm-repair, "note": "Includes different training-selected cutoffs; fixed-K contrast is separately tested."})
        for flag in ["historical_before_1950", "zero_definition_overlap", "surface_noise_proxy"]:
            for value, g in group.groupby(flag):
                for system in ["bm25_train_tuned", "dense_train_tuned", "hybrid_train_tuned", "frozen_xlmr_logistic_regression", "word_char_tfidf_linear_svm"]:
                    diagnostics.append({"protocol": protocol, "seed": seed, "diagnostic": flag, "present": bool(value),
                        "system": system, "n": len(g), "positive_rate": float(g.label_value.mean()),
                        "error_rate": float((g[system] != g.label_value).mean()), "causal_annotation": False})
    taxonomy = save("error_taxonomy_by_seed", pd.DataFrame(taxonomy))
    save("error_taxonomy_summary", taxonomy.groupby(["protocol", "category"]).agg(mean_count=("count", "mean"),
        std_count=("count", "std"), mean_error_share=("error_share", "mean")).reset_index())
    save("fusion_error_transitions", pd.DataFrame(transitions))
    save("context_diagnostics", pd.DataFrame(diagnostics))
    examples = []
    for protocol in config["split_protocols"]:
        used = set()
        for rule in config["error_taxonomy"]:
            part = wide[(wide.protocol == protocol)&(wide.seed == config["example_seed"])&(wide.error_category == rule["category"])].copy()
            part["length_distance"] = (part.text.str.len()-180).abs()
            for _ in range(min(len(part), config["examples_per_category"])):
                part["word_used"] = part.word.isin(used)
                row = part.sort_values(["word_used", "length_distance", "sentence_id"]).iloc[0]
                used.add(row.word)
                rank_text = "; ".join(f"{s} rank {int(row[f'{s}_rank'])}" if pd.notna(row[f"{s}_rank"]) else f"{s}: outside saved top-100" for s in ["bm25", "dense", "hybrid"])
                examples.append({"protocol": protocol, "seed": int(row.seed), "category": row.error_category,
                    "word": row.word, "sentence_id": row.sentence_id, "period_label": row.period_label,
                    "target_definition": row.definition, "usage": row.text, "gold": int(row.label_value),
                    "hybrid_prediction": int(row.hybrid_train_tuned), "dense_prediction": int(row.dense_train_tuned),
                    "xlmr_prediction": int(row.frozen_xlmr_logistic_regression),
                    "explanation": f"{row.error_category}: gold={row.label_value}, hybrid={row.hybrid_train_tuned}, dense={row.dense_train_tuned}; {rank_text}. Compare with the supplied sense definition. Operational category; no annotation error or linguistic cause is assumed."})
                part = part[part.sentence_id != row.sentence_id]
    save("representative_errors", pd.DataFrame(examples))
    wide.to_csv(output / "predictions/annotated_test_usages.csv", index=False)
    coverage = wide.groupby(["protocol", "word", "sentence_id"]).agg(test_appearances=("seed", "count"),
        hybrid_error_appearances=("error_category", lambda x: int((x != "correct").sum()))).reset_index()
    save("unique_test_coverage", coverage)
    resource_path = output / "resource_measurements.jsonl"
    save("measured_resources", pd.read_json(resource_path, lines=True))
    published = json.loads((ROOT / "Assignment_3/configs/published_comparisons.json").read_text(encoding="utf-8"))
    for row in published:
        row["local_result"] = "Current official scores: tables/benchmark_summary.csv; retrieval scores: tables/ranking_summary.csv."
    local_rows = [{"work": f"Local {system}", "comparison_type": "direct", "reason": "Identical test identities, official scorer and outer seed for each protocol; label access and model costs are explicit.", "local_result": "benchmark_summary.csv"}
                  for system in sorted(scores.system.unique())]
    local_rows.append({"work": "Archived pre-correction Part 1 primary scores", "comparison_type": "non-comparable",
                       "reason": "Previously called positive-class per-word F1 official; actual official metric is within-word two-class macro-F1. Compare only explicitly matching secondary metrics."})
    save("comparison_matrix", pd.DataFrame(local_rows+published))
    selected = summary[(summary.protocol == config["primary_protocol_for_findings"])&(summary.metric == "official_f1")].sort_values("mean", ascending=False)
    best = selected.iloc[0]
    rank_mean = query.groupby("system")[["mrr", "map", "ndcg@10"]].mean()
    hypotheses_payload = pd.DataFrame(ablation_rows).to_dict("records")
    findings = {"protocol": config["primary_protocol_for_findings"], "official_metric": config["official_metric"],
        "best_observed_system": best.system, "mean_official_f1": best["mean"], "std_official_f1": best["std"],
        "ci95": [best.ci95_lo, best.ci95_hi], "claim_scope": "Local development data; model selected descriptively after inspecting results, not hidden-test superiority.",
        "primary_retrieval_means": selected[selected.system.isin(["bm25_train_tuned", "dense_train_tuned", "hybrid_train_tuned", "dense_xlmr_train_tuned"])][["system", "mean", "std"]].to_dict("records"),
        "ranking_tradeoffs": {m: {"best_observed_system": rank_mean.loc[["bm25", "dense", "hybrid", "dense_xlmr"],m].idxmax(),
                                     "values": rank_mean.loc[["bm25", "dense", "hybrid", "dense_xlmr"],m].to_dict()} for m in rank_mean.columns},
        "ablation_findings_including_negative_results": hypotheses_payload,
        "limitations": ["Five overlapping splits; split tests are exploratory; minimum two-sided exact p is 0.0625.",
                        "Only Swedish development data and ten target definitions; no hidden test, unseen-language, source-document or chronological holdout.",
                        "Exact-text grouping prevents exact duplicate overlap, not near-duplicate/source effects.",
                        "Operational error categories and surface proxies do not constitute expert linguistic annotation.",
                        "Monetary/energy cost unavailable; sampled RSS is not allocator-exact peak memory.",
                        "Extended-depth ablation changes both candidate coverage and calibration grid; matched-depth encoder comparison controls retrieval task but not pretraining/model capacity."],
        "next_steps": ["Evaluate a frozen selection on untouched hidden or held-out-source data.", "Obtain blinded error annotations for OCR/spelling, metaphor and sense ambiguity.", "Test validation-selected reranking/weighted fusion with independent assessment."]}
    dump_json(output / "findings.json", findings)
    dump_json(output / "analysis_protocol.json", {"taxonomy": config["error_taxonomy"],
        "example_selection": "Fixed seed 42; different lemmas preferred, then distance from 180 context characters, then sentence ID.",
        "diagnostics": {"historical_before_1950": "year<1950", "zero_definition_overlap": "Exact NFKC/casefold alphabetic token overlap, length>=3, fixed stopword list, no stemming",
                        "surface_noise_proxy": "Replacement character, angle bracket, or mixed digit/letter token; not verified OCR damage"},
        "statistics": {"bootstrap_samples": n_boot, "seed": config["statistics_seed"], "seed_tests_family": "all listed two-metric contrasts within each split protocol", "query_tests_family": "15 exploratory contrasts", "multiple_testing": "Holm"},
        "error_count_unit": "Test appearances; unique usage coverage saved separately. Different seed appearances are not independent observations."})
    figures(config, output, summary, taxonomy, query, paired_rows)
    log.info("Analysis tables, figures, error examples and JSON findings generated; no paper/report output")
    return scores, pred


def figures(config, output, summary, taxonomy, query, effects):
    folder = output / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(folder / f"{name}.png", dpi=220)
        fig.savefig(folder / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    for ax, protocol in zip(axes, config["split_protocols"]):
        part = summary[(summary.protocol == protocol)&(summary.metric == "official_f1")].sort_values("mean")
        ax.barh(part.system, part["mean"], xerr=np.maximum(0, np.vstack([part["mean"]-part.ci95_lo, part.ci95_hi-part["mean"]])), capsize=3)
        ax.set_title(protocol)
        ax.set_xlabel("Official within-word macro-F1: mean and split-bootstrap 95% interval")
    finish(fig, "01_official_benchmark")
    fig, ax = plt.subplots(figsize=(8, 4))
    for protocol, g in taxonomy.groupby("protocol"):
        part = g.groupby("category")["count"].agg(["mean", "std"])
        offset = -.18 if protocol == "stratified" else .18
        ax.bar(np.arange(len(part))+offset, part["mean"], yerr=part["std"], width=.35, label=protocol, capsize=3)
    ax.set_xticks(np.arange(len(part)), part.index, rotation=15, ha="right")
    ax.set_ylabel("Hybrid error count per split: mean +/- sample std")
    ax.legend()
    finish(fig, "02_error_categories")
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric in ["mrr", "map", "ndcg@10"]:
        values = [query[query.system == f"hybrid_k{k}"][metric].mean() for k in config["rrf_k_grid"]]
        ax.plot(config["rrf_k_grid"], values, "o-", label=metric)
    ax.set_xlabel("RRF smoothing k")
    ax.set_ylabel("Mean metric over ten development queries")
    ax.legend()
    finish(fig, "03_rrf_sensitivity")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, names in zip(axes, [["dense", "dense_xlmr"], ["bm25", "dense", "hybrid"]]):
        means = query[query.system.isin(names)].groupby("system")[["mrr", "map", "ndcg@10"]].mean().loc[names]
        means.plot.bar(ax=ax, rot=0)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Query mean")
    axes[0].set_title("Matched encoder retrieval ablation")
    axes[1].set_title("Early ranking and MAP tradeoff")
    finish(fig, "04_encoder_and_ranking_comparison")
    fig, ax = plt.subplots(figsize=(9, 6))
    part = effects[(effects.protocol == config["primary_protocol_for_findings"])&(effects.metric == "official_f1")].sort_values("mean_diff")
    for i, row in enumerate(part.itertuples()):
        ax.plot([row.ci95_lo, row.ci95_hi], [i, i], color="#3472a8")
        ax.plot(row.mean_diff, i, "o", color="black")
    ax.set_yticks(np.arange(len(part)), [f"{r.system_a} minus {r.system_b}" for r in part.itertuples()])
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_xlabel("Official F1 paired difference\n95% split-bootstrap interval (descriptive)")
    finish(fig, "05_paired_ablation_effects")
    sweep = pd.read_csv(output / "tables/cutoff_sweeps_train_only.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for system in ["bm25", "dense", "hybrid", "dense_xlmr"]:
        p = sweep[(sweep.protocol == config["primary_protocol_for_findings"])&(sweep.seed == config["example_seed"])&(sweep.system == system)&(sweep.variant == "train_tuned")]
        ax.plot(p.cutoff, p.train_official_f1, label=system)
    ax.set_xlabel("Cutoff K")
    ax.set_ylabel("Training-only official F1 (seed 42)")
    ax.legend()
    finish(fig, "06_training_cutoff_sweep")
