"""Render the Part 2 evidence into Markdown and matching LaTeX sections."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A3 = ROOT / "Assignment_3"
REPORT = ROOT / "report/assignment_3"
TABLES = A3 / "outputs/tables"
NAMES = {
    "bm25_train_tuned": "BM25 (K*)", "dense_train_tuned": "Dense (K*)",
    "hybrid_train_tuned": "Hybrid (K*)", "frozen_xlmr_logistic_regression": "Frozen XLM-R + LR",
    "word_char_tfidf_linear_svm": "Word + character SVM", "all_positive": "Always positive",
    "eda_features_logistic_regression": "EDA + LR",
    "ablated_no_word_id_logistic_regression": "EDA + LR without word ID",
}


def tex_escape(text):
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
               "±": r"$\pm$", "Δ": r"$\Delta$", "≥": r"$\geq$", "≤": r"$\leq$", "→": r"$\rightarrow$",
               "\u2011": "-", "\u2013": "--", "\u2014": "---"}
    mapping["\u00ad"] = r"\-"
    return "".join(mapping.get(c, c) for c in str(text))


class Document:
    def __init__(self, title):
        self.md = [f"# {title}\n"]
        self.tex = []

    def heading(self, text):
        self.md.append(f"### {text}\n")
        self.tex.append(r"\subsection{" + tex_escape(text) + "}\n")

    def paragraph(self, text):
        self.md.append(text + "\n")
        # Links are rendered natively in both formats.
        pieces, last = [], 0
        for match in re.finditer(r"\[([^\]]+)\]\((https?://[^)]+)\)", text):
            pieces.extend([tex_escape(text[last:match.start()]),
                           r"\href{" + match[2] + "}{" + tex_escape(match[1]) + "}"])
            last = match.end()
        pieces.append(tex_escape(text[last:]))
        self.tex.append("".join(pieces) + "\n\n")

    def table(self, caption, headers, rows, widths=None):
        self.md.append(caption + "\n")
        def cell(value):
            return str(value).replace("|", "\\|").replace("\n", " ")
        self.md.append("| " + " | ".join(map(cell, headers)) + " |\n" +
                       "| " + " | ".join(["---"] * len(headers)) + " |\n" +
                       "\n".join("| " + " | ".join(map(cell, row)) + " |" for row in rows) + "\n")
        widths = widths or [0.9 / len(headers)] * len(headers)
        columns = "".join(r">{\raggedright\arraybackslash}p{" + f"{w:.3f}" + r"\textwidth}" for w in widths)
        self.tex.extend(["{\\small\n\\setlength{\\tabcolsep}{3pt}\n",
                         r"\begin{longtable}{" + columns + "}\n",
                         r"\caption{" + tex_escape(caption) + r"}\\" + "\n\\toprule\n",
                         " & ".join(tex_escape(x) for x in headers) + " \\\\\n\\midrule\n\\endfirsthead\n",
                         " & ".join(tex_escape(x) for x in headers) + " \\\\\n\\midrule\n\\endhead\n"])
        self.tex.extend(" & ".join(tex_escape(x) for x in row) + " \\\\\n" for row in rows)
        self.tex.append("\\bottomrule\n\\end{longtable}\n}\n\n")

    def figure(self, name, caption):
        self.md.append(f"![{caption}](outputs/figures/{name}.png)\n\n{caption}\n")
        self.tex.append("\\begin{figure}[htbp]\n\\centering\n" +
                        r"\includegraphics[width=0.96\textwidth]{../../Assignment_3/outputs/figures/" + name + ".png}\n" +
                        r"\caption{" + tex_escape(caption) + "}\n\\end{figure}\n\n")


def fmt(value):
    return f"{value:.3f}"


def write_report(config, scores, summary, effects, query_scores, query_effects, wide, taxonomy,
                 tax_summary, overlap, coverage, examples, published):
    summary = summary.set_index("system")
    rank_means = query_scores.groupby("system")[["mrr", "map", "ndcg@10", "recall@50", "recall@100"]].mean()
    def effect(a, b, metric="official_f1"):
        return effects[(effects.system_a == a) & (effects.system_b == b) & (effects.metric == metric)].iloc[0]
    def effect_row(a, b):
        row = effect(a, b)
        return [f"{NAMES[a]} minus {NAMES.get(b, b.replace('_naive_k50', ' K=50').upper())}",
                f"{row.mean_diff:+.3f}", f"[{row.ci95_lo:+.3f}, {row.ci95_hi:+.3f}]",
                f"{int(row.wins)}/{int(row.losses)}/{int(row.ties)}", f"{row.exact_p:.4f}", f"{row.holm_p:.4f}"]
    hybrid = summary.loc["hybrid_train_tuned"]
    dense = summary.loc["dense_train_tuned"]
    xlmr = summary.loc["frozen_xlmr_logistic_regression"]
    always = summary.loc["all_positive"]
    n_test = int(scores.n.iloc[0])
    seed42 = wide[wide.seed == config["example_seed"]]
    hybrid_errors = int((seed42.hybrid_train_tuned != seed42.label_value).sum())
    unique_errors = int((coverage.hybrid_error_appearances > 0).sum())
    errors_appearances = int((wide.hybrid_train_tuned != wide.label_value).sum())
    thresholds = pd.read_json(A3 / "checkpoints/optimal_thresholds.json")
    assert len(thresholds.columns) == 5

    s3 = Document("3 Ablation analysis and reliability")
    s3.paragraph("This analysis uses the saved Part 1 predictions for 16 systems and five seeds (42, 100, 2023, 2026, 777). "
        "Official F1 below means the unchanged Part 1 scorer: positive-class F1 calculated for each of the ten lemmas, then averaged. "
        "Pooled macro-F1 averages the two classes over all test usages. These are local development-split results, not scores from an official hidden-test server.")
    s3.heading("Evidence and interpretation checks")
    s3.paragraph("All 31,760 saved prediction rows match the released labels and periods. All 80 per-seed files match the consolidated predictions, "
        "and each model has the same 397 test identities within each seed. Recomputed primary metric means match Part 1 Table 1; "
        "ranking metrics match Table 2. The four RRF smoothing variants are reconstructed from the saved component lists and match the Assignment 2 ablation table. "
        "Assignment 2's binary evaluator calls pooled two-class macro-F1 'official', whereas Part 1 uses per-lemma positive-class F1; this report retains the fixed Part 1 convention and names both metrics explicitly.")
    s3.heading("Hypotheses compared with results")
    s3.table("Ablation decisions. Full hypotheses and limitations are preserved in table8_ablation_hypotheses.csv.",
        ["Axis", "Observed result", "Conclusion"], [
        ["Candidate scope", "BM25 MRR: 0.618 per-word vs 0.207 global", "Per-word restriction helps this task"],
        ["Encoder backbone", "E5 retrieval exists; XLM-R retrieval does not", "Planned hypothesis remains untested"],
        ["RRF smoothing", "k=30 best MRR/nDCG; k=60 best MAP", "Metric-dependent, exploratory"],
        ["Decision cutoff", "Official F1 gains +0.106 to +0.122", "Consistent gains; search-boundary caveat"],
        ["Word-ID feature", "Official F1 +0.119; pooled macro-F1 -0.063", "Feature tradeoff; no text masking"]], [.18, .39, .33])
    s3.paragraph("Candidate scope hypothesis: ranking only usages of the requested lemma reduces irrelevant cross-word matches. "
        "Per-word BM25 has MRR 0.6183 and MAP 0.2291, against 0.2074 and 0.0171 globally. The large observed drop supports the restriction for definition matching. "
        "However, global retrieval expands the candidate universe, and cross-word pairs without judgments are treated as non-relevant by the evaluator. "
        "This is a difference in task difficulty and relevance coverage; it does not establish train/test leakage or prove a particular lexical failure mechanism.")
    s3.paragraph("Encoder hypothesis: contrastively trained E5 should outperform generic masked-language-model XLM-R embeddings for definition-to-usage retrieval. "
        "This cannot be tested with the saved runs. XLM-R plus logistic regression is a supervised classifier, not an XLM-R retriever with matched pooling, candidate depth and prefixes. "
        "Attributing its difference from E5 retrieval to encoder pretraining would confound the encoder with supervision and the prediction rule.")
    s3.paragraph("Fusion hypothesis: intermediate smoothing k=30-60 should give the strongest retrieval. "
        "At k=10/30/60/100, MRR is 0.6750/0.7083/0.6917/0.6917, MAP is 0.2788/0.2908/0.2926/0.2914, "
        "and nDCG@10 is 0.4718/0.4914/0.4902/0.4774. The preferred k depends on the metric; the k=30 versus 60 nDCG difference is only 0.0012. "
        "These ten already-inspected development queries provide sensitivity evidence, not independent confirmation of a newly selected k. Keep k=60 for the reported system.")
    s3.paragraph("Operating-point hypothesis: choosing K on training labels improves binary F1 relative to K=50. "
        "Official F1 increases by 0.1223 for BM25, 0.1210 for dense and 0.1056 for hybrid. All three systems choose K=100 in every seed: "
        "the maximum searched cutoff and saved retrieval depth. K* is therefore the best tested value, not an established unconstrained optimum. "
        "The objective matters: hybrid pooled macro-F1 changes only from 0.5213 to 0.5245, while accuracy falls from 0.5980 to 0.5355. "
        "More predicted positives improve the primary metric without uniformly improving classification quality.")
    s3.paragraph("Word-identity hypothesis: removing the explicit lemma feature may reduce dependence on lemma-specific class priors. "
        "Official F1 rises from 0.3409 to 0.4598, but pooled macro-F1 falls from 0.8107 to 0.7480 and accuracy from 0.8338 to 0.7662. "
        "Code inspection shows that build_ablated_preprocessor removes only the categorical word column; the original text, target surface forms, numeric features, POS and period remain. "
        "The earlier 'morphological masking' description is inaccurate. These results support a feature/metric tradeoff, not a claim that masking forces contextual semantic learning.")
    s3.figure("06_part2_cutoff_effects", "Cutoff gains for each split, with mean and sample standard deviation. The primary and secondary objectives yield different conclusions.")
    s3.heading("Variance and significance")
    s3.paragraph("Differences are paired within a seed. The table reports 10,000-resample percentile bootstrap intervals over five paired split values, "
        "win/loss/tie counts and exact two-sided sign-flip p-values over all 32 sign assignments. Holm adjustment covers all 20 classification contrasts defined in this analysis "
        "(ten pairs times two metrics). With five nonzero pairs, the smallest possible two-sided exact p-value is 2/32=0.0625, so these seed-level tests cannot reject at 0.05. "
        "A bootstrap interval excluding zero can coexist with that discrete test result. The 80/20 splits overlap substantially, and retrieval rankings are fixed across seeds; "
        "the standard deviations describe split sensitivity rather than five independent neural-training replications. Intervals and tests do not establish population-level or unseen-lemma generalization.")
    s3.table("Paired official F1 differences. Confidence intervals describe resampling of split values; W/L/T is descriptive.",
        ["Comparison", "Delta", "95% interval", "W/L/T", "Exact p", "Holm p"], [
        effect_row("hybrid_train_tuned", "dense_train_tuned"),
        effect_row("hybrid_train_tuned", "bm25_train_tuned"),
        effect_row("frozen_xlmr_logistic_regression", "all_positive"),
        effect_row("hybrid_train_tuned", "all_positive"),
        effect_row("hybrid_train_tuned", "hybrid_naive_k50"),
        effect_row("ablated_no_word_id_logistic_regression", "eda_features_logistic_regression")], [.28, .085, .19, .095, .105, .105])
    qrows = []
    for a, b, metric in [("bm25", "bm25_global", "map"), ("hybrid", "dense", "mrr"),
                          ("hybrid", "dense", "map"), ("hybrid", "dense", "ndcg@10"), ("rrf_30", "rrf_60", "ndcg@10")]:
        row = query_effects[(query_effects.system_a == a) & (query_effects.system_b == b) & (query_effects.metric == metric)].iloc[0]
        qrows.append([f"{a} minus {b}", metric, f"{row.mean_diff:+.4f}", f"{row.exact_p:.4f}", f"{row.holm_p:.4f}"])
    s3.table("Query-level sensitivity: exact sign flips across ten paired lemma queries, with Holm adjustment over 15 ranking contrasts.",
             ["Comparison", "Metric", "Delta", "Exact p", "Holm p"], qrows, [.34, .14, .14, .14, .14])
    s3.paragraph("The query tests are a separate exploratory family, not additional random-seed runs. The ten queries belong to the same development collection. "
        "Per-word versus global BM25 remains distinguishable for MAP and nDCG@10 after this adjustment (Holm p=0.0293), but none of the hybrid-versus-dense ranking contrasts does. "
        "Part 1's unadjusted t-tests and its Monte Carlo sign-flip estimates should not be read as definitive evidence of general superiority. "
        "The paired vectors, bootstrap settings and all adjusted tests are saved so each claim can be checked.")
    s3.heading("Leakage and artifact limitations")
    text_overlap = int(overlap.test_rows_with_exact_text_in_train.sum())
    duplicate_scores = pd.read_csv(TABLES / "part2_duplicate_sensitivity.csv")
    max_duplicate_change = duplicate_scores.difference.abs().max()
    s3.paragraph(f"Reconstructed splits have zero train/test identity overlap. Across the five splits, {text_overlap} test appearances have an exact NFKC-normalized, "
        "case-folded text match in their corresponding training partition. These duplicate contexts are a potential leakage channel despite distinct usage IDs; the corresponding pairs are saved. "
        f"Excluding those test rows changes any one reported split score by at most {max_duplicate_change:.4f} official F1 across the six focus systems, "
        "but that scoring-only sensitivity check cannot remove any influence already present during fitting. "
        "This check does not detect near-duplicate documents or shared source articles. "
        "All ten lemmas occur in both training and test partitions, and period metadata is available to some classifiers. "
        "The protocol measures new usages of known lemmas, not transfer to unseen lemmas or future periods. Full-corpus BM25 indexing and frozen passage encoding include test text without test labels; "
        "this is a transductive local setup. Training-only cutoff selection avoids direct test-label tuning but does not undo earlier development-set inspection.")
    checkpoint_count = len(list((A3 / "checkpoints").glob("*.joblib")))
    s3.paragraph(f"The current checkout contains {checkpoint_count} fitted joblib checkpoints; optimal_thresholds.json and predictions are present. "
        "The claim that 45 fitted pipelines are archived cannot be verified here because joblib files are ignored by Git. "
        "Part 1's per-model latency, memory and several cost values are hard-coded in its script rather than measured by a profiler, so they cannot support an efficiency ranking. "
        "The saved total wall time is a historical run log, not an end-to-end model-training cost. The new analysis manifest records its own environment, elapsed time and input hashes; no model training is repeated.")

    s4 = Document("4 Error analysis")
    s4.heading("Categories and counting protocol")
    s4.paragraph("Define a positive label as a usage expressing the supplied target definition; a negative label is another sense or usage. "
        "The following operational taxonomy is declared in configs/part2_analysis.json before aggregation. It partitions hybrid errors using the dense decision on the same instance. "
        "The categories are exhaustive and mutually exclusive, and describe observable decisions rather than guessed linguistic causes.")
    s4.table("Error definitions, fixed before counting.", ["Category", "Rule", "Meaning"],
             [[r["category"], r["rule"], r["interpretation"]] for r in config["error_taxonomy"]], [.22, .24, .44])
    s4.paragraph(f"Seed {config['example_seed']} is the preselected illustration split, with {n_test} usages and {hybrid_errors} hybrid errors. "
        f"Across five splits there are {len(wide):,} test appearances per system but only {len(coverage):,} unique usages. "
        f"Hybrid makes {errors_appearances:,} erroneous appearances involving {unique_errors:,} distinct usages. "
        "Repeated appearances of a usage are not independent cases. The following mean and sample standard deviation summarize counts per split; "
        "percentages average each split's fraction of hybrid errors, rather than treating all appearances as an independent sample.")
    s4.table("Hybrid error counts and proportions.", ["Category", "Mean +/- std count", "Mean share of errors", "Seed 42 count"],
        [[r.category, f"{r.mean_count:.1f} ± {r.std_count:.1f}", f"{r.mean_error_share:.1%}", int(r.seed42_count)]
         for r in tax_summary.itertuples()], [.28, .22, .24, .16])
    s4.figure("07_part2_error_taxonomy", "Hybrid errors separated into shared failures and differences introduced by fusion. These are decision categories, not human linguistic annotations.")
    s4.table("Confusion counts averaged across five 397-usage test splits. FP and FN use the positive target sense.",
        ["System", "Mean TP", "Mean TN", "Mean FP", "Mean FN", "Error rate"],
        [[NAMES[s], f"{summary.loc[s,'mean_tp']:.1f}", f"{summary.loc[s,'mean_tn']:.1f}",
          f"{summary.loc[s,'mean_fp']:.1f}", f"{summary.loc[s,'mean_fn']:.1f}", f"{summary.loc[s,'mean_error_rate']:.1%}"]
         for s in ["bm25_train_tuned", "dense_train_tuned", "hybrid_train_tuned", "frozen_xlmr_logistic_regression", "word_char_tfidf_linear_svm", "all_positive"]],
        [.27, .12, .12, .12, .12, .15])
    s4.paragraph(f"Hybrid averages {hybrid.mean_fp:.1f} false positives and {hybrid.mean_fn:.1f} false negatives, versus {dense.mean_fp:.1f} and {dense.mean_fn:.1f} for dense. "
        "The fusion-loss and fusion-addition categories identify exactly where replacing the dense decision with hybrid harms a gold label. "
        "At the saved K*=100, every hybrid false negative is absent from its saved top-100 list; it is not merely below the binary cutoff inside that list. "
        "The unsaved tail is unknown, so 'absent' does not mean absent from the corpus or prove the encoder never retrieved the usage. "
        "A larger candidate depth or reranking may address these cases, but that is a future experiment.")
    transitions = pd.read_csv(TABLES / "part2_fusion_transitions.csv").drop(columns="seed").mean()
    harm = transitions.introduced_fn + transitions.introduced_fp
    repair = transitions.recovered_fn + transitions.removed_fp
    s4.paragraph(f"Fusion also repairs some dense errors: per split it recovers {transitions.recovered_fn:.1f} false negatives and removes {transitions.removed_fp:.1f} false positives. "
        f"Those {repair:.1f} repairs are outweighed by {harm:.1f} newly introduced errors "
        f"({transitions.introduced_fn:.1f} false negatives and {transitions.introduced_fp:.1f} false positives), a net {harm-repair:.1f} additional errors per split. "
        "The saved transition table verifies this accounting against the confusion totals; it explains the overall error-rate change without assigning speculative linguistic causes.")
    s4.heading("Representative errors")
    s4.paragraph("Examples use the fixed seed and selection rule in the configuration: prefer different lemmas and contexts nearest 180 characters, "
        "then break ties by sentence ID. Two examples are selected from each category. The original Swedish definition and full usage are preserved without normalization or translation. "
        "Gold labels are copied from the dataset; these examples do not establish annotation errors. Rank absence always refers to the saved list.")
    for i, ex in enumerate(examples, 1):
        s4.paragraph(f"Example {i}: {ex['category']}; lemma {ex['word']}; sentence {ex['sentence_id']}; period {ex['period']}.")
        s4.paragraph(f"Target definition: {ex['target_definition']}")
        s4.paragraph(f"Usage: {ex['usage']}")
        s4.paragraph(f"Gold target-sense label: {ex['gold']}; hybrid prediction: {ex['hybrid_prediction']}; "
                     f"dense prediction: {ex['dense_prediction']}; XLM-R + LR prediction: {ex['xlmr_prediction']}.")
        s4.paragraph(f"Explanation: {ex['explanation']}")
        s4.paragraph(f"Illustrative reading: {ex['qualitative_reading']}")
    s4.heading("Lemmas and context slices")
    perword = pd.read_csv(TABLES / "part2_per_word_errors.csv")
    pw = perword.groupby(["word", "system"]).official_f1.mean().unstack()
    prevalence = wide.groupby("word").label_value.mean()
    s4.table("Per-lemma primary F1 averaged within seed, then across seeds. Positive prevalence uses observed test appearances.",
        ["Lemma", "Gold positive rate", "Hybrid", "Dense", "XLM-R + LR"],
        [[word, f"{prevalence[word]:.1%}", fmt(pw.loc[word, "hybrid_train_tuned"]),
          fmt(pw.loc[word, "dense_train_tuned"]), fmt(pw.loc[word, "frozen_xlmr_logistic_regression"])] for word in sorted(pw.index)],
        [.18, .22, .16, .16, .18])
    s4.paragraph("The lemma table exposes heterogeneity hidden by the overall score. Dense exceeds hybrid most clearly on motiv and förort, "
        "while hybrid exceeds dense on dumpa. Both have low positive-class F1 on fröken and herre. "
        "For skär the pool contains 98 usages, so K=100 selects the entire pool for both retrievers: their identical F1 is an all-positive operating-point effect, "
        "not evidence that their rankings or sense representations are identical.")
    diagnostics = pd.read_csv(TABLES / "part2_context_diagnostics.csv")
    rows = []
    for diagnostic, label in [("historical", "Before 1950"), ("zero_definition_overlap", "Zero definition overlap"), ("noise_marker", "Surface noise marker")]:
        for value in [True, False]:
            g = diagnostics[(diagnostics.diagnostic == diagnostic) & (diagnostics.present == value)]
            rates = g.groupby("system").error_rate.mean()
            mix = g[g.system == "hybrid_train_tuned"]
            rows.append([label, "yes" if value else "no", f"{mix.n.mean():.1f}", f"{mix.gold_positive_rate.mean():.1%}",
                         f"{rates['hybrid_train_tuned']:.1%}", f"{rates['dense_train_tuned']:.1%}"])
    s4.table("Context diagnostics: mean counts and error rates across splits. Each diagnostic is a separate overlapping slice.",
        ["Diagnostic", "Present", "Mean n", "Positive rate", "Hybrid error", "Dense error"], rows, [.25, .10, .12, .14, .14, .15])
    s4.paragraph("Historical means year<1950 and is a descriptive slice rather than a chronological holdout. Zero definition overlap means no shared exact alphabetic token "
        "of length at least three after NFKC normalization, lowercasing and removal of the fixed small Swedish function-word list in the script; no stemming is applied. "
        "The surface-noise flag detects a replacement character, angle bracket, or mixed letter/digit token. It is a narrow proxy, not an annotation of OCR quality. "
        "Different class prevalence and lemma composition confound these slices; none establishes that date, lexical mismatch or OCR causes an error. "
        "Manual categories such as metaphor, historical spelling and genuine sense ambiguity require a blinded linguistic annotation protocol and are not assigned fabricated counts.")
    s4.figure("08_part2_word_and_era_errors", "Per-lemma F1 and error rates before/after 1950. Error bars show sample standard deviation across overlapping splits; era class mixtures differ.")

    s5 = Document("5 Comparison findings and next steps")
    s5.heading("Direct comparisons with the baselines")
    s5.paragraph("A direct comparison uses identical usage IDs, seed splits and metric definitions. All current Part 1 reruns meet those conditions. "
        "Supervised baselines train on 1,586 labeled usages per seed; frozen retrieval uses training labels to choose K but does not train its encoder locally. "
        "The table compares outcomes under this shared protocol without treating their label budgets as identical.")
    s5.table("Direct local comparison. Uncertainty is sample standard deviation across five splits.",
        ["System", "Official F1 mean +/- std", "Pooled macro-F1", "Accuracy"],
        [[NAMES[s], f"{summary.loc[s,'mean_official_f1']:.3f} ± {summary.loc[s,'std_official_f1']:.3f}",
          fmt(summary.loc[s,"mean_pooled_f1"]), fmt(summary.loc[s,"mean_accuracy"])]
         for s in ["frozen_xlmr_logistic_regression", "all_positive", "word_char_tfidf_linear_svm", "dense_train_tuned", "hybrid_train_tuned", "bm25_train_tuned"]],
        [.30, .26, .19, .15])
    s5.paragraph(f"The strongest observed primary result is frozen XLM-R + LR at {xlmr.mean_official_f1:.3f} ± {xlmr.std_official_f1:.3f}. "
        f"The always-positive baseline reaches {always.mean_official_f1:.3f} ± {always.std_official_f1:.3f}; XLM-R's mean advantage is only "
        f"{xlmr.mean_official_f1-always.mean_official_f1:+.3f}, with no exact seed-level rejection at 0.05. "
        f"Hybrid ({hybrid.mean_official_f1:.3f}) and dense ({dense.mean_official_f1:.3f}) both trail this trivial baseline on the primary metric, "
        "while giving better pooled macro-F1. The high always-positive score is consistent with a positive-class metric and is not itself evidence of a scoring bug. "
        "Reporting only accuracy or only the primary F1 would conceal this tradeoff.")
    s5.table("Retrieval comparison on the same ten definition queries. MAP and MRR are evaluated on saved lists truncated at 100.",
        ["System", "MRR", "MAP", "nDCG@10", "Recall@50"],
        [[s.upper(), *[fmt(rank_means.loc[s, m]) for m in ["mrr", "map", "ndcg@10", "recall@50"]]] for s in ["bm25", "dense", "hybrid"]])
    s5.paragraph("Hybrid improves early ranking over dense (MRR 0.6917 versus 0.6117; nDCG@10 0.4902 versus 0.4274), "
        "but dense has higher MAP (0.3229 versus 0.2926) and binary official F1 (0.4484 versus 0.4182). "
        "MRR rewards the first relevant usage and nDCG@10 the leading ranks; they do not guarantee better coverage of all positives or a better thresholded classifier. "
        "The actual macro Recall@50 values are BM25 0.2847, dense 0.3889 and hybrid 0.3812, correcting the larger values stated in the earlier prose. "
        "Each remains a local development ranking result; no hidden-test ranking outcome is available.")
    s5.figure("09_part2_metric_tradeoffs", "Different metrics favor different systems: early ranking favors hybrid, while MAP and binary F1 favor dense; the strongest primary classifier is XLM-R + LR.")
    s5.heading("Published work and comparability")
    s5.paragraph("Direct means matching data, split and metric; approximate means partial protocol alignment that prevents a strict improvement claim; "
        "non-comparable means that the task, evaluation unit, relevance judgments or metric differs enough to rule out a numerical performance comparison. "
        "Archived Assignment 1 aggregate scores are approximate to the current multi-seed summary: they use one seed and foreground pooled macro-F1. "
        "The current reruns, rather than those old aggregates, supply the direct baseline comparisons.")
    s5.table("Published comparisons. No directly matched published test result is available in these sources.",
        ["Published work", "Label", "Reason"], [[p["work"], p["comparison_type"], p["reason"]] for p in published], [.23, .17, .50])
    for p in published:
        s5.paragraph(f"[{p['work']}: {p['title']}]({p['url']}). {p['published_result']} {p['local_result']} "
            "These scores are not subtracted across benchmarks.")
    s5.paragraph("The comparisons support three methodological connections: rank fusion can improve leading hits without improving all metrics; "
        "retrieval-oriented multilingual embeddings motivate E5, but do not establish superiority on this historical corpus; and word-level change detection in SemEval-2020 "
        "has a different prediction target from usage-level definition matching. The component methods are established techniques; the local contribution is their application and evaluation, "
        "not the invention of BM25, multilingual encoding or RRF. Sources and table locations are recorded in configs/published_comparisons.json (checked 21 November 2026).")
    s5.heading("Findings limitations and next steps")
    s5.paragraph("Strongest finding: the best observed local primary score belongs to the frozen XLM-R supervised classifier; "
        "the proposed hybrid offers a narrower benefit in early retrieval. Negative result: hybrid does not beat dense on MAP or binary F1, "
        "and neither retriever beats always-positive on official F1. Unexpected result: removing explicit word ID helps official F1 but hurts pooled F1 and accuracy. "
        "The intermediate RRF smoothing hypothesis is metric-dependent, and the backbone hypothesis remains untested.")
    s5.paragraph("The evidence is limited to 1,983 Swedish development usages, ten lemmas, ten definitions and five overlapping random splits. "
        "There is no held-out language, source-document split, future-period test or unseen-lemma evaluation. Development-set reuse, the fixed top-100 tail, "
        "small-sample significance resolution, missing fitted checkpoints and unmeasured per-model cost limit broader claims. "
        "Automatic operational categories are reproducible but do not replace expert sense/error annotation.")
    next_steps = [
        "1. Separate model selection from final assessment: reserve an untouched set, group near-duplicate/source documents where provenance permits, "
        "and add lemma-held-out and chronological evaluations. Fit vocabularies and any inductive corpus statistics using training data only.",
        "2. Complete the matched E5 versus XLM-R retrieval ablation with identical definitions, pooling, candidate scope and depth. "
        "Report different pretraining, dimensions and task prefixes explicitly rather than attributing every difference to contrastive learning alone.",
        "3. Regenerate deeper component rankings and tune cutoffs or score thresholds on validation data beyond the present K=100 boundary. "
        "Compare against always-positive, optimize the declared primary objective and retain secondary metrics to expose tradeoffs.",
        "4. Test whether fusion-loss cases improve with validation-selected weighting, deeper candidates or reranking. "
        "Evaluate the final frozen choice once on untouched data; do not select a new k using the reported evaluation queries.",
        "5. Blindly double-annotate a stratified sample for historical spelling/OCR, metaphor, sense ambiguity and definition mismatch; report agreement and prevalence "
        "before making causal linguistic claims. Archive fitted models and measure encoding, indexing, inference time, peak memory and resource cost separately."
    ]
    for step in next_steps:
        s5.paragraph(step)
    s5.heading("Reproducible outputs")
    s5.paragraph("Run python Assignment_3/scripts/run_part2_analysis.py from the repository root. The run validates saved predictions, reconstructs splits "
        "and RRF variants, produces tables 8-12 and supporting CSVs, writes annotated_test_usages.csv, saves four figures as PNG and PDF, "
        "and regenerates this report, the findings summary and LaTeX Sections 3-5. The analysis manifest records input SHA-256 hashes, "
        "configuration, environment, verification checks and analysis elapsed time. Models and the fixed evaluation functions are unchanged.")

    documents = [s3, s4, s5]
    md = "# Assignment 3 analysis and comparison\n\n" + "\n\n".join("\n".join(d.md).replace("# ", "## ", 1) for d in documents)
    (A3 / "ANALYSIS_AND_COMPARISON_REPORT.md").write_text(md, encoding="utf-8")
    for d, name in zip(documents, ["03_ablation_analysis", "04_error_analysis", "05_comparison_and_findings"]):
        (REPORT / "sections" / f"{name}.tex").write_text("".join(d.tex), encoding="utf-8")
    findings = f"""# Assignment 3 findings summary

The strongest observed local official F1 is frozen XLM-R + logistic regression: {xlmr.mean_official_f1:.3f} ± {xlmr.std_official_f1:.3f} across five splits. Always-positive scores {always.mean_official_f1:.3f} ± {always.std_official_f1:.3f}, so the best classifier's advantage is small and is not confirmed by the exact five-pair test.

The hybrid's strongest result is early ranking: MRR 0.692 and nDCG@10 0.490, compared with dense retrieval's 0.612 and 0.427. Dense is better on MAP (0.323 versus 0.293) and official binary F1 (0.448 versus 0.418). Hybrid and dense both trail always-positive on the primary metric. These results support different systems for different objectives, not overall hybrid superiority.

Training-selected cutoffs improve official F1 by +0.122 (BM25), +0.121 (dense) and +0.106 (hybrid), but all choices reach the saved depth/search boundary K=100. Hybrid's pooled F1 improves only +0.003 and accuracy falls. Removing categorical word ID raises official F1 by +0.119 but lowers pooled F1 by -0.063; no morphological text masking was performed. The planned matched encoder retrieval ablation remains untested.

Hybrid makes {hybrid_errors}/{n_test} errors on the preselected seed-42 split. Across splits, {errors_appearances} error appearances involve {unique_errors} unique usages; repeated test appearances are not independent cases. Four predefined decision categories separate false positives/negatives shared with dense from errors introduced by fusion. Eight examples preserve the original target definition, usage, labels, predictions and ranks. Surface overlap, noise and historical-era slices are descriptive associations, not verified linguistic causes.

Five overlapping splits and ten reused development queries give limited inferential power. Exact two-sided sign-flip tests on five pairs cannot attain p<0.05 (minimum 0.0625); no strong significance claim follows from a narrow split bootstrap interval. Published RRF, multilingual E5, XLM-R and SemEval-2020 results are explicitly non-comparable numerically because data or tasks differ; source links and reasons appear in the full report and comparison matrix.

Next steps: establish an untouched grouped/temporal/lemma holdout; complete matched encoder retrieval; retrieve deeper and tune on validation data; test validation-selected fusion or reranking; obtain blinded expert error annotations. Restore the ignored fitted checkpoints and measure actual per-model resource costs before making reproducibility or efficiency claims.

See [Sections 3-5](ANALYSIS_AND_COMPARISON_REPORT.md), [tables](outputs/tables/), [figures](outputs/figures/) and [analysis manifest](logs/part2_analysis_manifest.json). Reproduce with `python Assignment_3/scripts/run_part2_analysis.py`.
"""
    (A3 / "FINDINGS_SUMMARY.md").write_text(findings, encoding="utf-8")
