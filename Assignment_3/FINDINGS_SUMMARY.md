# Assignment 3 findings summary

The strongest observed local official F1 is frozen XLM-R + logistic regression: 0.489 ± 0.022 across five splits. Always-positive scores 0.470 ± 0.018, so the best classifier's advantage is small and is not confirmed by the exact five-pair test.

The hybrid's strongest result is early ranking: MRR 0.692 and nDCG@10 0.490, compared with dense retrieval's 0.612 and 0.427. Dense is better on MAP (0.323 versus 0.293) and official binary F1 (0.448 versus 0.418). Hybrid and dense both trail always-positive on the primary metric. These results support different systems for different objectives, not overall hybrid superiority.

Training-selected cutoffs improve official F1 by +0.122 (BM25), +0.121 (dense) and +0.106 (hybrid), but all choices reach the saved depth/search boundary K=100. Hybrid's pooled F1 improves only +0.003 and accuracy falls. Removing categorical word ID raises official F1 by +0.119 but lowers pooled F1 by -0.063; no morphological text masking was performed. The planned matched encoder retrieval ablation remains untested.

Hybrid makes 186/397 errors on the preselected seed-42 split. Across splits, 922 error appearances involve 631 unique usages; repeated test appearances are not independent cases. Four predefined decision categories separate false positives/negatives shared with dense from errors introduced by fusion. Eight examples preserve the original target definition, usage, labels, predictions and ranks. Surface overlap, noise and historical-era slices are descriptive associations, not verified linguistic causes.

Five overlapping splits and ten reused development queries give limited inferential power. Exact two-sided sign-flip tests on five pairs cannot attain p<0.05 (minimum 0.0625); no strong significance claim follows from a narrow split bootstrap interval. Published RRF, multilingual E5, XLM-R and SemEval-2020 results are explicitly non-comparable numerically because data or tasks differ; source links and reasons appear in the full report and comparison matrix.

Next steps: establish an untouched grouped/temporal/lemma holdout; complete matched encoder retrieval; retrieve deeper and tune on validation data; test validation-selected fusion or reranking; obtain blinded expert error annotations. Restore the ignored fitted checkpoints and measure actual per-model resource costs before making reproducibility or efficiency claims.

See [Sections 3-5](ANALYSIS_AND_COMPARISON_REPORT.md), [tables](outputs/tables/), [figures](outputs/figures/) and [analysis manifest](logs/part2_analysis_manifest.json). Reproduce with `python Assignment_3/scripts/run_part2_analysis.py`.
