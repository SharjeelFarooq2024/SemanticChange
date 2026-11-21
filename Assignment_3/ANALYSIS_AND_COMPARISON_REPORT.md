# Assignment 3 analysis and comparison

## 3 Ablation analysis and reliability

This analysis uses the saved Part 1 predictions for 16 systems and five seeds (42, 100, 2023, 2026, 777). Official F1 below means the unchanged Part 1 scorer: positive-class F1 calculated for each of the ten lemmas, then averaged. Pooled macro-F1 averages the two classes over all test usages. These are local development-split results, not scores from an official hidden-test server.

### Evidence and interpretation checks

All 31,760 saved prediction rows match the released labels and periods. All 80 per-seed files match the consolidated predictions, and each model has the same 397 test identities within each seed. Recomputed primary metric means match Part 1 Table 1; ranking metrics match Table 2. The four RRF smoothing variants are reconstructed from the saved component lists and match the Assignment 2 ablation table. Assignment 2's binary evaluator calls pooled two-class macro-F1 'official', whereas Part 1 uses per-lemma positive-class F1; this report retains the fixed Part 1 convention and names both metrics explicitly.

### Hypotheses compared with results

Ablation decisions. Full hypotheses and limitations are preserved in table8_ablation_hypotheses.csv.

| Axis | Observed result | Conclusion |
| --- | --- | --- |
| Candidate scope | BM25 MRR: 0.618 per-word vs 0.207 global | Per-word restriction helps this task |
| Encoder backbone | E5 retrieval exists; XLM-R retrieval does not | Planned hypothesis remains untested |
| RRF smoothing | k=30 best MRR/nDCG; k=60 best MAP | Metric-dependent, exploratory |
| Decision cutoff | Official F1 gains +0.106 to +0.122 | Consistent gains; search-boundary caveat |
| Word-ID feature | Official F1 +0.119; pooled macro-F1 -0.063 | Feature tradeoff; no text masking |

Candidate scope hypothesis: ranking only usages of the requested lemma reduces irrelevant cross-word matches. Per-word BM25 has MRR 0.6183 and MAP 0.2291, against 0.2074 and 0.0171 globally. The large observed drop supports the restriction for definition matching. However, global retrieval expands the candidate universe, and cross-word pairs without judgments are treated as non-relevant by the evaluator. This is a difference in task difficulty and relevance coverage; it does not establish train/test leakage or prove a particular lexical failure mechanism.

Encoder hypothesis: contrastively trained E5 should outperform generic masked-language-model XLM-R embeddings for definition-to-usage retrieval. This cannot be tested with the saved runs. XLM-R plus logistic regression is a supervised classifier, not an XLM-R retriever with matched pooling, candidate depth and prefixes. Attributing its difference from E5 retrieval to encoder pretraining would confound the encoder with supervision and the prediction rule.

Fusion hypothesis: intermediate smoothing k=30-60 should give the strongest retrieval. At k=10/30/60/100, MRR is 0.6750/0.7083/0.6917/0.6917, MAP is 0.2788/0.2908/0.2926/0.2914, and nDCG@10 is 0.4718/0.4914/0.4902/0.4774. The preferred k depends on the metric; the k=30 versus 60 nDCG difference is only 0.0012. These ten already-inspected development queries provide sensitivity evidence, not independent confirmation of a newly selected k. Keep k=60 for the reported system.

Operating-point hypothesis: choosing K on training labels improves binary F1 relative to K=50. Official F1 increases by 0.1223 for BM25, 0.1210 for dense and 0.1056 for hybrid. All three systems choose K=100 in every seed: the maximum searched cutoff and saved retrieval depth. K* is therefore the best tested value, not an established unconstrained optimum. The objective matters: hybrid pooled macro-F1 changes only from 0.5213 to 0.5245, while accuracy falls from 0.5980 to 0.5355. More predicted positives improve the primary metric without uniformly improving classification quality.

Word-identity hypothesis: removing the explicit lemma feature may reduce dependence on lemma-specific class priors. Official F1 rises from 0.3409 to 0.4598, but pooled macro-F1 falls from 0.8107 to 0.7480 and accuracy from 0.8338 to 0.7662. Code inspection shows that build_ablated_preprocessor removes only the categorical word column; the original text, target surface forms, numeric features, POS and period remain. The earlier 'morphological masking' description is inaccurate. These results support a feature/metric tradeoff, not a claim that masking forces contextual semantic learning.

![Cutoff gains for each split, with mean and sample standard deviation. The primary and secondary objectives yield different conclusions.](outputs/figures/06_part2_cutoff_effects.png)

Cutoff gains for each split, with mean and sample standard deviation. The primary and secondary objectives yield different conclusions.

### Variance and significance

Differences are paired within a seed. The table reports 10,000-resample percentile bootstrap intervals over five paired split values, win/loss/tie counts and exact two-sided sign-flip p-values over all 32 sign assignments. Holm adjustment covers all 20 classification contrasts defined in this analysis (ten pairs times two metrics). With five nonzero pairs, the smallest possible two-sided exact p-value is 2/32=0.0625, so these seed-level tests cannot reject at 0.05. A bootstrap interval excluding zero can coexist with that discrete test result. The 80/20 splits overlap substantially, and retrieval rankings are fixed across seeds; the standard deviations describe split sensitivity rather than five independent neural-training replications. Intervals and tests do not establish population-level or unseen-lemma generalization.

Paired official F1 differences. Confidence intervals describe resampling of split values; W/L/T is descriptive.

| Comparison | Delta | 95% interval | W/L/T | Exact p | Holm p |
| --- | --- | --- | --- | --- | --- |
| Hybrid (K*) minus Dense (K*) | -0.030 | [-0.042, -0.015] | 0/5/0 | 0.0625 | 1.0000 |
| Hybrid (K*) minus BM25 (K*) | +0.060 | [+0.044, +0.080] | 5/0/0 | 0.0625 | 1.0000 |
| Frozen XLM-R + LR minus Always positive | +0.019 | [-0.001, +0.035] | 4/1/0 | 0.1875 | 1.0000 |
| Hybrid (K*) minus Always positive | -0.052 | [-0.072, -0.029] | 0/5/0 | 0.0625 | 1.0000 |
| Hybrid (K*) minus HYBRID K=50 | +0.106 | [+0.072, +0.137] | 5/0/0 | 0.0625 | 1.0000 |
| EDA + LR without word ID minus EDA + LR | +0.119 | [+0.092, +0.150] | 5/0/0 | 0.0625 | 1.0000 |

Query-level sensitivity: exact sign flips across ten paired lemma queries, with Holm adjustment over 15 ranking contrasts.

| Comparison | Metric | Delta | Exact p | Holm p |
| --- | --- | --- | --- | --- |
| bm25 minus bm25_global | map | +0.2120 | 0.0020 | 0.0293 |
| hybrid minus dense | mrr | +0.0800 | 0.6250 | 1.0000 |
| hybrid minus dense | map | -0.0303 | 0.5859 | 1.0000 |
| hybrid minus dense | ndcg@10 | +0.0628 | 0.4180 | 1.0000 |
| rrf_30 minus rrf_60 | ndcg@10 | +0.0012 | 0.9688 | 1.0000 |

The query tests are a separate exploratory family, not additional random-seed runs. The ten queries belong to the same development collection. Per-word versus global BM25 remains distinguishable for MAP and nDCG@10 after this adjustment (Holm p=0.0293), but none of the hybrid-versus-dense ranking contrasts does. Part 1's unadjusted t-tests and its Monte Carlo sign-flip estimates should not be read as definitive evidence of general superiority. The paired vectors, bootstrap settings and all adjusted tests are saved so each claim can be checked.

### Leakage and artifact limitations

Reconstructed splits have zero train/test identity overlap. Across the five splits, 7 test appearances have an exact NFKC-normalized, case-folded text match in their corresponding training partition. These duplicate contexts are a potential leakage channel despite distinct usage IDs; the corresponding pairs are saved. Excluding those test rows changes any one reported split score by at most 0.0026 official F1 across the six focus systems, but that scoring-only sensitivity check cannot remove any influence already present during fitting. This check does not detect near-duplicate documents or shared source articles. All ten lemmas occur in both training and test partitions, and period metadata is available to some classifiers. The protocol measures new usages of known lemmas, not transfer to unseen lemmas or future periods. Full-corpus BM25 indexing and frozen passage encoding include test text without test labels; this is a transductive local setup. Training-only cutoff selection avoids direct test-label tuning but does not undo earlier development-set inspection.

The current checkout contains 0 fitted joblib checkpoints; optimal_thresholds.json and predictions are present. The claim that 45 fitted pipelines are archived cannot be verified here because joblib files are ignored by Git. Part 1's per-model latency, memory and several cost values are hard-coded in its script rather than measured by a profiler, so they cannot support an efficiency ranking. The saved total wall time is a historical run log, not an end-to-end model-training cost. The new analysis manifest records its own environment, elapsed time and input hashes; no model training is repeated.


## 4 Error analysis

### Categories and counting protocol

Define a positive label as a usage expressing the supplied target definition; a negative label is another sense or usage. The following operational taxonomy is declared in configs/part2_analysis.json before aggregation. It partitions hybrid errors using the dense decision on the same instance. The categories are exhaustive and mutually exclusive, and describe observable decisions rather than guessed linguistic causes.

Error definitions, fixed before counting.

| Category | Rule | Meaning |
| --- | --- | --- |
| FN fusion loss | gold=1, hybrid=0, dense=1 | A positive selected by dense retrieval is excluded by hybrid fusion at the saved operating points. |
| FN shared miss | gold=1, hybrid=0, dense=0 | Both systems reject a gold positive; ranks distinguish cutoff rejection from absence in the saved top-100 lists. |
| FP fusion addition | gold=0, hybrid=1, dense=0 | Hybrid fusion introduces a false positive relative to the dense decision. |
| FP shared false match | gold=0, hybrid=1, dense=1 | Both systems accept a usage outside the annotated target sense. |

Seed 42 is the preselected illustration split, with 397 usages and 186 hybrid errors. Across five splits there are 1,985 test appearances per system but only 1,345 unique usages. Hybrid makes 922 erroneous appearances involving 631 distinct usages. Repeated appearances of a usage are not independent cases. The following mean and sample standard deviation summarize counts per split; percentages average each split's fraction of hybrid errors, rather than treating all appearances as an independent sample.

Hybrid error counts and proportions.

| Category | Mean +/- std count | Mean share of errors | Seed 42 count |
| --- | --- | --- | --- |
| FN fusion loss | 15.0 ± 5.1 | 8.2% | 8 |
| FN shared miss | 49.6 ± 4.0 | 26.9% | 51 |
| FP fusion addition | 28.8 ± 5.4 | 15.7% | 37 |
| FP shared false match | 91.0 ± 11.8 | 49.2% | 90 |

![Hybrid errors separated into shared failures and differences introduced by fusion. These are decision categories, not human linguistic annotations.](outputs/figures/07_part2_error_taxonomy.png)

Hybrid errors separated into shared failures and differences introduced by fusion. These are decision categories, not human linguistic annotations.

Confusion counts averaged across five 397-usage test splits. FP and FN use the positive target sense.

| System | Mean TP | Mean TN | Mean FP | Mean FN | Error rate |
| --- | --- | --- | --- | --- | --- |
| BM25 (K*) | 69.4 | 125.6 | 130.4 | 71.6 | 50.9% |
| Dense (K*) | 82.0 | 142.2 | 113.8 | 59.0 | 43.5% |
| Hybrid (K*) | 76.4 | 136.2 | 119.8 | 64.6 | 46.4% |
| Frozen XLM-R + LR | 92.0 | 200.4 | 55.6 | 49.0 | 26.3% |
| Word + character SVM | 94.0 | 237.8 | 18.2 | 47.0 | 16.4% |
| Always positive | 141.0 | 0.0 | 256.0 | 0.0 | 64.5% |

Hybrid averages 119.8 false positives and 64.6 false negatives, versus 113.8 and 59.0 for dense. The fusion-loss and fusion-addition categories identify exactly where replacing the dense decision with hybrid harms a gold label. At the saved K*=100, every hybrid false negative is absent from its saved top-100 list; it is not merely below the binary cutoff inside that list. The unsaved tail is unknown, so 'absent' does not mean absent from the corpus or prove the encoder never retrieved the usage. A larger candidate depth or reranking may address these cases, but that is a future experiment.

Fusion also repairs some dense errors: per split it recovers 9.4 false negatives and removes 22.8 false positives. Those 32.2 repairs are outweighed by 43.8 newly introduced errors (15.0 false negatives and 28.8 false positives), a net 11.6 additional errors per split. The saved transition table verifies this accounting against the confusion totals; it explains the overall error-rate change without assigning speculative linguistic causes.

### Representative errors

Examples use the fixed seed and selection rule in the configuration: prefer different lemmas and contexts nearest 180 characters, then break ties by sentence ID. Two examples are selected from each category. The original Swedish definition and full usage are preserved without normalization or translation. Gold labels are copied from the dataset; these examples do not establish annotation errors. Rank absence always refers to the saved list.

Example 1: FN fusion loss; lemma exportera; sentence 917b2d5_112_122; period 2006-2007.

Target definition: Föra ut en produkt ur ett land (till ett annat land)

Usage: Branschexperter bedömer att omkring fem procent av den råolja som produceras i landet blir stulen för att sedan exporteras av stora syndikat som har kontakter inom militären och regeringen .

Gold target-sense label: 1; hybrid prediction: 0; dense prediction: 1; XLM-R + LR prediction: 1.

Explanation: A positive selected by dense retrieval is excluded by hybrid fusion at the saved operating points. bm25 absent from saved top-100; dense rank 99; hybrid absent from saved top-100. Gold follows the released annotation; no annotation error or linguistic cause is assumed.

Illustrative reading: The usage describes crude oil being exported from a country, consistent with the product-export definition. Dense selects it only at rank 99; its absence from the BM25 top-100 leaves it with one component's support before fusion truncation.

Example 2: FN fusion loss; lemma kvinna; sentence 20a081f7_112_121; period 1904-1907.

Target definition: Generell beteckning för kvinnokönet eller samtliga kvinnor i ett land / samfund (ofta i singularis bestämd form, "kvinnan")

Usage: Det var en tid , som Sveriges folk värkligen kunde betecknas som försupet Men därvid fanns ett stort undantag : kvinnorna söpo ej .

Gold target-sense label: 1; hybrid prediction: 0; dense prediction: 1; XLM-R + LR prediction: 0.

Explanation: A positive selected by dense retrieval is excluded by hybrid fusion at the saved operating points. bm25 absent from saved top-100; dense rank 91; hybrid absent from saved top-100. Gold follows the released annotation; no annotation error or linguistic cause is assumed.

Illustrative reading: The clause 'kvinnorna söpo ej' refers collectively to women, consistent with the definition of women as a group. Dense places it at rank 91; the hybrid output excludes it. XLM-R also misses this example.

Example 3: FN shared miss; lemma fru; sentence 8fe95502_139_142; period 1888-1891.

Target definition: Maka, hustru (oavsett klass)

Usage: Ryske ministern i Stockholm Zinoviev afreste i Lördags kl. 11,45 e. m . med ångfartyget Vasa till Finland för att möta och hit åtfölja sin fru , då hon nu gör sin flyttning till Stockholm ,

Gold target-sense label: 1; hybrid prediction: 0; dense prediction: 0; XLM-R + LR prediction: 1.

Explanation: Both systems reject a gold positive; ranks distinguish cutoff rejection from absence in the saved top-100 lists. bm25 absent from saved top-100; dense absent from saved top-100; hybrid absent from saved top-100. Gold follows the released annotation; no annotation error or linguistic cause is assumed.

Illustrative reading: The phrase 'sin fru' refers to the minister's wife, matching the spouse definition. All three saved retrieval lists omit the usage, while the XLM-R classifier predicts the positive label.

Example 4: FN shared miss; lemma herre; sentence 3dcdcb9d_0_6; period 1880-1883.

Target definition: Manlig överklassperson med betoning på privilegier, högre samhällsstånd; förnäm och mäktig man, högt uppsatt person, potentat, magnat, aristokrat; ståndsperson, herreman

Usage: Herrar Ledamöter behagade sammanträda i Stockholms stads Brandförsäkringskontor , JW 4 Myntlorget , 2 tr. upp , torsdagen den 31 mars kl. 6 e. m . Förvaltningsutskottet en half timme tidigare .

Gold target-sense label: 1; hybrid prediction: 0; dense prediction: 0; XLM-R + LR prediction: 1.

Explanation: Both systems reject a gold positive; ranks distinguish cutoff rejection from absence in the saved top-100 lists. bm25 absent from saved top-100; dense absent from saved top-100; hybrid absent from saved top-100. Gold follows the released annotation; no annotation error or linguistic cause is assumed.

Illustrative reading: The usage addresses members at a formal meeting rather than explicitly naming wealth or social rank. Its positive label illustrates a context-dependent annotation that both retrievers miss; XLM-R predicts the gold label. This reading does not independently verify the addressees' social status.

Example 5: FP fusion addition; lemma motiv; sentence 32f1ee76_100_105; period 2014-2015.

Target definition: Ämne för konstnärlig fram­ställning in­om bild­konst och skönlitteratur

Usage: Bara för ett par dagar sedan inträffade skoldramat i Trollhättan när en maskerad man med rasistiska motiv målmedvetet stegade in i skolan och högg ner människor med annan hudfärg .

Gold target-sense label: 0; hybrid prediction: 1; dense prediction: 0; XLM-R + LR prediction: 0.

Explanation: Hybrid fusion introduces a false positive relative to the dense decision. bm25 rank 14; dense absent from saved top-100; hybrid rank 53. Gold follows the released annotation; no annotation error or linguistic cause is assumed.

Illustrative reading: The phrase 'rasistiska motiv' expresses a reason or motive, whereas the target definition concerns a subject represented in art or literature. The negative gold label is consistent with that distinction. BM25 places it at rank 14 and fusion retains it despite dense excluding it.

Example 6: FP fusion addition; lemma fröken; sentence fa5c2767_154_160; period 1896-1899.

Target definition: Gift eller ogift yrkesutövande kvinna, företrädesvis i skola, butik, postkontor, och bank

Usage: Jag träffade S:t Auban och de Brissac vid mottagningen i Louvre på förmiddagen , och de frågade mig strax , om jag nu var betänkt på att göra min kur för fröken de Grandeourt .

Gold target-sense label: 0; hybrid prediction: 1; dense prediction: 0; XLM-R + LR prediction: 1.

Explanation: Hybrid fusion introduces a false positive relative to the dense decision. bm25 rank 34; dense absent from saved top-100; hybrid rank 87. Gold follows the released annotation; no annotation error or linguistic cause is assumed.

Illustrative reading: Here 'fröken de Grandeourt' is a social title in a courtship context. The supplied target instead concerns an employed woman, especially in teaching or service occupations. BM25 and hybrid accept it; dense rejects it, while XLM-R also predicts a false positive.

Example 7: FP shared false match; lemma dumpa; sentence e07f4b5_67_74; period 2006-2007.

Target definition: Låta något falla; lägga eller sänka ner

Usage: En strandäng i reservatet har enligt kommunen förstörts då Bro Hof dumpade muddermassor över den känsliga floran med älgört , tuvtåtel och bladvass , vilket ej går att återställa .

Gold target-sense label: 0; hybrid prediction: 1; dense prediction: 1; XLM-R + LR prediction: 0.

Explanation: Both systems accept a usage outside the annotated target sense. bm25 absent from saved top-100; dense rank 44; hybrid rank 92. Gold follows the released annotation; no annotation error or linguistic cause is assumed.

Illustrative reading: The usage concerns dumping dredged material onto a meadow. Its physical-action overlap with the broad 'let fall; lay or lower' definition makes this an instructive sense-boundary case, but the released label is negative. Dense and hybrid accept it; XLM-R rejects it. No relabeling is justified by this example alone.

Example 8: FP shared false match; lemma suga; sentence d699d193_79_83; period 2018-2019.

Target definition: Dra in vätska eller gas i munnen genom åstadkommande av undertryck i munhålan eller kraftig inandning; alternativt stoppa någon fast kropp i munnen och genom kraftigt undertryck så småningom smälta och förtära den

Usage: Om magasinen inte fylls på nu av vårregn innan växtligheten slår ut och börjar suga upp vatten kommer flera kommuner runt om i landet att få problem med vattenförsörjningen framöver .

Gold target-sense label: 0; hybrid prediction: 1; dense prediction: 1; XLM-R + LR prediction: 0.

Explanation: Both systems accept a usage outside the annotated target sense. bm25 rank 45; dense rank 90; hybrid rank 58. Gold follows the released annotation; no annotation error or linguistic cause is assumed.

Illustrative reading: Vegetation taking up water shares the idea of suction with the definition but lacks the specified action of a mouth ingesting liquid, gas or a solid object. This is consistent with the negative gold label. All retrievers include it, while XLM-R predicts the negative label.

### Lemmas and context slices

Per-lemma primary F1 averaged within seed, then across seeds. Positive prevalence uses observed test appearances.

| Lemma | Gold positive rate | Hybrid | Dense | XLM-R + LR |
| --- | --- | --- | --- | --- |
| dumpa | 33.6% | 0.455 | 0.391 | 0.492 |
| exportera | 93.1% | 0.762 | 0.788 | 0.910 |
| fru | 91.1% | 0.452 | 0.475 | 0.812 |
| fröken | 4.8% | 0.132 | 0.157 | 0.138 |
| förort | 49.4% | 0.595 | 0.696 | 0.617 |
| herre | 6.6% | 0.146 | 0.158 | 0.135 |
| kvinna | 21.2% | 0.408 | 0.463 | 0.365 |
| motiv | 17.6% | 0.448 | 0.555 | 0.439 |
| skär | 27.6% | 0.431 | 0.431 | 0.555 |
| suga | 21.1% | 0.352 | 0.368 | 0.430 |

The lemma table exposes heterogeneity hidden by the overall score. Dense exceeds hybrid most clearly on motiv and förort, while hybrid exceeds dense on dumpa. Both have low positive-class F1 on fröken and herre. For skär the pool contains 98 usages, so K=100 selects the entire pool for both retrievers: their identical F1 is an all-positive operating-point effect, not evidence that their rankings or sense representations are identical.

Context diagnostics: mean counts and error rates across splits. Each diagnostic is a separate overlapping slice.

| Diagnostic | Present | Mean n | Positive rate | Hybrid error | Dense error |
| --- | --- | --- | --- | --- | --- |
| Before 1950 | yes | 154.6 | 31.7% | 51.2% | 51.3% |
| Before 1950 | no | 242.4 | 38.0% | 43.3% | 38.6% |
| Zero definition overlap | yes | 349.2 | 37.2% | 46.0% | 44.5% |
| Zero definition overlap | no | 47.8 | 23.5% | 49.2% | 36.6% |
| Surface noise marker | yes | 7.2 | 28.3% | 44.5% | 37.4% |
| Surface noise marker | no | 389.8 | 35.6% | 46.5% | 43.7% |

Historical means year<1950 and is a descriptive slice rather than a chronological holdout. Zero definition overlap means no shared exact alphabetic token of length at least three after NFKC normalization, lowercasing and removal of the fixed small Swedish function-word list in the script; no stemming is applied. The surface-noise flag detects a replacement character, angle bracket, or mixed letter/digit token. It is a narrow proxy, not an annotation of OCR quality. Different class prevalence and lemma composition confound these slices; none establishes that date, lexical mismatch or OCR causes an error. Manual categories such as metaphor, historical spelling and genuine sense ambiguity require a blinded linguistic annotation protocol and are not assigned fabricated counts.

![Per-lemma F1 and error rates before/after 1950. Error bars show sample standard deviation across overlapping splits; era class mixtures differ.](outputs/figures/08_part2_word_and_era_errors.png)

Per-lemma F1 and error rates before/after 1950. Error bars show sample standard deviation across overlapping splits; era class mixtures differ.


## 5 Comparison findings and next steps

### Direct comparisons with the baselines

A direct comparison uses identical usage IDs, seed splits and metric definitions. All current Part 1 reruns meet those conditions. Supervised baselines train on 1,586 labeled usages per seed; frozen retrieval uses training labels to choose K but does not train its encoder locally. The table compares outcomes under this shared protocol without treating their label budgets as identical.

Direct local comparison. Uncertainty is sample standard deviation across five splits.

| System | Official F1 mean +/- std | Pooled macro-F1 | Accuracy |
| --- | --- | --- | --- |
| Frozen XLM-R + LR | 0.489 ± 0.022 | 0.715 | 0.737 |
| Always positive | 0.470 ± 0.018 | 0.262 | 0.355 |
| Word + character SVM | 0.450 ± 0.049 | 0.811 | 0.836 |
| Dense (K*) | 0.448 ± 0.016 | 0.554 | 0.565 |
| Hybrid (K*) | 0.418 ± 0.029 | 0.525 | 0.536 |
| BM25 (K*) | 0.358 ± 0.016 | 0.481 | 0.491 |

The strongest observed primary result is frozen XLM-R + LR at 0.489 ± 0.022. The always-positive baseline reaches 0.470 ± 0.018; XLM-R's mean advantage is only +0.019, with no exact seed-level rejection at 0.05. Hybrid (0.418) and dense (0.448) both trail this trivial baseline on the primary metric, while giving better pooled macro-F1. The high always-positive score is consistent with a positive-class metric and is not itself evidence of a scoring bug. Reporting only accuracy or only the primary F1 would conceal this tradeoff.

Retrieval comparison on the same ten definition queries. MAP and MRR are evaluated on saved lists truncated at 100.

| System | MRR | MAP | nDCG@10 | Recall@50 |
| --- | --- | --- | --- | --- |
| BM25 | 0.618 | 0.229 | 0.383 | 0.285 |
| DENSE | 0.612 | 0.323 | 0.427 | 0.389 |
| HYBRID | 0.692 | 0.293 | 0.490 | 0.381 |

Hybrid improves early ranking over dense (MRR 0.6917 versus 0.6117; nDCG@10 0.4902 versus 0.4274), but dense has higher MAP (0.3229 versus 0.2926) and binary official F1 (0.4484 versus 0.4182). MRR rewards the first relevant usage and nDCG@10 the leading ranks; they do not guarantee better coverage of all positives or a better thresholded classifier. The actual macro Recall@50 values are BM25 0.2847, dense 0.3889 and hybrid 0.3812, correcting the larger values stated in the earlier prose. Each remains a local development ranking result; no hidden-test ranking outcome is available.

![Different metrics favor different systems: early ranking favors hybrid, while MAP and binary F1 favor dense; the strongest primary classifier is XLM-R + LR.](outputs/figures/09_part2_metric_tradeoffs.png)

Different metrics favor different systems: early ranking favors hybrid, while MAP and binary F1 favor dense; the strongest primary classifier is XLM-R + LR.

### Published work and comparability

Direct means matching data, split and metric; approximate means partial protocol alignment that prevents a strict improvement claim; non-comparable means that the task, evaluation unit, relevance judgments or metric differs enough to rule out a numerical performance comparison. Archived Assignment 1 aggregate scores are approximate to the current multi-seed summary: they use one seed and foreground pooled macro-F1. The current reruns, rather than those old aggregates, supply the direct baseline comparisons.

Published comparisons. No directly matched published test result is available in these sources.

| Published work | Label | Reason |
| --- | --- | --- |
| Cormack et al. 2009 | non-comparable | Different corpora, queries, relevance judgments and ranking depths. Shared fusion formula permits a methodological comparison only; published MAP cannot be subtracted from local MAP. Fusion is not guaranteed to improve every metric. |
| Wang et al. 2024 | non-comparable | MIRACL passage retrieval has different languages, supervision, query formulation and candidate pools from historical Swedish definition matching. The paper supports the encoder choice, not a numerical improvement claim here. |
| Conneau et al. 2020 | non-comparable | Different tasks, datasets, training and metrics. A frozen sentence representation with a local classifier is not the paper's benchmark setup; it also does not supply the missing XLM-R retrieval ablation. |
| Schlechtweg et al. 2020 | non-comparable | The Swedish language and diachronic setting are related, but prediction units, labels, data partitions and metrics differ. The 2020 task cannot serve as a directly comparable 2027 leaderboard. |

[Cormack et al. 2009: Reciprocal rank fusion outperforms Condorcet and individual rank learning methods](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf). TREC Robust MAP: RRF 0.3686; best individual 0.3586 (Table 2). RRF MAP 0.2926 versus dense 0.3229; RRF MRR and nDCG@10 are higher locally. These scores are not subtracted across benchmarks.

[Wang et al. 2024: Multilingual E5 Text Embeddings: A Technical Report](https://arxiv.org/html/2402.05672v1). MIRACL development, average over 16 languages: mE5-small nDCG@10 60.8 and R@100 92.4, on a 0-100 scale (Table 4). E5-small nDCG@10 0.4274 on ten Swedish definition queries. These scores are not subtracted across benchmarks.

[Conneau et al. 2020: Unsupervised Cross-lingual Representation Learning at Scale](https://aclanthology.org/2020.acl-main.747/). Cross-lingual classification, question answering and named entity recognition benchmarks. Frozen XLM-R plus supervised logistic regression: local official F1 0.4892. These scores are not subtracted across benchmarks.

[Schlechtweg et al. 2020: SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection](https://aclanthology.org/2020.semeval-1.1/). Word-level change classification uses accuracy; graded change ranking uses Spearman correlation (Section 5.1). Usage-level binary definition matching, averaged positive-class F1 over lemmas. These scores are not subtracted across benchmarks.

The comparisons support three methodological connections: rank fusion can improve leading hits without improving all metrics; retrieval-oriented multilingual embeddings motivate E5, but do not establish superiority on this historical corpus; and word-level change detection in SemEval-2020 has a different prediction target from usage-level definition matching. The component methods are established techniques; the local contribution is their application and evaluation, not the invention of BM25, multilingual encoding or RRF. Sources and table locations are recorded in configs/published_comparisons.json (checked 21 November 2026).

### Findings limitations and next steps

Strongest finding: the best observed local primary score belongs to the frozen XLM-R supervised classifier; the proposed hybrid offers a narrower benefit in early retrieval. Negative result: hybrid does not beat dense on MAP or binary F1, and neither retriever beats always-positive on official F1. Unexpected result: removing explicit word ID helps official F1 but hurts pooled F1 and accuracy. The intermediate RRF smoothing hypothesis is metric-dependent, and the backbone hypothesis remains untested.

The evidence is limited to 1,983 Swedish development usages, ten lemmas, ten definitions and five overlapping random splits. There is no held-out language, source-document split, future-period test or unseen-lemma evaluation. Development-set reuse, the fixed top-100 tail, small-sample significance resolution, missing fitted checkpoints and unmeasured per-model cost limit broader claims. Automatic operational categories are reproducible but do not replace expert sense/error annotation.

1. Separate model selection from final assessment: reserve an untouched set, group near-duplicate/source documents where provenance permits, and add lemma-held-out and chronological evaluations. Fit vocabularies and any inductive corpus statistics using training data only.

2. Complete the matched E5 versus XLM-R retrieval ablation with identical definitions, pooling, candidate scope and depth. Report different pretraining, dimensions and task prefixes explicitly rather than attributing every difference to contrastive learning alone.

3. Regenerate deeper component rankings and tune cutoffs or score thresholds on validation data beyond the present K=100 boundary. Compare against always-positive, optimize the declared primary objective and retain secondary metrics to expose tradeoffs.

4. Test whether fusion-loss cases improve with validation-selected weighting, deeper candidates or reranking. Evaluate the final frozen choice once on untouched data; do not select a new k using the reported evaluation queries.

5. Blindly double-annotate a stratified sample for historical spelling/OCR, metaphor, sense ambiguity and definition mismatch; report agreement and prevalence before making causal linguistic claims. Archive fitted models and measure encoding, indexing, inference time, peak memory and resource cost separately.

### Reproducible outputs

Run python Assignment_3/scripts/run_part2_analysis.py from the repository root. The run validates saved predictions, reconstructs splits and RRF variants, produces tables 8-12 and supporting CSVs, writes annotated_test_usages.csv, saves four figures as PNG and PDF, and regenerates this report, the findings summary and LaTeX Sections 3-5. The analysis manifest records input SHA-256 hashes, configuration, environment, verification checks and analysis elapsed time. Models and the fixed evaluation functions are unchanged.
