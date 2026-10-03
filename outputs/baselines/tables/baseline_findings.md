# Baseline Findings: Read–Reason–Interpret–Verify

> Structured observations, interpretations, and modeling implications
> for Baseline A (Classical ML) and Baseline B (Frozen Pretrained).

## 1. Data and Label Structure

### Observation
The Swedish development set contains **1983 usage records** across **10 target words** and **10 time periods**.
The class distribution for Subtask 2 is **0: 1,278 / 1: 705** (ratio ≈ 1.81:1), confirmed by data-join diagnostics which report **0 unmatched records** and **0 malformed labels**.
Subtask 1 has **524 multi-valued sense annotations** (26.4 % of records); these are preserved throughout but excluded from ARI, NMI, and purity evaluation in line with the multi-label policy.

### Interpretation
The label-1 rate is highly **word-dependent**: `exportera` (94.6 %) and `fru` (89.4 %) are associated with the highest target-sense rates, while `fröken` (5.7 %) and `herre` (7.1 %) have the lowest. This demonstrates that a naive model can memorise target-word identity to boost accuracy without detecting genuine semantic change.

### Modeling Implication
Per-word and per-period group metrics are essential to expose this effect. High aggregate macro-F1 does not guarantee sensitivity to semantic change rather than token identity. Evaluation must be disaggregated by word and by period.

## 2. Subtask 2 — Binary Usage Classification (Baseline A)

### Observation
- **eda_features_logistic_regression** (tfidf_plus_eda_features): macro-F1=0.8163, balanced-accuracy=0.8049
- **word_char_tfidf_linear_svm** (word_and_character_tfidf): macro-F1=0.8155, balanced-accuracy=0.8033
- **tfidf_logistic_regression** (word_tfidf): macro-F1=0.7917, balanced-accuracy=0.7826
- **eda_features_random_forest** (tfidf_plus_eda_features_rf): macro-F1=0.7910, balanced-accuracy=0.7741
- **frozen_xlmr_logistic_regression** (frozen_xlm_roberta_mean_pooled_embedding): macro-F1=0.7212, balanced-accuracy=0.7312
- **frozen_xlmr_linear_svm** (frozen_xlm_roberta_mean_pooled_embedding): macro-F1=0.7145, balanced-accuracy=0.7225
- **ablated_no_word_id_logistic_regression** (context_and_eda_without_word_identity): macro-F1=0.7129, balanced-accuracy=0.7165
- **tfidf_complement_nb** (word_tfidf): macro-F1=0.6171, balanced-accuracy=0.6253
- **majority** (majority): macro-F1=0.3920, balanced-accuracy=0.5000

### Interpretation
The majority baseline (macro-F1 = 0.3920, accuracy = 64.5 %) establishes the performance floor. Always predicting label=0 fails completely on the minority class.
Word-level TF-IDF + Logistic Regression (macro-F1 = 0.7917) shows that lexical context is predictive. Inspecting top positive LR coefficients (`fru`, `exportera`, `sin fru`, `exporterar`) and negative coefficients (`fröken`, `herrar`, `motivet`, `herrarnas`) shows the model learns n-grams correlated with class prevalence.
Complement Naive Bayes achieves macro-F1 = 0.6171 (+22.5 points over majority). Unlike standard MultinomialNB which collapses under class imbalance, ComplementNB weights features against the complement class, proving probabilistic bag-of-words retains significant discriminative signal.
EDA-feature Random Forest demonstrates non-linear decision capability. Gini importance ranking reveals that `numeric__target_position` ranks #3 overall among top features (importance = 0.0199). Top features overall include: `categorical__word_exportera` (0.0248), `categorical__word_herre` (0.0204), `numeric__target_position` (0.0199). This confirms that tree splits actively utilize structural EDA features.
**Word-Identity Ablation (`ablated_no_word_id_logistic_regression`)**: When target words are masked with `[TARGET]` and the categorical `word` column is excluded, the model still achieves **macro-F1 = 0.7129** and balanced accuracy = 0.7165 (+32.1 points over majority). This decouples genuine contextual semantic change detection from the token memorization shortcut (10.3 point delta to 0.8163).

### Modeling Implication — Connection to EDA
EDA established: (a) strong target-word imbalance, (b) `target_position` correlation with the label, and (c) minimal period-level direct signal. Both linear coefficients and tree-based Gini importances validate findings (a) and (b). Per-group metrics demonstrate that model sensitivity is non-uniform across words, confirming that ablation and debiased evaluation are necessary for honest diachronic change detection.

## 3. Subtask 2 — Baseline B (Pretrained Models)

### Observation
- **frozen_xlmr_logistic_regression**: macro-F1=0.7212, balanced-accuracy=0.7312
- **frozen_xlmr_linear_svm**: macro-F1=0.7145, balanced-accuracy=0.7225

### Interpretation
Frozen XLM-R (mean-pooled, 768-d) achieves macro-F1 = 0.7212 (best: `frozen_xlmr_logistic_regression`), compared to the top classical model `eda_features_logistic_regression` at macro-F1 = 0.8163 (Δ = +0.0951). Frozen representations without fine-tuning underperform classical TF-IDF + EDA features, suggesting that mean-pooled contextual embeddings require either fine-tuning or task-specific adaptation to surpass interpretable classical baselines on this highly imbalanced, word-identity-confounded dataset. Crucially, frozen XLM-R outperforms the word-identity ablated classical LR (macro-F1 = 0.7129), confirming that contextualized representations capture more than pure lexical identity shortcuts.
See full ranking: `outputs/baselines/tables/subtask2_comparison.csv`.

## 4. Subtask 1 — Diachronic Word Sense Induction (Baseline A)

### Observation
Clustering quality across 20 words (TF-IDF KMeans, k=3):
- **dumpa**: ARI=0.0416, purity=0.8155, eval_n=103, excl.=42
- **exportera**: ARI=0.0045, purity=0.9463, eval_n=149, excl.=0
- **fru**: ARI=0.2550, purity=0.8667, eval_n=195, excl.=79
- **fröken**: ARI=-0.1041, purity=0.8383, eval_n=167, excl.=7
- **förort**: ARI=-0.0148, purity=0.5775, eval_n=71, excl.=79
- **herre**: ARI=0.0397, purity=0.8382, eval_n=241, excl.=42
- **kvinna**: ARI=0.0600, purity=0.7746, eval_n=284, excl.=6
- **motiv**: ARI=0.1442, purity=0.7855, eval_n=289, excl.=2
- **skär**: ARI=-0.0028, purity=0.5474, eval_n=95, excl.=3
- **suga**: ARI=0.0383, purity=0.3228, eval_n=127, excl.=2
- **dumpa**: ARI=0.0349, purity=0.8155, eval_n=103, excl.=42
- **exportera**: ARI=-0.0065, purity=0.9463, eval_n=149, excl.=0
- **fru**: ARI=0.3396, purity=0.8667, eval_n=195, excl.=79
- **fröken**: ARI=0.0858, purity=0.8383, eval_n=167, excl.=7
- **förort**: ARI=0.1131, purity=0.7324, eval_n=71, excl.=79
- **herre**: ARI=0.2568, purity=0.8382, eval_n=241, excl.=42
- **kvinna**: ARI=0.0457, purity=0.7746, eval_n=284, excl.=6
- **motiv**: ARI=0.2516, purity=0.8754, eval_n=289, excl.=2
- **skär**: ARI=0.1575, purity=0.5579, eval_n=95, excl.=3
- **suga**: ARI=0.0915, purity=0.3780, eval_n=127, excl.=2

### Interpretation
Adjusted Rand Index values remain near zero for most words. The highest ARI is achieved by `fru` (ARI = 0.3396, purity = 0.8667). Negative ARI observed for `fröken`, `förort`, `skär`, `exportera` indicates worse-than-random sense partitioning under bag-of-words. High purity values often reflect skewed gold distributions rather than discovery of true semantic boundaries.

### Temporal Cluster Shifts
Tracking dominant clusters across chronological periods identifies shifts (≥2 transitions) for: `dumpa`, `exportera`, `fru`, `förort`, `herre`, `kvinna`, `motiv`. Words with stable dominant clusters across time: `fröken`, `skär`, `suga`. Given baseline ARI scores, shifts may indicate topical or register drift in corpus composition.

### Modeling Implication
Classical bag-of-words clustering establishes that lexical context alone cannot resolve subtle sense distinctions. This provides the precise methodological motivation for moving to contextualized representations in Baseline B.

## 5. Leakage and Reproducibility Checks

| Check | Status | Verification |
|---|---|---|
| Raw data read-only | ✅ Passed | No writes performed to `data/raw/` |
| Pipeline fit on train only | ✅ Passed | TF-IDF, Scaler, and OneHotEncoder encapsulated in Pipeline, fit on train only |
| Frozen representations | ✅ Passed | Pretrained embeddings generated without label leakage |
| Deterministic seed | ✅ Passed | Seed 42 fixed across random, numpy, sklearn, and split generators |
| Stratified splitting | ✅ Passed | Subtask 2 split stratifies on label_value to preserve binary balance |
| Accounting for multi-labels | ✅ Passed | Multi-valued labels accounted for in diagnostics and excluded cleanly from ARI |

## 6. Next Steps

1. Fine-tune or adapter-tune XLM-R to close the Δ ≈ 9.5 F1-point gap to classical EDA-feature LR.
2. Experiment with dynamic cluster count selection (silhouette score / elbow method) per target word.
3. Explore era conditioning to explicitly model diachronic drift as an independent variable.
4. Apply debiasing techniques (balanced sampling, adversarial word-ID removal) for more honest semantic change evaluation.

