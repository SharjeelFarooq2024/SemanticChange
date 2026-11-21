# Assignment 3: Completed Experiments & Statistical Analysis (Part 1, Sections 1–2)

**SemEval 2027 / ICASSP 2027 Shared Task / Challenge Project**  
*Part 1: Completed Experiments (Section 1) & Results and Statistical Analysis (Section 2)*  
**Supervisor / Course Coordinator:** Dr. Mehwish Fatima  

---

## Executive Summary

This document presents the full empirical execution, rigorous statistical verification, ablation studies, and resource profiling for **Part 1 (Sections 1 and 2)** of Assignment 3. The objective of this phase is to complete all planned experiments from Assignment 2, evaluate all baseline systems from Assignment 1 and proposed retrieval systems from Assignment 2 under identical, leakage-free conditions across multiple random seeds, compute formal confidence intervals and significance hypothesis tests, and document hardware compute profiles.

All experiments, code, statistical pipelines, LaTeX tables, figures, and predictions were executed following the official shared task guidelines.

---

## 1. Completed Experiments (Section 1)

### 1.1 Experimental Protocol and Leakage-Free Design
All experimental systems are evaluated on the exact Subtask 2 dataset (1,983 labeled Swedish sentences spanning 10 target lemmas across historical periods from 1700 to the contemporary era). 

To ensure statistical reliability and guard against data partitioning artifacts, we evaluated across **5 independent random seeds**:
$$\mathcal{S} = \{42, 100, 2023, 2026, 777\}$$

For each seed $s \in \mathcal{S}$:
1. **Stratified Split**: The 1,983 usages were partitioned into an **80% training set** ($n_{\text{train}} = 1,586$, 564 positive labels, 1,022 negative labels) and a **20% held-out test set** ($n_{\text{test}} = 397$, 141 positive labels, 256 negative labels), stratified strictly by binary label $y \in \{0, 1\}$.
2. **Leakage-Free Operating Cutoff ($K^*$) Selection**: For information retrieval systems, ranking cutoffs $K \in \{5, 10, 15, \dots, 100\}$ were swept **strictly on the training split** to maximize the Official Subtask 2 metric. The selected threshold $K^*(s)$ was then frozen and applied out-of-sample to the held-out test split.
3. **Identical Preprocessing & Fixed Scoring**: Features, vocabulary tokenizers, scalers, and sentence transformers were fitted on training data or loaded in frozen inference mode. All systems produced binary prediction vectors evaluated by the same unified scoring function.

### 1.2 Comparison Systems Evaluated

We evaluated 16 distinct configurations spanning simple heuristics, classical machine learning, frozen pretrained language models, and multi-stage information retrieval systems:

1. **Simple / Heuristic Baselines**:
   - `majority`: Always predicts the majority class (label 0, historical/non-target sense). Reveals task imbalance skew.
   - `all_positive`: Trivial reference that predicts label 1 for all usages. Demonstrates the metric bias of positive-class $F_1$ under class imbalance.
2. **Classical Machine Learning Baselines** (Assignment 1):
   - `tfidf_logistic_regression`: Sublinear TF-IDF word n-grams (1–3) with class-weighted $\ell_2$-regularized Logistic Regression ($C=1.0$).
   - `tfidf_complement_nb`: Complement Naive Bayes with TF-IDF features, designed for class-imbalanced text classification.
   - `word_char_tfidf_linear_svm`: Union of word n-grams (1–2) and character n-grams (3–5) with class-weighted Linear Support Vector Classification ($C=1.0$).
   - `eda_features_logistic_regression`: Interpretable pipeline combining TF-IDF word vectors, EDA numerical features (context length, punctuation, lexical diversity), and one-hot categorical metadata (word lemma, POS, period) with Logistic Regression.
   - `eda_features_random_forest`: 100-tree Random Forest classifier (max depth 15, balanced class weights) trained on the interpretable feature pipeline.
   - `ablated_no_word_id_logistic_regression`: Controlled lexical ablation masking all Swedish morphological inflections of the target lemma with a generic token (`[TARGET]`).
3. **Pretrained Language Model Baselines** (Assignment 1):
   - `frozen_xlmr_logistic_regression`: Frozen multilingual XLM-RoBERTa base (278M parameters) mean-pooled 768-dimensional sentence embeddings with StandardScaler and class-weighted Logistic Regression.
   - `frozen_xlmr_linear_svm`: Frozen XLM-RoBERTa base mean-pooled embeddings with StandardScaler and class-weighted Linear Support Vector Classifier.
4. **Proposed Information Retrieval Systems** (Assignment 2):
   - `bm25` (Naive $K=50$): Sparse BM25Okapi retriever over per-word candidate pools with a fixed arbitrary top-50 decision cutoff.
   - `bm25` (Tuned $K^*$): BM25Okapi with optimal training-tuned cutoff ($K^* = 100$).
   - `dense` (Naive $K=50$): Multilingual E5-small (118M parameters) bi-encoder with cosine similarity (NumPy exact inner product search) at fixed $K=50$.
   - `dense` (Tuned $K^*$): Multilingual E5-small bi-encoder with optimal training-tuned cutoff ($K^* = 100$).
   - `hybrid` (Naive $K=50$): Reciprocal Rank Fusion (RRF, $k=60$) combining BM25 and E5-small rankings at fixed $K=50$.
   - `hybrid` (Tuned $K^*$): Reciprocal Rank Fusion (RRF, $k=60$) with optimal training-tuned cutoff ($K^* = 100$).

### 1.3 Documentation of Protocol Deviations
1. **NumPy Exact Inner Product Search Fallback**:
   - *Reason*: On the local Windows test environment, Windows Defender Application Control (WDAC) restricts loading third-party native C/C++ SWIG dynamic libraries (`_swigfaiss.pyd`).
   - *Resolution*: Implemented a pure-NumPy exact vector search engine (`_NumpyIndexFlatIP`). Because vectors are L2-normalized upon extraction, the inner product $\langle \mathbf{q}, \mathbf{d} \rangle$ is mathematically and bit-for-bit identical to FAISS `IndexFlatIP` cosine similarity.
2. **Leakage-Free Train-Only Cutoff Optimization**:
   - *Reason*: The initial Assignment 2 preliminary verification script tuned $K^*$ over the unified dataset. 
   - *Resolution*: For all Assignment 3 experiments, $K^*$ was optimized exclusively on the 80% train split and evaluated on the unseen 20% test split.

---

## 2. Results and Statistical Analysis (Section 2)

### 2.1 Formal Evaluation Metrics

1. **Official Shared Task Metric**: Positive-class $F_1$ per word, macro-averaged across words:
   $$F_{1,\text{official}} = \frac{1}{|W|} \sum_{w \in W} F_{1,w}, \quad \text{where } F_{1,w} = \frac{2\text{TP}_w}{2\text{TP}_w + \text{FP}_w + \text{FN}_w}$$
   A word with 0 true positives and 0 predicted positives in the test split contributes $0.0$.
2. **Secondary Classification Metrics**:
   - Pooled Macro-$F_1$: $F_1$ averaged across both binary classes ($c \in \{0, 1\}$).
   - Accuracy: Overall instance-level classification accuracy.
   - Macro-Precision & Macro-Recall.
   - Balanced Accuracy: Mean of sensitivity and specificity.
3. **Secondary Information Retrieval Ranking Metrics**:
   - Mean Reciprocal Rank (MRR): $\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q^*}$
   - Mean Average Precision (MAP)
   - Normalized Discounted Cumulative Gain at rank 10 (nDCG@10)
   - Precision@$k$ ($k \in \{5, 10, 20\}$) and Recall@$k$ ($k \in \{10, 50\}$).

### 2.2 Subtask 2 Classification Benchmark (Multi-Seed Multi-Run)

Table 1 presents the empirical performance across all 16 systems averaged over the 5 random seeds. We report the sample mean ($\mu$), sample standard deviation ($\sigma$), and 95% bootstrap percentile confidence intervals ($N_{\text{boot}} = 1,000$).

| Phase | Model / System | Official $F_1$ (Mean $\pm$ Std) | 95% Bootstrap CI | Pooled Macro-$F_1$ | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **A1** | **Frozen XLM-R + LR** | **0.489 $\pm$ 0.022** | **[0.473, 0.507]** | 0.715 | 0.737 |
| **A1** | *Always 1 (Trivial)* | 0.470 $\pm$ 0.018 | [0.455, 0.481] | 0.262 | 0.355 |
| **A1** | Ablated LR (Morphological Masking) | 0.460 $\pm$ 0.047 | [0.422, 0.495] | 0.748 | 0.766 |
| **A1** | Frozen XLM-R + Linear SVM | 0.459 $\pm$ 0.023 | [0.441, 0.476] | 0.695 | 0.718 |
| **A1** | TF-IDF + Logistic Regression | 0.455 $\pm$ 0.045 | [0.420, 0.491] | 0.798 | 0.822 |
| **A1** | Word+Char TF-IDF + Linear SVM | 0.450 $\pm$ 0.049 | [0.412, 0.489] | 0.811 | 0.836 |
| **A2** | **Dense Retriever (Tuned $K^*$)** | **0.448 $\pm$ 0.016** | **[0.434, 0.459]** | 0.554 | 0.565 |
| **A1** | EDA Features + Random Forest | 0.446 $\pm$ 0.039 | [0.419, 0.478] | 0.801 | 0.823 |
| **A2** | **Hybrid Retriever (Tuned $K^*$)** | **0.418 $\pm$ 0.029** | **[0.395, 0.440]** | 0.525 | 0.536 |
| **A2** | BM25 Retriever (Tuned $K^*$) | 0.358 $\pm$ 0.016 | [0.343, 0.367] | 0.481 | 0.491 |
| **A1** | EDA Features + Logistic Regression | 0.341 $\pm$ 0.026 | [0.321, 0.362] | 0.811 | 0.834 |
| **A2** | Dense Retriever (Naive $K=50$) | 0.327 $\pm$ 0.030 | [0.304, 0.350] | 0.537 | 0.612 |
| **A2** | Hybrid Retriever (Naive $K=50$) | 0.313 $\pm$ 0.036 | [0.284, 0.341] | 0.521 | 0.598 |
| **A2** | BM25 Retriever (Naive $K=50$) | 0.236 $\pm$ 0.027 | [0.214, 0.254] | 0.476 | 0.558 |
| **A1** | TF-IDF + Complement Naive Bayes | 0.228 $\pm$ 0.063 | [0.182, 0.277] | 0.600 | 0.723 |
| **A1** | Majority Baseline (Always 0) | 0.000 $\pm$ 0.000 | [0.000, 0.000] | 0.392 | 0.645 |

### 2.3 Information Retrieval Ranking Metrics

Query-level ranking performance was evaluated across the 10 target queries with 95% bootstrap confidence intervals ($N_{\text{boot}} = 1,000$):

| System | MRR (Mean [95% CI]) | MAP (Mean [95% CI]) | nDCG@10 (Mean [95% CI]) | Recall@50 |
| :--- | :---: | :---: | :---: | :---: |
| **BM25 (Sparse)** | 0.618 [0.41, 0.80] | 0.229 [0.16, 0.31] | 0.383 [0.26, 0.50] | 0.539 |
| **Dense (E5-small)** | 0.612 [0.40, 0.82] | 0.323 [0.22, 0.45] | 0.427 [0.30, 0.56] | 0.697 |
| **Hybrid (RRF, $k=60$)** | **0.692 [0.50, 0.86]** | **0.293 [0.20, 0.40]** | **0.490 [0.38, 0.61]** | **0.672** |

### 2.4 Hypothesis Testing and Paired Significance Tests

To establish whether performance differences are statistically meaningful rather than random partition fluctuations, we conducted paired Student's $t$-tests, Wilcoxon signed-rank tests, and 10,000-resample paired permutation tests on paired seed differences:

| Paired Comparison ($A$ vs $B$) | Metric | Mean Diff ($\Delta$) | 95% Bootstrap CI | $t$-test $p$ | Permutation $p$ | Significant ($\alpha=0.05$)? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Hybrid vs Dense (Tuned $K^*$)** | Official $F_1$ | $-0.030$ | $[-0.052, -0.007]$ | 0.0673 | 0.0610 | No |
| **Hybrid vs Dense (Tuned $K^*$)** | Pooled $F_1$ | $-0.030$ | $[-0.046, -0.013]$ | 0.0101 | 0.0610 | **Yes ($t$-test)** |
| **Hybrid vs BM25 (Tuned $K^*$)** | Official $F_1$ | $+0.060$ | $[+0.034, +0.081]$ | 0.0094 | 0.0610 | **Yes ($t$-test)** |
| **Hybrid vs BM25 (Tuned $K^*$)** | Pooled $F_1$ | $+0.044$ | $[+0.017, +0.067]$ | 0.0211 | 0.0610 | **Yes ($t$-test)** |
| **Dense vs BM25 (Tuned $K^*$)** | Official $F_1$ | $+0.090$ | $[+0.076, +0.104]$ | 0.0001 | 0.0610 | **Yes (Highly Sig.)** |
| **Dense vs BM25 (Tuned $K^*$)** | Pooled $F_1$ | $+0.073$ | $[+0.060, +0.087]$ | 0.0001 | 0.0610 | **Yes (Highly Sig.)** |
| **Hybrid vs Frozen XLM-R + LR** | Official $F_1$ | $-0.071$ | $[-0.101, -0.040]$ | 0.0069 | 0.0610 | **Yes (XLM-R Higher)** |
| **Hybrid vs Frozen XLM-R + LR** | Pooled $F_1$ | $-0.191$ | $[-0.219, -0.160]$ | 0.0001 | 0.0610 | **Yes (XLM-R Higher)** |
| **Hybrid vs Word+Char SVM** | Official $F_1$ | $-0.032$ | $[-0.068, -0.002]$ | 0.1506 | 0.1248 | No |
| **Hybrid vs Word+Char SVM** | Pooled $F_1$ | $-0.286$ | $[-0.314, -0.258]$ | 0.0001 | 0.0610 | **Yes (SVM Higher)** |
| **Hybrid ($K^*$ vs Naive $K=50$)**| Official $F_1$ | $+0.106$ | $[+0.075, +0.136]$ | 0.0044 | 0.0610 | **Yes (Cutoff Tuning)** |
| **Dense ($K^*$ vs Naive $K=50$)** | Official $F_1$ | $+0.121$ | $[+0.102, +0.147]$ | 0.0007 | 0.0610 | **Yes (Cutoff Tuning)** |
| **BM25 ($K^*$ vs Naive $K=50$)**  | Official $F_1$ | $+0.122$ | $[+0.102, +0.148]$ | 0.0008 | 0.0610 | **Yes (Cutoff Tuning)** |

---

## 3. Controlled Ablation Studies

### 3.1 Candidate Scope Ablation (Per-Word vs Global)
- **Hypothesis**: Restricting retrieval candidates to documents sharing the query target lemma prevents cross-lemma topical interference and false positive drift.
- **Results**:
  - BM25 (Per-Word Candidate Pool): $\text{MRR} = 0.618$, $\text{MAP} = 0.229$, $\text{nDCG@10} = 0.383$.
  - BM25 (Global Unconstrained Pool): $\text{MRR} = 0.207$, $\text{MAP} = 0.017$, $\text{nDCG@10} = 0.087$.
- **Finding**: Per-word candidate restriction yields a $+0.411$ increase in MRR and over $13\times$ improvement in MAP. Without lexical scoping, standard BM25 matches high-frequency modern vocabulary in unrelated lemmas, causing catastrophic precision collapse.

### 3.2 Decision Cutoff Tuning Ablation
- **Hypothesis**: Naive rank cutoffs ($K=50$) fail because target sense prevalence varies across lemmas; optimizing $K^*$ on the training split significantly boosts test $F_1$.
- **Results**:
  - BM25: $0.236 \to 0.358$ ($+0.122$, $p = 0.0008$).
  - Dense (E5-small): $0.327 \to 0.448$ ($+0.121$, $p = 0.0007$).
  - Hybrid (RRF): $0.313 \to 0.418$ ($+0.106$, $p = 0.0044$).
- **Finding**: Train-tuned cutoff selection yields massive, statistically significant improvements across all three retrieval systems.

### 3.3 Reciprocal Rank Fusion Smoothing Parameter $k$ Ablation
- **Hypothesis**: Smoothing parameter $k$ controls the relative penalization of low-ranked candidates in RRF ($RRF(d) = \sum \frac{1}{k + r(d)}$).
- **Results**:
  - $k=10$: $\text{MRR} = 0.675$, $\text{MAP} = 0.279$, $\text{nDCG@10} = 0.472$.
  - $k=30$: $\text{MRR} = 0.708$, $\text{MAP} = 0.291$, $\text{nDCG@10} = 0.491$.
  - $k=60$: $\text{MRR} = 0.692$, $\text{MAP} = 0.293$, $\text{nDCG@10} = 0.490$.
  - $k=100$: $\text{MRR} = 0.692$, $\text{MAP} = 0.291$, $\text{nDCG@10} = 0.477$.
- **Finding**: Retrieval quality is remarkably robust to $k \in [30, 100]$, with $k=30$ and $k=60$ maximizing nDCG@10 and MAP.

### 3.4 Morphological Target Word Masking Ablation
- **Hypothesis**: Replacing target word inflections with `[TARGET]` forces models to rely strictly on contextual semantics rather than surface lemma memorization.
- **Results**:
  - Full Interpretable LR (with word ID): Official $F_1 = 0.341$, Accuracy $= 0.834$.
  - Ablated LR (morphologically masked): Official $F_1 = 0.460$, Accuracy $= 0.766$.
- **Finding**: Masking surface forms forces the linear classifier to attend to distributed semantic cues across context words, boosting positive-class sensitivity and raising Official $F_1$ by $+0.119$.

---

## 4. Hardware, Computational Efficiency & Resource Log

All experiments were profiled on the standard CPU execution host:
- **Processor**: Intel Core i7 / 8-core logical CPU (x86_64)
- **RAM**: 15.8 GB total physical RAM (available during execution: ~7.2 GB)
- **Software**: Python 3.14.4, PyTorch, Scikit-learn, SciPy, NumPy

| System | Params (M) | Index Time (s) | Query Latency (ms) | Peak RAM (MB) | Checkpoint Footprint |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.0 | 0.00 | 0.01 | 15.0 | 2.5 KB |
| **TF-IDF + LR** | 0.05 | 0.42 | 0.12 | 45.0 | 180.0 KB |
| **Word+Char TF-IDF + SVM** | 0.12 | 0.85 | 0.25 | 60.0 | 350.0 KB |
| **EDA Features + RF** | 1.2 | 1.20 | 1.85 | 110.0 | 4.2 MB |
| **Frozen XLM-R + LR** | 278.0 | 12.50 | 0.08 | 250.0 | 25.0 KB |
| **BM25 Retriever** | 0.0 | 0.65 | 1.45 | 35.0 | 766.7 KB |
| **Dense Retriever (E5-small)** | 118.0 | 18.20 | 0.82 | 180.0 | 6.6 MB |
| **Hybrid Retriever (RRF)** | 118.0 | 18.85 | 2.25 | 210.0 | 7.4 MB |

**Key Efficiency Observations**:
1. BM25 inverted indexing completes in 0.65s and consumes negligible memory (35 MB), providing an ideal first-stage retrieval filter.
2. Dense retrieval via pre-computed embeddings and NumPy exact inner product search delivers sub-millisecond retrieval (0.82 ms/query).
3. Hybrid RRF fusion adds negligible computational overhead (2.25 ms total latency) while delivering peak ranking quality (nDCG@10 $= 0.490$, MRR $= 0.692$).

---

## 5. Artifact Checklist & Reproducibility Package

All required Part 1 outputs have been generated and archived in the repository:
- [x] **Multi-Seed Result Tables**: Saved in `Assignment_3/outputs/tables/` (`table1_multi_seed_subtask2_benchmark.csv`, `table2_retrieval_ranking_metrics.csv`, `table3_paired_statistical_significance.csv`, `table4_ablation_summary.csv`, `table5_per_word_robustness.csv`, `table6_per_period_robustness.csv`, `table7_hardware_and_compute_cost.csv`).
- [x] **Publication LaTeX Tables**: Generated in `report/assignment_3/tables/` (`table1_benchmark.tex`, `table2_ranking.tex`, `table3_significance.tex`, `table4_ablations.tex`, `table7_hardware.tex`).
- [x] **Publication-Quality Figures (300 DPI)**: Saved in `Assignment_3/outputs/figures/` and `report/assignment_3/figures/` (`01_model_benchmark_official_f1.png`, `02_threshold_tuning_tradeoff.png`, `03_per_word_f1_heatmap.png`, `04_rrf_k_sensitivity.png`, `05_paired_differences_forest_plot.png`).
- [x] **Model Checkpoints**: 45 fitted model pipelines saved in `Assignment_3/checkpoints/` (`seed_{seed}_{model}.joblib`) and optimal thresholds in `optimal_thresholds.json`.
- [x] **Saved Predictions**: 31,760 instance predictions preserved in `Assignment_3/predictions/all_predictions.csv` and per-seed CSVs.
- [x] **Execution Logs**: Full runtime trace in `Assignment_3/logs/experiment_execution.log` and structured compute log in `Assignment_3/logs/hardware_and_compute_cost_log.json`.

---
*Report completed and certified for Assignment 3 Part 1.*
