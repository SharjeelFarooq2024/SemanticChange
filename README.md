# Semantic Change Across Multiple Time Periods

> **SemEval-2027 Task 3** — Swedish development set  
> Phases complete: **EDA** ✅ | **Baseline A (Classical ML)** ✅ | **Baseline B (Frozen XLM-R)** ⏳

---

## Overview

This project builds a fully reproducible foundation for diachronic semantic
change detection in Swedish. It covers:

- **Subtask 1**: Diachronic Word Sense Induction (unsupervised clustering per word)
- **Subtask 2**: Binary usage classification (supervised, label ∈ {0, 1})

All preprocessing is modular and config-driven. Raw data is never modified.
Seeds are fixed to `42` throughout for exact reproducibility.

---

## Repository Structure

```
SemanticChange/
├── configs/
│   ├── eda.yaml                # EDA pipeline configuration
│   └── baselines.yaml          # All baseline hyperparameters and feature lists
│
├── data/
│   ├── raw/dev/SV/             # Official Swedish development data (read-only)
│   │   ├── usages.jsonl
│   │   ├── subtask1.jsonl
│   │   ├── subtask2.jsonl
│   │   └── definitions.jsonl
│   ├── raw/external/           # Auxiliary synthetic English data (not used in SV analysis)
│   └── processed/eda/          # Cached embeddings and processed artifacts
│
├── src/
│   ├── eda/                    # EDA pipeline modules
│   └── baselines/              # Baseline models
│       ├── __init__.py
│       ├── common.py           # Data loading, embedding cache, leakage-safe split
│       ├── classical_features.py   # Feature matrix builder (TF-IDF + EDA + categorical)
│       ├── baseline_a_subtask2.py  # 7 classical models for Subtask 2
│       ├── baseline_a_subtask1.py  # TF-IDF KMeans sense induction for Subtask 1
│       ├── pretrained_subtask2.py  # Frozen XLM-R → LinearSVC / LR for Subtask 2
│       └── pretrained_subtask1.py  # Frozen XLM-R → KMeans for Subtask 1
│       └── evaluate.py         # Unified evaluation: rankings, confusion matrices, findings
│
├── outputs/
│   ├── eda/                    # EDA tables, figures, PDF report, findings.md
│   └── baselines/
│       ├── classical/          # Per-model metrics, coefficients, feature importances
│       ├── pretrained/         # XLM-R embedding metrics (pending)
│       ├── figures/            # Confusion matrices, coefficient plots
│       └── tables/             # Subtask 1/2 comparison CSV and baseline_findings.md
│
├── tests/
│   ├── test_baseline_data.py   # 8 tests: bundle integrity, join, leakage, cache
│   └── test_classical_features.py  # 5 tests: feature frame, TF-IDF determinism, ablation
│
├── report/assignment_1/        # LaTeX report scaffolding
│   ├── main.tex
│   ├── references.bib
│   └── sections/
│       ├── task.tex
│       ├── dataset.tex
│       ├── eda.tex
│       ├── related_work.tex
│       ├── baselines.tex       # Full Baseline A + B comparison section
│       └── next_steps.tex
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
python -m pip install -r requirements.txt
```

---

## Running the Pipelines

### 1 — EDA Pipeline

```bash
python -m src.eda.run_eda
```

Outputs: `outputs/eda/` (tables, figures, `eda_report.pdf`, `eda_findings.md`)

### 2 — Baseline A: Classical ML

```bash
# Subtask 2 — binary classification (7 models)
python -m src.baselines.baseline_a_subtask2

# Subtask 1 — diachronic sense induction (TF-IDF KMeans per word)
python -m src.baselines.baseline_a_subtask1
```

### 3 — Baseline B: Frozen XLM-R

```bash
# Requires FacebookAI/xlm-roberta-base weights (~1.1 GB, auto-downloaded)
python -m src.baselines.pretrained_subtask2
python -m src.baselines.pretrained_subtask1
```

Embeddings are SHA-256 fingerprinted and cached at
`data/processed/eda/embeddings/`. Subsequent runs reuse the cache instantly.

### 4 — Unified Evaluation

```bash
python -m src.baselines.evaluate
```

Consolidates outputs across Classical (Baseline A) and Pretrained (Baseline B) pipelines, validates data integrity and leakage prevention, computes disaggregated group metrics, generates confusion matrices, and writes an academic findings report:

- `outputs/baselines/tables/baseline_findings.md` — structured Read–Reason–Interpret–Verify findings connecting baseline results back to EDA
- `outputs/baselines/tables/subtask2_comparison.csv` — macro-F1 ranked model comparison table across all configurations
- `outputs/baselines/tables/subtask2_per_group.csv` — disaggregated performance by target word and historical time period
- `outputs/baselines/tables/subtask1_comparison.csv` — per-word clustering concordance (ARI, NMI, purity)
- `outputs/baselines/tables/subtask1_cluster_period_distributions.csv` — cluster count and proportion trajectories across periods
- `outputs/baselines/tables/subtask1_cluster_assignments.csv` — sentence-level cluster membership assignments
- `outputs/baselines/tables/subtask1_temporal_interpretation.md` — temporal dominant-cluster shifts and transition diagnostics
- `outputs/baselines/figures/confusion_matrix_*.png` — per-model confusion matrix visualizations (150 DPI)

The evaluation script also outputs a structured JSON summary to stdout detailing active models, missing baseline suites (handled gracefully), class balance, and join diagnostics.

### 5 — Test Suite

```bash
python -m pytest tests/ -v
```

Expected: **13 passed** in ~6 s.

---

## Baseline Evaluation Framework

The baseline evaluation module ([`evaluate.py`](file:///c:/Users/DAR/Desktop/SemanticChange/src/baselines/evaluate.py)) adheres to a strict evaluation protocol designed to avoid common evaluation fallacies in lexical semantic change:

### 1. Subtask 2 Evaluation (Binary Usage Classification)
- **Primary Metric — Macro-F1**: The Swedish development set exhibits severe class imbalance (64.5 % label 0 vs. 35.5 % label 1). Standard accuracy is misleading: a naive majority baseline achieves 64.5 % accuracy with zero minority-class recall (Macro-F1 = 0.3920). Macro-F1 gives equal weight to both sense classes.
- **Secondary Metrics**: Balanced Accuracy, Macro-Precision, Macro-Recall, and Overall Accuracy.
- **Disaggregated Group Evaluation (`subtask2_per_group.csv`)**: Evaluates performance broken down by:
  - **Target word**: Exposes base-rate memorization shortcuts (e.g. models exploiting that `fru` has an 82.5 % target-sense rate while `fröken` has only 12.6 %).
  - **Time period**: Validates temporal stability and detects performance degradation across historical epochs.
- **Visual Diagnostics**: Produces confusion matrices for each candidate model (`outputs/baselines/figures/confusion_matrix_*.png`).

### 2. Subtask 1 Evaluation (Word Sense Induction)
- **Clustering Concordance**: Evaluated against gold sense partitions using **Adjusted Rand Index (ARI)** (chance-corrected), **Normalized Mutual Information (NMI)** (information-theoretic overlap), and **Purity** (cluster homogeneity).
- **Multi-Label Policy**: In SemEval-2027 Task 3 Subtask 1, 219 usages (11.0 % of Swedish dev records) have multi-valued sense annotations. Per task protocol:
  - Multi-label usages are strictly tracked and preserved in data artifacts.
  - They are excluded from single-partition clustering metric calculations (ARI, NMI, purity) to avoid ground-truth distortion without modifying raw data.
- **Temporal Trajectory Tracking (`subtask1_temporal_interpretation.md`)**:
  - Tracks cluster distribution shifts across periods (`subtask1_cluster_period_distributions.csv`).
  - Explicitly distinguishes genuine semantic drift from non-semantic confounders (genre shift, source corpus distribution, annotator conventions).

### 3. Automated Data Quality & Leakage Audits
On every evaluation run, `evaluate.py` verifies:
- **Leakage Prevention**: Confirms all transformers (TF-IDF vectorizers, scalers, one-hot encoders) are encapsulated inside `sklearn.Pipeline` objects fitted strictly on training folds.
- **Join Integrity**: Audits label-to-usage joins for unmatched records or malformed IDs.
- **Missing Baseline Handlers**: Detects whether Baseline A and Baseline B outputs are present, gracefully reporting missing pipelines without pipeline failure.
- **Synthesis Report**: Automatically compiles `baseline_findings.md` adhering to the **Read–Reason–Interpret–Verify** scientific framework.

---

## Baseline A Results — Subtask 2 (SV dev, `n_test=397`)

Primary metric: **macro-F1** (class-imbalance robust).  
Split: stratified 80/20, seed 42. Class ratio: 0 → 64.5 %, 1 → 35.5 %.

| Rank | Model | Representation | F1-macro | Bal-Acc | Accuracy |
|------|-------|---------------|----------|---------|----------|
| 1 | **EDA-feat. LR** | TF-IDF + EDA features | **0.8163** | **0.8049** | 83.9 % |
| 2 | Word+Char TF-IDF SVM | Word & char TF-IDF | 0.8155 | 0.8033 | 83.9 % |
| 3 | TF-IDF LR | Word TF-IDF | 0.7917 | 0.7826 | 81.6 % |
| 4 | EDA-feat. RF | TF-IDF + EDA features | 0.7910 | 0.7741 | 82.4 % |
| 5 | Ablated LR *(no word ID)* | Context only | 0.7129 | 0.7165 | 73.3 % |
| 6 | Complement NB | Word TF-IDF | 0.6171 | 0.6253 | 73.0 % |
| 7 | Majority baseline | Constant 0 | 0.3920 | 0.5000 | 64.5 % |
| — | Frozen XLM-R LR | 768-d mean-pool | *(pending)* | — | — |
| — | Frozen XLM-R SVM | 768-d mean-pool | *(pending)* | — | — |

> **Word-identity ablation gap**: ΔF1 ≈ 10.3 points (0.8163 → 0.7129) quantifies
> how much signal comes from token-identity shortcuts vs. contextual features alone.

---

## Baseline A Results — Subtask 1 (TF-IDF KMeans, k=3)

Evaluation on singleton-labelled records only; multi-label records excluded.

| Word | Records | Eval n | Excl. | ARI | NMI | Purity |
|------|---------|--------|-------|-----|-----|--------|
| fru | 274 | 195 | 79 | **0.255** | 0.096 | 0.867 |
| motiv | 291 | 289 | 2 | 0.144 | 0.051 | 0.785 |
| kvinna | 290 | 284 | 6 | 0.060 | 0.039 | 0.775 |
| dumpa | 145 | 103 | 42 | 0.042 | 0.107 | 0.815 |
| herre | 283 | 241 | 42 | 0.040 | 0.066 | 0.838 |
| suga | 129 | 127 | 2 | 0.038 | 0.082 | 0.323 |
| exportera | 149 | 149 | 0 | 0.005 | 0.042 | 0.946 |
| skär | 98 | 95 | 3 | −0.003 | 0.052 | 0.547 |
| förort | 150 | 71 | 79 | −0.015 | 0.018 | 0.577 |
| fröken | 174 | 167 | 7 | −0.104 | 0.065 | 0.838 |

> ⚠️ **Important**: Temporal cluster shifts do not directly prove semantic change.
> Alternative explanations (genre shift, source corpus change, sample imbalance,
> annotator conventions) must be ruled out before causal claims are made.

---

## Interpretability Highlights

**LR Coefficients (TF-IDF LR, top positive → label 1):**
`fru` (+4.37), `exportera` (+2.02), `sin fru` (+1.60), `exporterar` (+1.57)

**LR Coefficients (top negative → label 0):**
`fröken` (−2.12), `herrar` (−1.46), `motivet` (−1.17)

These reflect **per-word label prevalence**, not semantic change detection.
The word-identity ablation confirms this: removing target-word identity costs
~10 F1 points, quantifying the base-rate shortcut.

**RF Gini Top Features** (of 45,190 total):
1. `categorical__word_exportera` (0.0248)
2. `categorical__word_herre` (0.0204)
3. `numeric__target_position` (0.0199) ← validates EDA structural finding

---

## Configuration

All hyperparameters and feature lists are controlled from a single file:

```yaml
# configs/baselines.yaml
features:
  numeric_features:
    - text_length_chars
    - token_count
    - ...
  categorical_features:
    - word
    - pos
    - period_label
  ablated_categorical_features:
    - pos
    - period_label   # 'word' excluded in ablation

models:
  subtask2:
    tfidf_logistic_regression:
      max_features: 50000
      ngram_range: [1, 2]
      ...
```

---

## Reproducibility Notes

| Concern | Implementation |
|---------|---------------|
| Seed | Fixed to `42` in `random`, `numpy`, `sklearn` |
| Train/test split | Stratified 80/20 via `sklearn.model_selection.train_test_split` |
| Leakage prevention | All transformers inside `sklearn.Pipeline`, fitted on train only |
| Embedding cache | SHA-256 fingerprint over data + model config; validated before reuse |
| Raw data | `data/raw/` is read-only; never modified |
| Model weights | Excluded from git via `.gitignore` |

---

## Baseline B — Frozen XLM-R (Pending)

**Architecture**: `FacebookAI/xlm-roberta-base` (frozen, no gradient updates)  
**Method**: mean-pooled final hidden layer over ±80-char context window  
**Downstream**: `LinearSVC` + `LogisticRegression` (Subtask 2); `KMeans` (Subtask 1)

Baseline B is **feature extraction**, not fine-tuning. The encoder parameters
are not updated. This establishes a reproducible reference point against which
future fine-tuned models can be directly compared.

Run once XLM-R weights are cached:
```bash
python -m src.baselines.pretrained_subtask2
python -m src.baselines.pretrained_subtask1
python -m src.baselines.evaluate
```

---

## Report

The LaTeX report is under `report/assignment_1/`. The baseline comparison
section (`sections/baselines.tex`) covers:

1. Task formulation (Subtask 1 WSI vs. Subtask 2 classification)
2. Data split and leakage prevention
3. Classical feature design and EDA connections
4. TF-IDF model configurations (7 models)
5. LR coefficient and RF Gini interpretation
6. Frozen XLM-R method (feature extraction vs. fine-tuning)
7. Evaluation metrics rationale
8. Full results tables (Subtask 2 and Subtask 1)
9. Per-word and per-period error analysis
10. Exploratory vs. predictive evidence distinction
11. Alternative explanations for apparent cluster shifts
12. Limitations (7 items)
