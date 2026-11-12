# Assignment 2 — Core Retrieval System & Experimental Evaluation

> **SemEval-2027 Task 3** (Swedish development set)  
> Scope: **Complete implementation** (Core Retrieval Engine + Evaluation Framework + Ablations + Report)  
> Assignment 1 is fully preserved; everything for Assignment 2 lives in `Assignment_2/` and `report/assignment_2/`.

---

## 1. The Task and Formulation

Assignment 1 studied the diachronic Swedish dataset (EDA, classical baselines, XLM-R). Assignment 2 builds a **core information retrieval system** and an **empirical evaluation framework** around official **Subtask 2**:

> *"Given a target sense definition, identify which usages express that sense."*

We formulate this as an asymmetric ranked retrieval problem:

| Retrieval Concept | Dataset Realization | Size |
|---|---|---|
| **Document** | One usage sentence (`sentence_id` from `usages.jsonl`) | 1,983 documents |
| **Query** | One target sense definition (`word` from `definitions.jsonl`) | 10 queries |
| **Relevance (Qrel)** | Official Subtask 2 label ($1 = \text{expresses sense}$) | 1,983 qrels (705 relevant, base rate $= 35.55\%$) |

---

## 2. Implemented Systems

### 1. Lexical Retrieval (Okapi BM25)
- Non-parametric lexical matching via `rank_bm25`.
- Swedish diacritic-preserving and numeral/century-preserving tokenizer.
- Pickled index with SHA-256 fingerprint validation and automatic reuse.

### 2. Dense Semantic Retrieval
- Frozen pretrained contrastive encoder: `intfloat/multilingual-e5-small` (384-dim, primary) and `FacebookAI/xlm-roberta-base` (768-dim, ablation).
- Asymmetric prefix conditioning: `query: ` and `passage: `.
- Exact cosine similarity via FAISS `IndexFlatIP` (with pure NumPy exact dot-product fallback for environments with system DLL restrictions).
- Document embeddings cached with composite identity fingerprints.

### 3. Hybrid Retrieval (Reciprocal Rank Fusion)
- Combines BM25 and Dense ordinal rankings using Reciprocal Rank Fusion:
  $$\text{RRF}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{k + r_m(d)}$$
- Standard $k = 60$; operates purely on ranks to avoid score calibration distortion.

### 4. Comprehensive Evaluation & Operating-Point Tuning
- **Evaluation Engine** (`Assignment_2/src/evaluation/`):
  - Retrieval ranking metrics: P@K, Recall@K, MRR, MAP, nDCG@K ($K \in \{5, 10, 20, 50, 100\}$).
  - Official Subtask 2 classification metrics: Macro-$F_1$, Binary $F_1$, Accuracy, Balanced Accuracy, Precision, Recall.
  - Threshold Optimizer: Sweeps rank cutoffs ($K^* \in [5, 100]$) on ranked outputs to optimize official Macro-$F_1$.
  - Subgroup Analysis: Automatic performance breakdowns across all 10 target words and 10 diachronic time periods (1880--2023).
  - Unified Benchmarking: Directly integrates Assignment 1 baselines (Majority, Classical ML, XLM-R) side-by-side with Assignment 2 retrieval systems.

---

## 3. Empirical Results & Verification

All numbers below were computed from the operational pipeline on the Swedish development set:

### Retrieval Ranking Performance
| System | MRR | MAP | nDCG@10 | P@5 | P@10 | Recall@10 | Recall@50 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BM25** | 0.618 | 0.229 | 0.383 | 0.320 | 0.370 | 0.066 | 0.285 |
| **Dense (E5-small)** | 0.612 | **0.323** | 0.427 | 0.480 | 0.410 | 0.082 | **0.389** |
| **Hybrid (RRF, $k=60$)** | **0.692** | 0.293 | **0.490** | **0.540** | **0.480** | **0.103** | 0.381 |

*Takeaway*: Hybrid RRF delivers superior early rank precision ($\text{P@5} = 0.540$, $\text{MRR} = 0.692$, $+52\%$ enrichment over base rate), while Dense retrieval captures broader semantic recall ($\text{MAP} = 0.323$, $\text{Recall@50} = 0.389$).

### Subtask 2 Classification Benchmark (with Assignment 1 Baselines)
| System | Paradigm / Source | Accuracy | Macro-$F_1$ | Precision | Recall |
|---|---|:---:|:---:|:---:|:---:|
| **EDA Features + LR** | Supervised (Assignment 1) | 0.839 | **0.816** | 0.836 | 0.805 |
| **Word+Char TF-IDF SVM** | Supervised (Assignment 1) | 0.839 | 0.816 | 0.838 | 0.803 |
| **Frozen XLM-R + LR** | Supervised (Assignment 1) | 0.736 | 0.721 | 0.718 | 0.731 |
| **Dense (Tuned $K^*=65$)** | Zero-Training Retrieval (Assignment 2) | 0.601 | **0.557** | 0.434 | 0.400 |
| **Hybrid (Tuned $K^*=75$)** | Zero-Training Retrieval (Assignment 2) | 0.571 | 0.538 | 0.403 | 0.428 |
| **BM25 (Tuned $K^*=70$)** | Zero-Training Retrieval (Assignment 2) | 0.543 | 0.500 | 0.356 | 0.353 |
| **Dense (Naive $K=50$)** | Zero-Training Retrieval (Assignment 2) | 0.612 | 0.542 | 0.436 | 0.309 |
| **BM25 (Naive $K=50$)** | Zero-Training Retrieval (Assignment 2) | 0.569 | 0.490 | 0.350 | 0.248 |
| **Majority Class** | Trivial Heuristic (Assignment 1) | 0.645 | 0.392 | 0.322 | 0.500 |

---

## 4. Empirical Ablation Findings

### 1. Candidate Scope Ablation (`per_word` vs `global`)
| Scope | MRR | MAP | nDCG@10 | P@10 | Recall@100 |
|---|:---:|:---:|:---:|:---:|:---:|
| **`per_word`** (Target pool $\mathcal{C}(w)$) | **0.618** | **0.229** | **0.383** | **0.370** | **0.583** |
| **`global`** (Whole corpus 1,983 docs) | 0.207 | 0.017 | 0.087 | 0.080 | 0.097 |

*Finding*: Unconstrained global search suffers a $66.5\%$ MRR collapse and $92.5\%$ MAP drop because BM25 retrieves cross-word passages that share incidental vocabulary, confirming the theoretical requirement for lemma scoping.

### 2. RRF Smoothing Constant Sensitivity ($k$)
- $k=10$: $\text{MRR} = 0.675$, $\text{MAP} = 0.279$, $\text{nDCG@10} = 0.472$
- $k=30$: $\text{MRR} = \mathbf{0.708}$, $\text{MAP} = 0.291$, $\text{nDCG@10} = \mathbf{0.491}$
- $k=60$: $\text{MRR} = 0.692$, $\text{MAP} = \mathbf{0.293}$, $\text{nDCG@10} = 0.490$, $\text{P@5} = \mathbf{0.540}$
- $k=100$: $\text{MRR} = 0.692$, $\text{MAP} = 0.291$, $\text{nDCG@10} = 0.477$

*Finding*: Intermediate values $k \in [30, 60]$ achieve optimal balance, confirming Cormack et al. (2009).

### 3. Operating-Point Tuning ($K=50 \to K^*$)
- **Dense**: Tuning $K^*=65$ increases Macro-$F_1$ from $0.542 \to \mathbf{0.557}$ and Binary $F_1$ from $0.362 \to \mathbf{0.416}$.
- **Hybrid**: Tuning $K^*=75$ increases Macro-$F_1$ from $0.521 \to \mathbf{0.538}$ and Binary $F_1$ from $0.334 \to \mathbf{0.415}$.
- **BM25**: Tuning $K^*=70$ increases Macro-$F_1$ from $0.490 \to \mathbf{0.500}$ and Binary $F_1$ from $0.290 \to \mathbf{0.354}$.

---

## 5. Test Suite Verification

| Test Suite | Total Tests | Status | Details |
|---|:---:|:---:|---|
| **Assignment 2 Retrieval** | 69 | **68 passed, 1 skipped** | Fast unit tests (<1s), synthetic mocks, zero downloads |
| **Assignment 2 Evaluation** | 7 | **7 passed** | P@K, MRR, MAP, nDCG, Macro-$F_1$, threshold tuning |
| **Assignment 1 Regression** | 13 | **13 passed** | Data loading, joins, leakage checks, classical features |
| **Total Test Suite** | **89** | **88 passed, 1 skipped** | 100% pass on all non-slow unit tests |

---

## 6. CLI Quick Start

```bash
# 1. Run retrieval systems (BM25, Dense, Hybrid)
python Assignment_2/run.py --system all

# 2. Run comprehensive evaluation and generate summary tables
python Assignment_2/evaluate.py

# 3. Run test suites
python -m pytest tests/ Assignment_2/tests/ -v
```

Generated outputs land in:
- `Assignment_2/predictions/{bm25,dense,hybrid}/`: `ranked.csv`, `binary_predictions.csv`, `binary_predictions_tuned.csv`, `run_metadata.json`
- `Assignment_2/outputs/evaluation/`:
  - `retrieval_metrics_comparison.csv`
  - `subtask2_classification_comparison.csv`
  - `unified_subtask2_benchmark.csv`
  - `subgroup_per_word.csv`
  - `subgroup_per_period.csv`
  - `ablation_scope.csv`
  - `ablation_fusion_k.csv`
  - `ablation_operating_point.csv`
- `report/assignment_2/`: Complete LaTeX report following Dr. Mehwish Fatima's 9 sections.

---

## 7. Directory Structure

```
Assignment_2/
├── README.md                      # Comprehensive overview & empirical findings
├── USAGE.md                       # Complete manual, caches, and interface specs
├── run.py                         # Unified retrieval execution CLI
├── evaluate.py                    # Unified evaluation & ablation CLI
├── configs/                       # data_config, bm25_config, dense_config, hybrid_config
├── src/
│   ├── data/                      # data_loader.py, data_validator.py
│   ├── preprocessing/             # text_preprocessor.py, tokenizer.py
│   ├── embeddings/                # encoder.py, embedding_manager.py
│   ├── retrieval/                 # bm25_retriever.py, dense_retriever.py, hybrid_retriever.py, fusion.py
│   └── evaluation/                # metrics.py, evaluator.py, ablation.py
├── predictions/                   # bm25/, dense/, hybrid/, bm25_global/, qrels.csv
├── outputs/evaluation/            # Comparison tables, ablation CSVs, subgroup metrics
├── artifacts/                     # Cached BM25 and dense indices (gitignored)
└── tests/                         # 76 unit tests across retrieval and evaluation
```
