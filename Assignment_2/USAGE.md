# Assignment 2 — Core Retrieval System & Evaluation Framework

> **SemEval-2027 Task 3** — Swedish development set  
> Systems: **BM25** ✅ | **Dense (frozen pretrained encoder)** ✅ | **Hybrid (RRF)** ✅  
> Assignment 1 is untouched: this directory is fully self-contained and only *reads*  
> the shared dataset under `data/raw/dev/SV/`.

---

## 1. Project overview

Assignment 2 reformulates Subtask 2 (*Hypothesis-Driven Change Detection*) as a
retrieval problem — the shared task itself defines it as *"given a target sense
definition, identify which usages express that sense"*:

| Retrieval concept | Assignment 2 realization |
|---|---|
| **Document** | one usage sentence (`doc_id` = `sentence_id` from `usages.jsonl`) |
| **Query** | one target-sense definition (`query_id` = `word` from `definitions.jsonl`) |
| **Relevance (qrel)** | Subtask 2 label: `1` = usage expresses the sense, `0` otherwise (`subtask2.jsonl`) |
| **Collection** | 1,983 documents · 10 queries · 1,983 qrels (705 relevant) |

Three ranked retrieval systems are implemented:

1. **BM25** — Okapi lexical retrieval (`rank_bm25`, k1/b configurable).
2. **Dense** — frozen pretrained encoder (`intfloat/multilingual-e5-small`,
   primary; `FacebookAI/xlm-roberta-base`, ablation) + FAISS exact cosine search.
3. **Hybrid** — BM25 + dense candidates fused with **Reciprocal Rank Fusion (RRF)**.

All systems are inference-only: no training, no fine-tuning, no gradient updates.
Raw Assignment 1 data is never modified.

## 2. What the system does

For each of the 10 definition queries, each system ranks the eligible usage
documents (candidate scope is configurable: `per_word` = only the target word's
usages — the default, matching official Subtask 2 semantics; `global` = the whole
corpus). Outputs per system:

- a **ranked list** (`query_id, doc_id, rank, score`) — the primary artifact, with
  full scores so any cutoff/metric can be computed downstream;
- a **binary export** (`query_id, sentence_id, word, label, score, rank`) covering
  the query's *entire* candidate pool, where `label = 1 iff rank <= binary_top_k`
  (naive default operating point, see §16);
- a `run_metadata.json` with the exact config snapshot.

## 3. Architecture and data flow

```
data/raw/dev/SV/*.jsonl  (Assignment 1, read-only)
        │
        ▼
 data_loader ──► data_validator ──► RetrievalBundle {documents, queries, qrels}
        │
        ├──► preprocessing (TextPreprocessor / Tokenizer)  ──► BM25Retriever
        │                                                       │ build/load index (pickle)
        └──► DenseEncoder (frozen, mean-pool, prefixes)         │
                 │ encode passages once                         │
                 ▼                                              │
          EmbeddingCache (.npz + identity metadata)             │
                 │                                              │
                 ▼                                              │
          FAISS IndexFlatIP (cosine over normalized vectors)    │
                 │                                              │
                 └──────────────┬───────────────────────────────┘
                                ▼
                    HybridRetriever / fusion.reciprocal_rank_fusion
                                │
                                ▼
              prediction_io → predictions/{bm25,dense,hybrid}/
```

## 4. Directory structure

```
Assignment_2/
├── README.md                  # this file
├── run.py                     # unified CLI entry point
├── configs/
│   ├── data_config.yaml       # dataset paths + retrieval formulation
│   ├── bm25_config.yaml       # k1, b, tokenizer, index path, top_k, scope
│   ├── dense_config.yaml      # encoder profiles, cache paths, top_k, scope
│   └── hybrid_config.yaml     # RRF k, per-system candidate counts, top_k
├── src/
│   ├── pipeline.py            # end-to-end orchestration for all 3 systems
│   ├── data/
│   │   ├── data_loader.py     # Document/Query/Qrel + strict JSONL loading
│   │   └── data_validator.py  # integrity checks (clear errors, no silent drops)
│   ├── preprocessing/
│   │   ├── text_preprocessor.py  # non-destructive normalization
│   │   └── tokenizer.py          # Unicode/Swedish/digit-preserving tokenizer
│   ├── retrieval/
│   │   ├── base.py            # Hit dataclass + shared Retriever interface
│   │   ├── bm25_retriever.py  # rank_bm25 wrapper with save/load
│   │   ├── dense_retriever.py # FAISS vector search with candidate filtering
│   │   ├── hybrid_retriever.py# independent retrieval + fusion
│   │   ├── fusion.py          # RRF (extensible fusion registry)
│   │   └── prediction_io.py   # ranked/binary CSV + metadata writers
│   ├── embeddings/
│   │   ├── encoder.py         # frozen DenseEncoder (e5_small / xlmr_base)
│   │   └── embedding_manager.py  # fingerprinted, validated embedding cache
│   └── utils/
│       ├── paths.py           # repo/A2 path resolution + config loader
│       ├── logging_utils.py   # console + per-run file logging
│       └── reproducibility.py # seed control (42)
├── scripts/
│   ├── build_bm25_index.py    # prebuild/refresh the BM25 index
│   ├── generate_embeddings.py # prebuild/refresh embeddings + FAISS index
│   ├── run_bm25.py            # BM25 retrieval only
│   ├── run_dense.py           # dense retrieval only
│   └── run_hybrid.py          # hybrid retrieval only
├── tests/                     # 69 unit/integration tests (no downloads)
├── artifacts/                 # derived, gitignored (indexes, embeddings, model cache)
├── predictions/               # outputs (tracked): bm25/ dense/ hybrid/ + qrels.csv
└── logs/                      # per-run log files (gitignored)
```

## 5. Dependencies and installation

Python 3.10. All Assignment 1 dependencies still apply. Assignment 2 adds:

```bash
python -m pip install rank_bm25 faiss-cpu
```

No other new packages: dense encoding uses the already-installed
`torch` + `transformers` directly (same style as Assignment 1), not
`sentence-transformers`.

## 6. Dataset paths / configuration

All paths are declared in `configs/*.yaml` and resolved by `src/utils/paths.py`:

```yaml
# configs/data_config.yaml
data:
  root: data/raw/dev/SV        # relative to the repository root
  usages: usages.jsonl         # documents
  subtask2: subtask2.jsonl     # qrels
  definitions: definitions.jsonl  # queries
```

The dataset is **referenced, never copied**. Schemas (verified against the files):

- `usages.jsonl`: `word, sentence_id, text, pos, period_label, start, end, year`
- `definitions.jsonl`: `word, definition` (10 rows)
- `subtask2.jsonl`: `word, sentence_id, label ∈ {0,1}` (1,983 rows)

`load_bundle_validated()` verifies: unique/non-empty IDs, non-empty text, every
qrel joins to an existing document and query, every query has a non-empty
candidate pool. All problems are reported together in one `ValidationError`.

## 7. Build the BM25 index

```bash
python Assignment_2/scripts/build_bm25_index.py
```

Creates `artifacts/bm25_index/bm25_index.pkl`. The index is reused on later runs
and automatically rebuilt if the document set changes (ID-list consistency check).
Parameters `k1`, `b`, and the tokenizer live in `configs/bm25_config.yaml`.

## 8. Generate dense embeddings

```bash
python Assignment_2/scripts/generate_embeddings.py                 # e5_small (default)
python Assignment_2/scripts/generate_embeddings.py --encoder xlmr_base
python Assignment_2/scripts/generate_embeddings.py --force         # ignore cache
```

First run downloads `intfloat/multilingual-e5-small` (~470 MB, one time) into
`artifacts/model_cache/`, embeds all 1,983 passages (~3 min on CPU), and writes:

- `artifacts/document_embeddings/e5_small.npz` — float32 matrix + ordered doc IDs
- `artifacts/document_embeddings/e5_small_meta.json` — identity + shape
- `artifacts/document_embeddings/e5_small.faiss` + `..._ids.json` — vector index

**Cache safety:** the cache is reused only if *every* identity field matches —
encoder key, model name, max_length, pooling, prefixes, normalization, and a
SHA-256 fingerprint of the ordered `(doc_id, text)` pairs — plus exact doc-ID
list and dimension checks. Any mismatch triggers regeneration with a logged
reason (never a silent reuse).

## 9. Run BM25 retrieval

```bash
python Assignment_2/run.py --system bm25
# or
python Assignment_2/scripts/run_bm25.py
python Assignment_2/scripts/run_bm25.py --scope global --top-k 50 --limit 3
```

## 10. Run dense retrieval

```bash
python Assignment_2/run.py --system dense
python Assignment_2/run.py --system dense --encoder xlmr_base   # ablation encoder
# or
python Assignment_2/scripts/run_dense.py
```

## 11. Run hybrid retrieval

```bash
python Assignment_2/run.py --system hybrid
python Assignment_2/run.py --system all          # bm25 → dense → hybrid
# or
python Assignment_2/scripts/run_hybrid.py
```

Hybrid fetches `candidates.bm25` + `candidates.dense` lists independently (raw
scores are never mixed), fuses ranks with `RRF_score(d) = Σ 1/(k + rank_i(d))`
(`fusion.rrf_k`, default 60 = Cormack et al. 2009 standard), deduplicates by
`doc_id`, and re-sorts deterministically (score desc, doc_id asc on ties).

### CLI reference (all entry points)

| Flag | Values | Meaning |
|---|---|---|
| `--system` | `bm25\|dense\|hybrid\|all` | which system(s) to run (`run.py`) |
| `--scope` | `per_word\|global` | candidate restriction (default from config) |
| `--top-k` | int | retrieval cutoff (default 100) |
| `--encoder` | `e5_small\|xlmr_base` | dense encoder profile |
| `--limit` | int | run only the first N queries (sorted by query_id) |

Note: `--limit` runs write to `predictions/<system>_limitN/` so subset tests
never overwrite full-run results. Model weights are looked up first in
`artifacts/model_cache/` and fall back to the default HuggingFace cache
(reusing Assignment 1's existing XLM-R download for the ablation encoder).

## 12. Loading cached artifacts

```python
from Assignment_2.src.retrieval.bm25_retriever import BM25Retriever
from Assignment_2.src.retrieval.dense_retriever import DenseRetriever
from Assignment_2.src.utils.paths import load_config

bm25 = BM25Retriever(load_config("bm25_config"))
bm25.load_or_build(documents, "Assignment_2/artifacts/bm25_index/bm25_index.pkl")

dense = DenseRetriever(load_config("dense_config"))
dense.build_index(documents)   # reuses validated .npz + .faiss automatically
```

Nothing is ever rebuilt when a compatible artifact exists (verified in logs:
`Loaded cached embeddings (validated)` / `Loaded FAISS index`).

## 13. Where predictions are saved

```
predictions/
├── qrels.csv                        # reference qrels for evaluation
├── bm25/
│   ├── ranked.csv                   # query_id,doc_id,rank,score
│   ├── binary_predictions.csv       # query_id,sentence_id,word,label,score,rank
│   └── run_metadata.json            # config snapshot + counts + timestamp
├── dense/                           # same three files
└── hybrid/                          # same three files
```

Sample (`predictions/hybrid/ranked.csv`):

```csv
query_id,doc_id,rank,score
dumpa,ae4e7245_76_83,1,0.029287
dumpa,de8eb8a_43_49,2,0.028992
dumpa,32f06c53_71_77,3,0.028624
```

Sample (`predictions/hybrid/binary_predictions.csv`):

```csv
query_id,sentence_id,word,label,score,rank
dumpa,91a035ba_75_80,dumpa,0,0.014378,69
dumpa,de8eb8a_43_49,dumpa,1,0.028992,2
```

## 14. Run tests

```bash
python -m pytest Assignment_2/tests/ -v        # 69 tests, < 5 s, no downloads
python -m pytest tests/ -v                     # Assignment 1 regression (13 tests)
```

Optional real-encoder smoke test (downloads/uses the e5 weights):

```bash
set A2_RUN_SLOW=1 && python -m pytest Assignment_2/tests/test_dense.py -v
```

Tests cover: data loading + ID preservation + validator failure modes,
preprocessing determinism, BM25 ranking/persistence/config, embedding-cache
fingerprint rejection, FAISS retrieval validity, RRF math (hand-verified),
duplicate handling, and prediction schema integrity. Unit tests use a
deterministic fake encoder and a tiny synthetic corpus — no full-dataset runs.

## 15. Hardware requirements

| Resource | Measured on this machine (i5-1155G7, 8 GB RAM, CPU-only) |
|---|---|
| Model download | ~470 MB one-time (`e5_small`) |
| Embedding generation | 1,983 passages ≈ **3 min** (CPU, batch 64) |
| Embedding cache | `e5_small.npz` ≈ **3 MB** (1983 × 384 float32) |
| FAISS index | ≈ **3 MB** (IndexFlatIP, exact search) |
| BM25 index | ≈ **1 MB** (pickle) |
| Per-run retrieval | BM25 < 1 s · dense ≈ 15 s (query encoding dominates) · hybrid ≈ 15 s |
| RAM | well under 2 GB peak |

No GPU required. If CUDA is present it is used automatically
(`encoder.device: auto`).

## 16. Known limitations

1. **10 queries only** — the dataset provides one definition per target word.
   Any metric computed over 10 queries has high variance; report per-query
   results alongside means.
2. **Naive binary operating point** — `binary_export.top_k: 50` is an arbitrary
   default (documented, not tuned). Operating-point selection belongs to the
   evaluation phase (Member 2); the ranked CSVs contain full scores so any
   cutoff can be recomputed without re-running retrieval.
3. **No official retrieval output schema exists** — the shared task's own
   Subtask 2 output is per-usage binary labels; the ranked CSV is this
   project's documented interface (confirmed with the group).
4. **Exact FAISS search** — `IndexFlatIP` over ≤1,983 vectors is exact and
   instant. For much larger corpora an ANN index (IVF/HNSW) plus ID-selector
   filtering would be needed; per-word restriction currently fetches the full
   ranking then filters (exact, cheap at this scale).
5. **XLM-R ablation is not retrieval-trained** — it exists for controlled
   comparison with Assignment 1, not as a competitive retriever.
6. **Encoder revision pinning** — caches record the model *name* but not a
   HuggingFace revision hash; if the upstream model changes in place, the cache
   would not detect it (pin `revision` in config if this becomes a concern).
7. **Dev-only data** — the official test set is not available yet; everything
   runs on the Swedish development release.

---

## 17. Evaluation, Tuning, and Ablations (Fully Implemented)

The evaluation, threshold optimization, and ablation framework is fully operational:

### 1. Unified Evaluation CLI

```bash
python Assignment_2/evaluate.py
```

Runs comprehensive evaluation over BM25, Dense, and Hybrid predictions against `predictions/qrels.csv`:
- Calculates all IR metrics: P@K, Recall@K, MRR, MAP, nDCG@K.
- Sweeps rank cutoffs ($K^* \in [5, 100]$) to find optimal decision thresholds for official Subtask 2 Macro-$F_1$.
- Exports calibrated binary predictions (`predictions/{system}/binary_predictions_tuned.csv`).
- Conducts subgroup robustness audits across 10 target words and 10 diachronic periods.
- Runs full ablation sweeps across Candidate Scope (`per_word` vs `global`), RRF smoothing ($k \in \{10, 30, 60, 100\}$), and decision cutoffs.

### 2. Generated Outputs (`outputs/evaluation/`)

| Artifact | Contents |
|---|---|
| `retrieval_metrics_comparison.csv` | Cross-system ranking metrics (MRR, MAP, nDCG, P@K, Recall@K) |
| `subtask2_classification_comparison.csv` | Naive vs tuned classification metrics per retrieval system |
| `unified_subtask2_benchmark.csv` | Direct benchmark against Assignment 1 Majority, Classical ML, & XLM-R |
| `subgroup_per_word.csv` | Performance broken down across all 10 target words |
| `subgroup_per_period.csv` | Performance broken down across all 10 diachronic periods |
| `ablation_scope.csv` | Empirical proof of `per_word` vs `global` candidate scoping |
| `ablation_fusion_k.csv` | Sensitivity curve across RRF smoothing constant $k$ |
| `ablation_operating_point.csv` | Gain quantification from naive $K=50$ to tuned $K^*$ |

### 3. Report Scaffolding (`report/assignment_2/`)

A publication-grade LaTeX report following Dr. Mehwish Fatima's 9-section specification is located under `report/assignment_2/main.tex`.

