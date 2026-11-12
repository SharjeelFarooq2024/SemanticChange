# Assignment 2 — What We Did

> **Core retrieval system for SemEval-2027 Task 3** (Swedish development set)
> Role: **Member 1 (60%) — coding** · Status: **complete and tested**
> Assignment 1 is fully preserved; everything here lives in `Assignment_2/`.

---

## 1. The task and the key idea

Assignment 1 studied the dataset (EDA, baselines). Assignment 2 asked us to build
a **core retrieval pipeline**: data loading, BM25, dense retrieval, hybrid
fusion (RRF), and ranked prediction generation.

The catch: the Assignment 1 dataset contains **no retrieval files** — no queries,
no qrels, no ranked-output format. So the first job was to formulate the task.
The shared task itself gave the answer. Official **Subtask 2** is defined as:

> *"Given a target sense definition, identify which usages express that sense."*

That is a retrieval problem in disguise. We reformulated it:

| Retrieval concept | What it is in our data |
|---|---|
| **Document** | one usage sentence (`sentence_id` from `usages.jsonl`) — 1,983 docs |
| **Query** | one target-sense definition (`word` from `definitions.jsonl`) — 10 queries |
| **Relevance** | the Subtask 2 gold label: `1` = usage expresses the sense — 1,983 qrels (705 relevant) |

From that foundation we built three ranked retrieval systems over the same
validated data, all inference-only (frozen encoder, no training).

## 2. Decisions we made before coding

Confirmed with the group up front so nothing irreversible was assumed:

1. **Formulation** — definition→usage retrieval, qrels = Subtask 2 labels.
2. **Candidate scope** — configurable: `per_word` (default; only the target
   word's usages are candidates, matching official Subtask 2 semantics) or
   `global` (whole corpus), so scope itself becomes an ablation axis.
3. **Output schema** — no official ranked format exists, so we ship both a
   **ranked CSV** (`query_id,doc_id,rank,score`) and an official-style
   **binary export** (`sentence_id,word,label`) per system.
4. **Encoder** — `intfloat/multilingual-e5-small` as the primary dense encoder
   (contrastive retrieval training, Swedish-capable, 384-dim, CPU-friendly),
   with Assignment 1's frozen `xlm-roberta-base` kept as an ablation encoder.

## 3. What we built

**Data layer**
- `data_loader.py` — loads documents/queries/qrels from the existing Assignment
  1 JSONL files (referenced, never copied or modified). Malformed JSON fails
  with the exact line number; nothing is silently dropped.
- `data_validator.py` — checks ID uniqueness, empty text, qrel join integrity,
  and candidate-pool coverage; reports *all* problems in one clear error.

**Preprocessing**
- A non-destructive normalizer and a Unicode-aware tokenizer. Swedish letters
  (å ä ö) and **digits/years are preserved** — temporal information is signal in
  this task, so no stopword/digit/period stripping happens anywhere.

**BM25** (`rank_bm25`)
- Okapi BM25 with configurable `k1`/`b`, shared tokenizer, original IDs kept,
  index pickled to disk and automatically reused (rebuilt only if the corpus
  changes).

**Dense retrieval**
- `encoder.py` — frozen pretrained encoder (eval mode, `no_grad`, mean pooling,
  E5 `query:`/`passage:` prefixes, L2 normalization, batching, CPU/GPU auto).
- `embedding_manager.py` — document embeddings are generated **once** and
  cached with a full compatibility check (model, pooling, prefixes, plus a
  SHA-256 fingerprint of every doc ID and text). An incompatible cache is
  never silently reused — it regenerates with a logged reason.
- `dense_retriever.py` — FAISS exact search (cosine over normalized vectors)
  with a vector-position → `sentence_id` map, save/reload, and exact per-word
  candidate filtering.

**Hybrid + RRF**
- `fusion.py` — Reciprocal Rank Fusion: `RRF(d) = Σ 1/(k + rank_i(d))`,
  default `k = 60` (Cormack et al., 2009). Fusion combines **ranks only** —
  raw BM25 and cosine scores are never mixed. Duplicate docs are collapsed,
  ties broken deterministically. New fusion methods can be registered later
  without touching retrieval code.
- `hybrid_retriever.py` — pulls candidate lists from BM25 and dense
  independently, fuses, truncates to `top_k`.

**Prediction generation + pipeline**
- Every system writes `ranked.csv`, `binary_predictions.csv`, and
  `run_metadata.json` (config snapshot + counts + timestamp), plus a shared
  `qrels.csv` for evaluation.
- `run.py` / `scripts/` — one-command runs for each system, with `--scope`,
  `--top-k`, `--encoder`, and `--limit` (subset runs go to their own directory
  so they never overwrite full results).

## 4. Architecture at a glance

```
data/raw/dev/SV/*.jsonl  (Assignment 1, read-only)
        │
        ▼
 loader ─► validator ─► RetrievalBundle {documents, queries, qrels}
        │
        ├──► Tokenizer ──────────────► BM25Retriever ────┐
        │                                                │
        └──► DenseEncoder (frozen) ─► cache ─► FAISS ────┤
                                                         ▼
                              fusion.reciprocal_rank_fusion (RRF)
                                                         │
                                                         ▼
                              predictions/{bm25,dense,hybrid}/
                                    ranked.csv · binary · metadata
```

## 5. Results and verification

Everything below was actually run and tested — nothing is assumed:

| Check | Result |
|---|---|
| Assignment 2 test suite | **68 passed, 1 skipped** (skip = optional real-encoder smoke test, `A2_RUN_SLOW=1`), ~1 s, no downloads |
| Assignment 1 regression | **13 passed** — A1 still runs untouched |
| End-to-end run (`run.py --system all`) | all 3 systems: 1,983 docs, 10 queries, **998 ranked hits** + **1,983 binary rows** each |
| Cache behavior | logs confirm `Loaded cached embeddings (validated)` and `Loaded FAISS index` on reruns — no recomputation |
| Encoder | e5-small: 117.7M params, 384-dim, ~3 min one-time CPU embedding, ~3 MB cache; XLM-R ablation verified at 768-dim using your existing HF cache (no re-download) |
| Retrieval sanity (not an official eval) | dense mean P@10 = **0.41** vs 0.36 base rate; BM25 and hybrid produce valid ranked lists |
| A1 files | `git status` shows only the new `Assignment_2/` folder + additive `.gitignore` rules |

Bugs caught and fixed during testing: a stale FAISS index surviving embedding
regeneration, and a hash-randomized fake encoder making tests seed-dependent.

## 6. Quick start

```bash
python -m pip install rank_bm25 faiss-cpu     # only new dependencies

python Assignment_2/run.py --system all       # bm25 → dense → hybrid
python Assignment_2/run.py --system bm25 --scope global --limit 3

python -m pytest Assignment_2/tests/ -v       # Assignment 2 tests
python -m pytest tests/ -v                    # Assignment 1 regression
```

Predictions land in `Assignment_2/predictions/<system>/`.
Full commands, cache management, hardware numbers, and the complete
interface reference are in **[USAGE.md](USAGE.md)**.

## 7. Where things live

```
Assignment_2/
├── README.md          # this file — what we did
├── USAGE.md           # full manual: commands, caches, interfaces, limitations
├── run.py             # unified entry point
├── configs/           # data, bm25, dense, hybrid YAML configs
├── src/               # data · preprocessing · retrieval · embeddings · utils
├── scripts/           # build index / generate embeddings / per-system runners
├── tests/             # 69 tests (no downloads, tiny synthetic corpora)
├── predictions/       # outputs: bm25/ dense/ hybrid/ + qrels.csv
├── artifacts/         # indexes, embeddings, model cache (gitignored)
└── logs/              # per-run logs (gitignored)
```

## 8. What we deliberately did NOT do

That is **Member 2's scope** (evaluation, comparisons, ablations, experiment
management, reporting). We only built the retrieval engine and left clean
hooks for them:

- every system returns the same `Hit(doc_id, score, rank)` structure;
- ranked CSVs carry full scores so any metric or cutoff can be computed
  without re-running retrieval;
- the binary export's operating point (`binary_top_k: 50`) is a documented
  naive default, intentionally left for their threshold-tuning experiments;
- config switches ready for their ablation matrix: scope, encoder, RRF `k`,
  candidate counts, top-k.

The full **handover interface** (import examples, guarantees, suggested
experiment matrix) is in [USAGE.md §17](USAGE.md).
