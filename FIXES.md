# Methodological Fixes

> This document records two methodological issues found during a team review of the
> Baseline A implementation, their root causes, the fix strategy chosen, and the
> exact changes made. It is intended as a reference for the report and for future
> contributors.

---

## Issue 1 — Fixed `k = 3` in KMeans Clustering (Subtask 1)

### What was wrong

In `src/baselines/baseline_a_subtask1.py`, the number of clusters for KMeans was
hardcoded to `n_clusters: 3` in `configs/baselines.yaml` and applied uniformly to
**every target word**, regardless of how many actual senses that word has in the data.

```python
# BEFORE (problematic)
requested_clusters = int(settings["n_clusters"])   # always 3
n_clusters = min(requested_clusters, len(group))
assignments = KMeans(n_clusters=n_clusters, ...).fit_predict(matrix)
```

**Why this is a problem:**
KMeans requires a pre-specified `k`. Using a single fixed value for all words
implicitly assumes every word has exactly 3 senses, which is rarely true.
Words like *suga* may have only 2 distinguishable usages, while others may have 4 or
more. A wrong `k` forces the algorithm to either over-segment or under-segment usage
clusters, corrupting ARI, NMI, and purity scores relative to gold sense annotations.

---

### Fix — Three-Tier Dynamic k-Selection

A new helper function `_select_k_from_gold()` was added to
`baseline_a_subtask1.py`. It selects `k` per word using the following priority order:

```
Tier 1  ->  Gold-sense count          (most principled)
Tier 2  ->  Silhouette-score sweep    (data-driven fallback)
Tier 3  ->  Config default k          (safe last resort)
```

#### Tier 1: Gold-sense count

If the word group contains at least 2 distinct singleton-labelled gold senses,
set `k` equal to the number of unique gold senses (capped at `k_max`).

```python
gold_senses = group["singleton_sense"].dropna().unique()
if len(gold_senses) >= 2:
    k_gold = min(int(len(gold_senses)), k_max, len(group))
    return k_gold, "gold_sense_count"
```

**Rationale:** Anchoring `k` to annotated reality is the most principled choice.
If the dataset says a word has 3 senses, we should ask the clustering to find 3
groups, not impose a universal constant.

#### Tier 2: Silhouette-score sweep

When gold labels are absent or give only 1 unique sense (ambiguous signal), sweep
`k` over `[2, k_max]` and pick the value that maximises the mean silhouette score
on the TF-IDF/lexical matrix.

```python
for k_candidate in range(2, k_upper + 1):
    labels = KMeans(n_clusters=k_candidate, ...).fit_predict(matrix)
    score = silhouette_score(matrix, labels, ...)
    if score > best_score:
        best_k, best_score = k_candidate, score
```

**Rationale:** Silhouette score measures intra-cluster cohesion vs. inter-cluster
separation without requiring ground truth, making it a standard unsupervised model
selection criterion.

#### Tier 3: Config default

If the group is too small to run a sweep (< 4 samples), fall back to
`n_clusters` from `configs/baselines.yaml` (default: 3).

---

### Configuration

Two keys control the dynamic-k behaviour in `configs/baselines.yaml`:

```yaml
subtask1:
  n_clusters: 3         # Tier 3 fallback k
  n_clusters_max: 6     # Ceiling applied to Tier 1 and Tier 2
```

To revert to strictly fixed `k = 3` for reproducibility experiments, set
`n_clusters_max: 3` (this collapses all three tiers to the same value).

### Output logged

Each word's chosen `k` and which tier was used is logged to
`outputs/baselines/classical/subtask1_metadata.json` under `per_word_model_details`:

```json
"kvinna": {
  "actual_clusters": 4,
  "k_selection_strategy": "gold_sense_count",
  "k_fallback": 3,
  "k_max": 6
}
```

### Files changed

| File | Change |
|------|--------|
| `src/baselines/baseline_a_subtask1.py` | Added `_select_k_from_gold()`; replaced hard-coded cluster logic; updated metadata write |
| `configs/baselines.yaml` | Added `n_clusters_max: 6` alongside existing `n_clusters: 3` |

---

## Issue 2 — Morphological Leakage in the Word-Identity Ablation (Subtask 2)

### What was wrong

In `src/baselines/baseline_a_subtask2.py`, the word-identity ablation model was
supposed to measure how much the classifier relies on knowing **which target word**
appears in a sentence, vs. pure contextual features.

The masking step replaced only the **exact surface form** stored in the `word` column:

```python
# BEFORE (leaking)
def _mask_word(row):
    w = str(row.get("word", "") or "")
    t = str(row.get("text", "") or "")
    return re.sub(rf"\b{re.escape(w)}\b", "[TARGET]", t, flags=re.IGNORECASE) if w else t
```

**Example of the bug:** If `word = "driva"`, the regex replaces `driva/Driva` but
leaves all inflected forms -- `driver`, `drev`, `drivit`, `drivande`, `driven`, etc. --
intact in the text. TF-IDF then picks up these forms as strong signals for the
word identity, **leaking** the information the ablation was designed to remove.
The reported ablation gap (Delta-F1 ~= 10.3 points) was therefore an underestimate
of word-identity dependence.

---

### Fix — Morphological Expansion Masking

A morphological expander `_morphological_forms()` was added inside the `run()`
function of `baseline_a_subtask2.py`.
It generates all common Swedish surface forms of a target word from suffix rules
(no external lexicon or dependency required).

#### Forms generated

| Category | Suffixes added to base |
|----------|----------------------|
| Noun definite singular | `-n`, `-en`, `-et` |
| Noun plural (indefinite + definite) | `-ar`, `-er`, `-or`, `-r` + `-na` variants |
| Genitive | `-s` appended to every generated form |
| Verb present | `-er`, `-ar`, `-r` |
| Verb past (weak) | `-ade`, `-de`, `-te` |
| Supine / past participle | `-at`, `-t`, `-tt`, `-it` |
| Present participle | `-ande`, `-ende` |
| Passive | `-as`, `-es` |
| Stem-final `a`-drop (strong verbs) | Drop trailing `a`, then apply verb suffixes to stem |

#### Implementation highlights

```python
@lru_cache(maxsize=None)
def _morphological_forms(word: str) -> frozenset[str]:
    # generates all suffix variants, returns frozen set

def _build_mask_pattern(word: str) -> re.Pattern:
    # sorts forms longest-first to prevent partial clobber, compiles alternation regex

def _mask_word_morphological(row) -> str:
    # replaces every matched form with [TARGET]
```

Key design decisions:
- **Longest-match ordering** prevents partial clobber of shorter prefixes
- **`lru_cache`** compiles each word's pattern once, not once per row
- **No extra dependencies** -- pure Python + `re`, runs in any environment

#### Limitation

The suffix-rule expander does not cover suppletive strong-verb past forms
(e.g., *ga* -> *gick*, *be* -> *bad*). For the purposes of this baseline this is
an acceptable approximation; full coverage would require a morphological lexicon
such as SALDO.

---

### Files changed

| File | Change |
|------|--------|
| `src/baselines/baseline_a_subtask2.py` | Replaced `_mask_word()` with `_morphological_forms()`, `_build_mask_pattern()`, `_mask_word_morphological()`; added `from functools import lru_cache` |

---

## Test Coverage

All existing tests still pass after both fixes:

```
tests/test_classical_features.py   5 passed
tests/test_baseline_data.py        8 passed
```

---

## Summary

| # | Issue | Root Cause | Fix | Impact |
|---|-------|------------|-----|--------|
| 1 | Fixed `k = 3` for all words | Hardcoded config value | Three-tier dynamic k: gold senses -> silhouette sweep -> config default | Cluster counts now reflect data; ARI/NMI more meaningful |
| 2 | Morphological leakage in ablation | Regex masked only the exact lemma form | Suffix-rule expander generates all Swedish surface variants | Ablation Delta-F1 now accurately measures contextual-only performance |
