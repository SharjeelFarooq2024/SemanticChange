# SemEval-2027 Task 3 Subtask 2: System Evaluation, Calibration & Generalization Report

**Evaluation Harness**: `eval_harness.py` (5-fold $\times$ 5-repeat `StratifiedGroupKFold` on near-duplicate text clusters, pooled OOF across all 1,983 usages)  
**Headline Metric**: Authoritative SemEval per-word two-class macro-F1 averaged equally across the 10 target lemmata  

---

## 1. Executive Summary & Audited Benchmark Progression

Every model comparison below follows fixed, leakage-free protocols: all scalers, TF-IDF vocabularies, sample weights, and decision thresholds are fitted strictly inside training/inner-OOF folds, and model adoption requires the lower bound of the 10,000-sample paired bootstrap 95% CI to exceed `0`.

| Stage / System | Official Macro-F1 (5x5 Strict Inner-OOF) | Nested CV (5x3 Unbiased) | Pooled Macro-F1 / Acc | LOPO (10 Periods) | LOLO (10 Words) | 95% CI vs Reference | Protocol Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Old Report (`advanced_hybrid`, exact-hash)** | 0.6663 ± 0.0214 *(5 splits)* | — | 0.8288 / 0.8443 | — | — | Tied w/ SVM ($p=0.50$) | **INVALIDATED** *(Leakage)* |
| **1. `advanced_hybrid` (Near-dup 5x5 Uncal)** | 0.6057 ± 0.0032 | — | 0.8483 / 0.8524 | — | — | Failed baseline | **REJECTED** *(0% min recall on 4 words)* |
| **2. Baseline Linear SVM (Uncalibrated)** | 0.6136 ± 0.0058 | — | 0.8082 / 0.8149 | — | — | Reference | **BASELINE** |
| **3. Calibrated Baseline SVM ($s=0.50$)** | 0.6358 ± 0.0055 | **0.6268** | 0.8023 / 0.8215 | 0.6219 | 0.3958 | [+0.0100, +0.0348] vs uncal SVM | **ADOPTED** |
| **4. `Lemma_Balanced_SVM` ($s=0.50$)** | 0.6417 ± 0.0061 | — | — | — | — | [-0.0021, +0.0148] vs cal SVM | **NOT ADOPTED** *(CI includes 0)* |
| **5. `Dense_E5` Interaction Head ($s=0.50$)** | 0.6266 ± 0.0097 | — | — | — | — | [-0.0424, +0.0237] vs cal SVM | **REJECTED** |
| **6. `Dense_XLM-R` (Docs, $C=0.1, s=0.50$)** | **0.6809 ± 0.0048** *(uncal 0.6713)* | **0.6873** | **0.8346 / 0.8457** | **0.6743** | 0.3990 | **[+0.0085, +0.0800]** vs cal SVM | **ADOPTED (OFFICIAL WINNER)** |
| **7. *Exploratory:* Dual-Ctx Dense + LemmaSVM** | — | **0.7006** *(unbiased)* | — | **0.7168** | **0.4204** | [-0.0220, +0.0455] vs #6 Nested | **EXPLORATORY** *(Stronger LOPO; tied on 10-word CI)* |

---

## 2. Dataset Confounds & Generalization Reality (`data/raw/dev/SV/`)

The Swedish development set comprises **1,983 usages** across **10 target words** and **10 historical periods** (`1880-1883` to `2022-2023`, not 7). Two structural constraints govern ceiling and generalization:
1. **Severe Label Skew**: Four words have extreme minority ratios—`exportera` (**8** negatives = 5.4%, flagged unstable `<10`), `fröken` (**10** positives = 5.7%), `herre` (**20** positives = 7.1%), and `fru` (**29** negatives = 10.6%). Because official F1 averages macro-F1 equally across all 10 words, minority collapse on these 4 words caps the metric.
2. **Era-Word Asymmetry**: `skär` appears *only* in the 5 historical periods (`1880-1915`), whereas `dumpa`, `exportera`, `förort`, and `suga` appear *only* in the 5 modern periods (`2006-2023`); only `fru`, `fröken`, `herre`, `kvinna`, and `motiv` span all 10 periods.

### Shuffled-Definition Ablation (Crucial Scientific Disclosure)
Under strict 5x5 inner-OOF evaluation (`results/stage3_definition_matching_results.json`), `Dense_XLM-R` ($[|u - d|, u \odot d, \cos(u, d)]$) scored **0.6809** (Nested CV: **0.6873**), beating Calibrated Baseline SVM (**0.6358**) by **+0.0451** (95% CI: `[+0.0085, +0.0800]`, $p = 0.0082$). However, permuting definitions across the 10 words (`Dense_XLM-R_ShuffledDef_Ablation`) also scored **0.6968** (`0.6923` uncal), while single-vector `docs_only` scored `0.5837` and `cos_only` scored `0.4014`.  
**Interpretation**: Because $d_w$ is constant per lemma $w$, multiplying $u \odot d_w$ acts primarily as **word-conditioned subspace gating** (`lemma x embedding`) that enables per-lemma separation across known lemmata and improves cross-period stability (**0.6743 LOPO** vs Baseline `0.6219`), **not** zero-shot definition reasoning (**0.3990 LOLO** on unseen lemmata).

---

## 3. Per-Word Ceiling & Diagnostics (Final `Dense_XLM-R` Nested CV)

From `results/diagnostics_final_calibrated_dense_xlmr.csv` (Nested Official F1: **0.687341**, Mean AUC: **0.8200**, Mean Oracle F1 Ceiling: **0.7268**):

| Lemma | Total | Minority Support ($N<10$ Unstable) | Baseline SVM Nested F1 | Dense XLM-R Nested F1 | Gain | Oracle F1 | Minority Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `dumpa` | 145 | 45 (31.0%) | 0.6470 | **0.6824** | +0.0354 | 0.7046 | 62.2% |
| `exportera` | 149 | **8 (5.4%) [TRUE]** | 0.5576 | **0.5861** | +0.0285 | 0.7185 | 12.5% |
| `fru` | 274 | 29 (10.6%) | 0.5849 | **0.6526** | +0.0677 | 0.6816 | 31.0% |
| `fröken` | 174 | 10 (5.7%) | 0.5825 | **0.6485** | +0.0660 | 0.6865 | 30.0% |
| `förort` | 150 | 74 (49.3%) | 0.7067 | **0.7062** | -0.0005 | 0.7756 | 75.7% |
| `herre` | 283 | 20 (7.1%) | 0.5229 | **0.6585** | **+0.1356** | 0.7041 | 45.0% |
| `kvinna` | 290 | 66 (22.8%) | 0.6141 | **0.6862** | **+0.0721** | 0.7071 | 53.0% |
| `motiv` | 291 | 53 (18.2%) | 0.7371 | **0.8770** | **+0.1399** | 0.8996 | 88.7% |
| `skär` | 98 | 27 (27.6%) | 0.6781 | **0.6726** | -0.0055 | 0.6726 | 59.3% |
| `suga` | 129 | 24 (18.6%) | 0.6366 | **0.7032** | +0.0666 | 0.7174 | 58.3% |

---

## 4. How to Optimally Push Toward `0.75+` While Preserving Generalization

1. **Confirmed on CPU without overfitting**: Combining full-sentence `documents` + ±80-char target-window `classifier_contexts` with stronger regularization ($C=0.05$), per-lemma loss weights, and `0.2 * LemmaBalancedSVM` lifts Nested CV to **0.7006** and boosts **temporal LOPO generalization to 0.7168** (+9.5 points over baseline SVM) and LOLO to **0.4204**.
2. **Breaking the 0.727 frozen-sentence ceiling (`0.75–0.82`)**:
   - **Target-Span Subword Pooling**: Extract top-4-layer hidden states *only* at the `start`/`end` character offsets of the target token using `KBLab/bert-base-swedish-cased` or `xlm-roberta-base`.
   - **Cross-Encoder Sequence-Pair Fine-Tuning**: Feed `[CLS] <target> word </target> in context [SEP] definition [SEP]` with frozen lower layers (LoRA $r=8$ or top 2–4 layers, lr $2\times 10^{-5}$, per-lemma focal loss). Verify genuine definition usage via shuffled-definition ablation and LOLO.

---

## 5. One-Sentence Stage Verdicts

- **Stage 0 (Reconciliation)**: *Adopted* — Recomputing seed 42 confirmed `0.6986` (exposing `0.6789` as a docs typo), and `eval_harness.official_f1` verified $\le 1.1\times 10^{-16}$ parity with the vendored scorer.
- **Stage 1 (Measurement & Leakage)**: *Adopted* — Leak-free `near_dup` 5x5 CV showed Baseline SVM (`0.6136`) beat `advanced_hybrid` (`0.6057`), and exact-hash grouping leaked `+0.0037` to `+0.0060` F1.
- **Stage 2 (Cheap Calibration)**: *Adopted* — Strict inner-OOF `best_threshold(shrinkage=0.5)` boosted Baseline SVM to `0.6358` (`+0.0222`, 95% CI `[+0.0100, +0.0348]`, $p = 0.0002$; Nested CV `0.6268`).
- **Stage 3 (Dense Definition Head)**: *Adopted* — `Dense_XLM-R` reached `0.6809` strict 5x5 calibrated F1 (`+0.0451`, 95% CI `[+0.0085, +0.0800]`), while shuffled-definition ablation (`0.6968`) clarified it acts as word-conditioned gating.
- **Stage 5 (Generalization LOLO / LOPO)**: *Analyzed* — Across all 10 periods, LOPO improved from `0.6219` (SVM) $\rightarrow$ `0.6743` (`Dense_XLM-R`) $\rightarrow$ `0.7168` (`Exploratory Blend`), whereas LOLO remained ~`0.40–0.42` without unseen-lemma cross-attention.
- **Stage 6 (Final Nested CV)**: *Adopted* — Unbiased 5-outer $\times$ 3-inner Nested CV confirmed **0.687341** official F1 (`0.8346` pooled macro-F1, `84.57%` accuracy) for the frozen `Dense_XLM-R` system.
