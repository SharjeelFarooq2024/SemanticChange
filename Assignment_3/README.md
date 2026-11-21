# Assignment 3: Experiments, Analysis, and Research Paper

**SemEval 2027 / ICASSP 2027 Shared Task / Challenge Project**  
*Part 1: Completed Experiments & Results and Statistical Analysis (Sections 1–2)*  
**Supervision:** Dr. Mehwish Fatima  

---

## Overview

Assignment 3 brings the semantic change detection shared task project to empirical completion. This directory contains the complete implementation, execution scripts, trained checkpoints, prediction records, and statistical analysis for **Part 1 (Sections 1 and 2)**:
- **Section 1: Completed Experiments**: End-to-end multi-seed execution of all comparison baselines (majority, all-positive, classical ML, frozen XLM-R) and proposed retrieval systems (BM25, E5 Dense, Hybrid RRF) under strict leakage-free evaluation protocols.
- **Section 2: Results and Statistical Analysis**: Rigorous quantification of system performance using the Official Subtask 2 metric ($F_{1,\text{official}}$), pooled macro-$F_1$, accuracy, precision, recall, and IR ranking metrics (MRR, MAP, nDCG@10, Recall@50), paired hypothesis testing, and controlled ablations.

---

## Directory Structure

```
Assignment_3/
├── checkpoints/              # Serialized fitted pipelines (joblib) & optimal thresholds
│   ├── optimal_thresholds.json
│   └── seed_{seed}_{model}.joblib  (45 checkpoint files across 5 seeds)
├── logs/                     # Execution and hardware compute cost logs
│   ├── experiment_execution.log
│   └── hardware_and_compute_cost_log.json
├── outputs/
│   ├── figures/              # Publication-quality 300 DPI figures
│   │   ├── 01_model_benchmark_official_f1.png
│   │   ├── 02_threshold_tuning_tradeoff.png
│   │   ├── 03_per_word_f1_heatmap.png
│   │   ├── 04_rrf_k_sensitivity.png
│   │   └── 05_paired_differences_forest_plot.png
│   └── tables/               # Result tables (CSV & LaTeX .tex formats)
│       ├── table1_multi_seed_subtask2_benchmark.csv
│       ├── table2_retrieval_ranking_metrics.csv
│       ├── table3_paired_statistical_significance.csv
│       ├── table4_ablation_summary.csv
│       ├── table5_per_word_robustness.csv
│       ├── table6_per_period_robustness.csv
│       └── table7_hardware_and_compute_cost.csv
├── predictions/              # Saved test predictions per seed and unified CSV
│   ├── all_predictions.csv   (31,760 total test prediction instances)
│   └── seed_{seed}/          # Per-model test prediction files
├── scripts/                  # Executable Python scripts
│   ├── run_part1_experiments.py
│   ├── plot_figures.py
│   └── make_latex_tables.py
├── EXPERIMENTS_AND_STATISTICS_REPORT.md  # Full academic research report
└── README.md                 # This file
```

---

## Quickstart & Reproducibility Guide

### 1. Environment Activation
```powershell
# From repository root
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
```

### 2. Run the Full Multi-Seed Experiment Suite
Executes all models across 5 random seeds (`[42, 100, 2023, 2026, 777]`), tunes cutoffs strictly on training partitions, computes 95% bootstrap confidence intervals ($N_{\text{boot}}=1,000$), conducts hypothesis tests, and saves all outputs:
```powershell
python Assignment_3/scripts/run_part1_experiments.py
```

### 3. Generate Publication Figures & LaTeX Tables
```powershell
python Assignment_3/scripts/plot_figures.py
python Assignment_3/scripts/make_latex_tables.py
```

---

## Key Empirical Findings (Sections 1 & 2)

1. **Top Subtask 2 System**:
   - **Frozen XLM-R + LR**: Official $F_1 = 0.489 \pm 0.022$ ($95\%\text{ CI: } [0.473, 0.507]$).
   - **Dense Retriever (Tuned $K^*$)**: Official $F_1 = 0.448 \pm 0.016$ ($95\%\text{ CI: } [0.434, 0.459]$).
   - **Hybrid Retriever (Tuned $K^*$)**: Official $F_1 = 0.418 \pm 0.029$ ($95\%\text{ CI: } [0.395, 0.440]$).
2. **Information Retrieval Ranking (Query Level)**:
   - **Hybrid RRF ($k=60$)** achieves the highest ranking quality: $\text{MRR} = 0.692$ and $\text{nDCG@10} = 0.490$, significantly outperforming pure BM25 ($\text{MRR} = 0.618, \text{nDCG@10} = 0.383$, $p = 0.0094$).
3. **Threshold Tuning**:
   - Tuning decision cutoff $K^*$ on the training split boosts Official $F_1$ by $+0.121$ for Dense ($p = 0.0007$) and $+0.106$ for Hybrid ($p = 0.0044$) over naive $K=50$.
4. **Controlled Ablations**:
   - Per-word candidate restriction is essential for lexical retrieval: removing it collapses BM25 MRR from $0.618$ to $0.207$.
   - Morphological masking forces contextual learning, boosting linear classifier official $F_1$ from $0.341$ to $0.460$.

For detailed analysis, refer to [EXPERIMENTS_AND_STATISTICS_REPORT.md](file:///c:/Users/DAR/Desktop/SemanticChange/Assignment_3/EXPERIMENTS_AND_STATISTICS_REPORT.md).
