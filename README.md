# Semantic Change Across Multiple Time Periods

This project is the EDA and foundation layer for the SemEval-2027 Task 3 project on semantic change across multiple time periods.

## Scope

- Official development dataset: Swedish only
- Primary focus for Assignment 1: data understanding, EDA, and reusable preprocessing foundations
- Synthetic English data is auxiliary and not mixed into the official Swedish development analysis
- Final model work is intentionally deferred until the foundation is validated

## Directory structure

- `data/raw/dev/SV/`: official Swedish development data
- `data/raw/external/synthetic_english/`: auxiliary external dataset, not used in the official SV EDA
- `data/processed/eda/`: processed artifacts and cached embeddings
- `outputs/eda/`: tables, figures, PDF report, and findings
- `src/eda/`: reusable EDA and pipeline modules
- `report/assignment_1/`: report scaffolding for the assignment

## Installation

```bash
python -m pip install -r requirements.txt
```

## Reproducible EDA pipeline

From the project root:

```bash
python -m src.eda.run_eda
```

## Outputs

- `outputs/eda/tables/`: summary CSV tables
- `outputs/eda/figures/`: selected EDA plots
- `outputs/eda/eda_report.pdf`: combined PDF of the final EDA figures
- `outputs/eda/eda_findings.md`: structured observation-to-interpretation-to-modeling-implication findings

## Reproducibility notes

- Random seed is fixed to `42` wherever applicable.
- Model cache is stored under `data/processed/eda/embeddings`.
- Raw files under `data/raw/` are never modified.
- All processed artifacts are derived in `data/processed/`.

## Official task framing

SemEval-2027 Task 3 is semantic change across multiple time periods, with two main subtasks:

1. Subtask 1: diachronic word sense induction and temporal sense dynamics
2. Subtask 2: hypothesis-driven sense detection with binary usage labels

The current project focuses on the data understanding and EDA foundation for Swedish development data before later baseline work.
