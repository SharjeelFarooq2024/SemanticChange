# Running the selected final model

Run commands from the repository root with Python 3.10. Install `Assignment_3/requirements.txt`. The local `.deps` folder supplies `bcubed` on this checkout; it is also pinned in the requirements.

```powershell
# Inference defaults to the selected seed-42 checkpoint.
python -B Assignment_3/scripts/infer.py --input Assignment_3/runs/completed/demo_inputs.jsonl --output Assignment_3/results/demo_predictions.jsonl
python -B Assignment_3/scripts/demo.py

# Rebuild analysis and the single final package from saved fitted models.
python -B Assignment_3/scripts/finalize_results.py

# Evaluate the selected model with the fixed official evaluator.
python -B Assignment_3/scripts/evaluate_predictions.py --gold Assignment_3/runs/advanced_hybrid/official/seed_42/gold/SV/subtask2.jsonl --predictions Assignment_3/runs/advanced_hybrid/official/seed_42/SV_subtask2.jsonl

python -B -m pytest tests Assignment_2/tests Assignment_3/tests -q
```

Input JSONL needs `word` and `text`. Optional fields are `sentence_id`, `pos` and `period_label`; omitted metadata uses the same defaults as the fitted model. Use `--checkpoint Assignment_3/runs/advanced_hybrid/checkpoints/seed_100.joblib` to select another fitted seed explicitly. Seed 42 is the fixed demo default, not a choice made by its test score. Baseline checkpoints remain available via an explicit `--checkpoint` path.

For a complete refit, run:

```powershell
# Add --allow-download only if the pinned public pretrained weights are absent.
python -B Assignment_3/scripts/prepare_representations.py --allow-download
python -B Assignment_3/scripts/run_complete_experiments.py
python -B Assignment_3/scripts/run_advanced_hybrid.py
```

The first experiment command reproduces all required baselines and original retrieval ablations. The second fits the selected word-aware hybrid procedure over five seeds and publishes the final package. It compares word-class-balanced linear models, per-lemma experts, RBF classifiers, gradient boosting and predefined ensembles. Vocabulary, SVD, scaling, heads and thresholds are fitted inside grouped training folds. All five choices are frozen before outer evaluation. The full candidate search traces are required selection evidence, not additional final models.

Final deliverables are in `Assignment_3/results/`. The selected model's raw predictions, five checkpoints, split definitions, official-format files and measured resource logs are in `Assignment_3/runs/advanced_hybrid/`. Required baseline/ablation results and frozen caches are in `Assignment_3/runs/completed/`; they are comparison evidence. `Assignment_3/final_package.zip` includes these artifacts and their SHA-256 manifests. Discarded experimental runs and older ZIPs are excluded. Rebuild the package after creating new output files.

Public pretrained weights are external because of their size. Models are `intfloat/multilingual-e5-small` and `FacebookAI/xlm-roberta-base`, with exact revisions in `configs/complete_experiments.json`. Cache identities, model hashes and representation hashes are in `runs/completed/artifacts/`. New text requires those cached model weights; existing corpus inputs can use the packaged frozen representations.

The unmodified official scorer averages two-class macro F1 within each word, then across words. The selected model achieves **0.666339 ? 0.021437 official F1**, **0.828844 pooled macro F1**, and **0.844255 accuracy** across seeds 42, 100, 2023, 2026 and 777. Five overlapping development splits measure split sensitivity, not independent population-level replications. The data was inspected during model development; no untouched hidden-test score or statistically established superiority over the lexical SVM is claimed. Frozen label-free full-corpus retrieval references retain the documented transductive protocol.

Wall/CPU time, sampled RSS and actual artifact sizes are measured. Monetary and energy cost are unavailable. Exact-text grouping blocks duplicate contexts, while source/near-duplicate leakage and unseen-lemma generalization remain limitations. No paper, detailed report or slides are generated.
