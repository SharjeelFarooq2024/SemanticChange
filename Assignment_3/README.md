# Assignment 3: selected model and results

The selected model is `advanced_hybrid`: official F1 **0.666339 ? 0.021437** over five grouped development splits, pooled macro F1 **0.828844**, accuracy **0.844255**. The official 0.8 target remains unmet; these reused development splits do not establish hidden-test performance.

- [results/model_scores.csv](results/model_scores.csv): selected model and required baseline/ablation comparisons.
- [results/](results/): seed statistics, paired tests, per-word metrics, quantified errors, representative examples, figure and JSON findings.
- [runs/advanced_hybrid/](runs/advanced_hybrid/): five fitted model checkpoints, predictions, official JSONL, frozen selections, grouped splits, hardware and measured resource logs.
- [runs/completed/](runs/completed/): retained baseline and ablation evidence, frozen representations and comparison analyses.
- [final_package.zip](final_package.zip): one package with code, configurations, permitted local data, actual fitted models and reproducibility artifacts.
- [RUNNING.md](RUNNING.md): execution, inference and verification commands.

Inference and the default demo use the selected model. Discarded intermediate and neural pilot result artifacts, duplicate archives and superseded result tables have been removed. Baselines and ablations are retained for assignment comparisons. Training-only candidate search traces remain part of the selected system's reproducibility evidence. Historical reports and retired source remain locally outside the final package.

The official scorer is unmodified and pinned in `vendor/official/`. Official F1 and secondary metrics remain separate. No paper, detailed report or slides are produced.
