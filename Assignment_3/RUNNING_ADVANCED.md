# Advanced hybrid experiments

Run `python -B Assignment_3/scripts/run_advanced_hybrid.py` from the repository root. It compares word-class-balanced linear models, per-lemma experts, RBF classifiers, gradient boosting and predefined ensembles using the existing frozen BM25/E5/XLM-R views. Scaling, text SVD, vocabulary, heads and thresholds are fitted using training data only. All five seeds are selected before any new test score is computed.

Results, checkpoints, split definitions, OOF selection traces and measured resources are in `Assignment_3/runs/advanced_hybrid/`. `result_summary.json` explicitly records whether the official F1 target of 0.8 was reached; accuracy and pooled macro-F1 remain separate metrics. Earlier model packages remain unchanged.

```powershell
python -B Assignment_3/scripts/infer_advanced_hybrid.py --input Assignment_3/runs/completed/demo_inputs.jsonl --output Assignment_3/runs/advanced_hybrid/demo_predictions.jsonl
python -B -m pytest Assignment_3/tests/test_advanced_hybrid.py -q
```

For the separate neural pilot, run `python -B Assignment_3/scripts/run_neural_pilot.py`. It fine-tunes the last two E5 blocks on CPU using only a split of outer training data. Its validation scores are model-selection evidence, not final test scores. The pilot compares against a matched classical control before deciding whether full multi-seed neural evaluation is warranted.

The completed advanced run reached official F1 **0.666339 ± 0.021437** over five seeds (previous hybrid: 0.652406 ± 0.026327). Pooled macro-F1 was 0.828844 and accuracy 0.844255. The official target 0.8 was not reached. The paired comparison with the previous hybrid was inconclusive (exact p=0.375; Holm-adjusted p=0.75); these reused, overlapping development splits do not establish hidden-test performance. The pilot's best validation F1 was 0.500031 versus control 0.592560, so it did not pass the predefined gate for a full neural run.

Run `python -B Assignment_3/scripts/package_advanced_trials.py` after verification to produce `Assignment_3/advanced_trials_package.zip`. This adds all five advanced checkpoints, the neural pilot checkpoint, predictions, configs, splits, measured resource logs and source to the unchanged earlier experiment package. Public pretrained weights remain external with pinned revisions. Rebuild this archive after rerunning either trial.

These are further development experiments on previously inspected data. Neither an arbitrary score target nor a high accuracy value establishes 0.8 official F1. No paper, detailed report or slides are generated.
