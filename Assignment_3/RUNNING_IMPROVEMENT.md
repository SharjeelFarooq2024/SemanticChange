# Supervised hybrid improvement

Run `python -B Assignment_3/scripts/run_hybrid_improvement.py` from the repository root. It uses the existing dependencies and frozen representations in `runs/completed`; no encoder retraining or model download is required for the experiments.

Outputs are in `Assignment_3/runs/improved_hybrid/`: training-only inner OOF selections, split definitions, fitted checkpoints, official-format predictions, paired statistics, ablations, feature coefficients, measured resource logs, a comparison figure and a JSON result summary. The original completed experiments remain unchanged.

The new hybrid keeps BM25, E5 and RRF-derived features, adds sentence-context features and a supervised linear head, and calibrates decisions from grouped OOF training predictions. Context-only and retrieval-only controls use the same selection procedure. All five seeds' settings are frozen before outer test scoring. The previously inspected outer development splits are reused for paired comparison; these results are not an untouched hidden test.

```powershell
python -B Assignment_3/scripts/infer_improved_hybrid.py --input Assignment_3/runs/completed/demo_inputs.jsonl --output Assignment_3/runs/improved_hybrid/demo_predictions.jsonl
python -B -m pytest Assignment_3/tests/test_improved_hybrid.py -q
```

The inference input needs `word` and `text`, plus an optional `sentence_id`. Known dataset texts reuse validated frozen embeddings; new texts load the pinned E5 model locally. The checkpoint supports the ten supplied target definitions. Use the model-download instructions in `RUNNING.md` if the pretrained weights are absent.

`Assignment_3/improved_hybrid_package.zip` includes the original reproducibility package contents plus the new code, configuration, checkpoints and results. Public pretrained weights remain referenced by exact revision rather than bundled. No paper, detailed report or slides are generated.
