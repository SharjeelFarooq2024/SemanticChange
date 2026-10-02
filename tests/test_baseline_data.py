from pathlib import Path

import pandas as pd

from src.baselines.common import join_subtask2_labels, load_baseline_bundle, load_config
from src.baselines.baseline_a_subtask1 import cluster_target_word, parse_sense_labels


ROOT = Path(__file__).resolve().parents[1]


def test_repository_bundle_has_expected_inputs():
    config = load_config(ROOT)
    bundle = load_baseline_bundle(ROOT, config)
    assert len(bundle["usage"]) > 0
    assert {"sentence_id", "text", "word", "period_label"}.issubset(bundle["usage"].columns)


def test_subtask2_join_reports_unmatched_and_malformed_rows():
    usage = pd.DataFrame({"sentence_id": ["a", "b"], "text": ["x", "y"]})
    labels = pd.DataFrame({"sentence_id": ["a", "missing", "bad"], "label": [1, 0, "unknown"]})
    merged, diagnostics = join_subtask2_labels(usage, labels)
    assert len(merged) == 1
    assert diagnostics["unmatched_usage_records"] == 1
    assert diagnostics["malformed_labels"] == 1


def test_subtask1_multi_valued_labels_are_preserved():
    assert parse_sense_labels([1, 2]) == ["1", "2"]
    assert parse_sense_labels("[3, 4]") == ["3", "4"]


def test_subtask1_clustering_is_per_word_and_deterministic():
    config = load_config(ROOT)
    frame = pd.DataFrame({
        "word": ["a"] * 4,
        "text": ["red house", "red home", "blue car", "blue vehicle"],
        "pos": ["NN"] * 4,
        "period_label": ["1900-1901", "1900-1901", "2000-2001", "2000-2001"],
        "sense_labels": [["1"], ["1"], ["2"], ["2"]],
        "singleton_sense": ["1", "1", "2", "2"],
    })
    first, _ = cluster_target_word(frame, config)
    second, _ = cluster_target_word(frame, config)
    assert first["period_label"].tolist() == frame["period_label"].tolist()
    assert first["cluster"].tolist() == second["cluster"].tolist()


def test_unified_subtask2_schema_and_missing_output_report():
    from src.baselines.evaluate import SUBTASK2_COLUMNS, build_subtask2_comparison
    config = load_config(ROOT)
    comparison, missing = build_subtask2_comparison(ROOT, config)
    assert list(comparison.columns) == SUBTASK2_COLUMNS
    assert "classical" not in missing
    assert set(comparison["model"]) >= {
        "majority",
        "tfidf_logistic_regression",
        "word_char_tfidf_linear_svm",
        "eda_features_logistic_regression",
    }


def test_classical_comparison_is_rankable_by_macro_f1():
    from src.baselines.evaluate import build_subtask2_comparison
    config = load_config(ROOT)
    comparison, _ = build_subtask2_comparison(ROOT, config)
    ranked = comparison.sort_values("macro_f1", ascending=False)
    assert ranked.iloc[0]["macro_f1"] >= ranked.iloc[-1]["macro_f1"]


def test_valid_embedding_cache_is_reused(tmp_path):
    import json
    import pickle
    import numpy as np
    from src.baselines.common import _cache_paths, _fingerprint_usage, load_frozen_usage_embeddings

    config = {
        "seed": 42,
        "pretrained": {
            "model_name": "FacebookAI/xlm-roberta-base",
            "max_length": 512,
            "pooling": "mean",
            "cache_dir": "cache",
            "batch_size": 2,
            "device": "cpu",
        },
    }
    usage = pd.DataFrame({
        "word": ["kvinna"],
        "sentence_id": ["a"],
        "period_label": ["1880-1883"],
        "year": ["1883"],
        "text": ["En kvinna stod där."],
    })
    contexts = ["En kvinna stod där."]
    cache_path, metadata_path = _cache_paths(tmp_path, config)
    cache_path.parent.mkdir(parents=True)
    cached = usage.copy()
    cached["embedding"] = [np.array([1.0, 2.0], dtype=np.float32)]
    cached["context_used"] = contexts
    with cache_path.open("wb") as handle:
        pickle.dump(cached, handle)
    metadata_path.write_text(json.dumps({
        "fingerprint": _fingerprint_usage(usage, contexts),
        "model_name": config["pretrained"]["model_name"],
        "max_length": 512,
        "pooling": "mean",
    }), encoding="utf-8")
    result, metadata = load_frozen_usage_embeddings(usage, config, tmp_path)
    assert metadata["cache_used"] is True
    assert result.iloc[0]["embedding"].tolist() == [1.0, 2.0]


def test_missing_model_dependencies_raise_clear_error(tmp_path, monkeypatch):
    import pytest
    from src.baselines.common import BaselineDataError, load_frozen_usage_embeddings
    import src.eda.embedding_analysis as ea

    config = {
        "seed": 42,
        "pretrained": {
            "model_name": "FacebookAI/xlm-roberta-base",
            "max_length": 512,
            "pooling": "mean",
            "cache_dir": "cache",
            "batch_size": 2,
            "device": "cpu",
        },
    }
    usage = pd.DataFrame({
        "word": ["kvinna"],
        "sentence_id": ["a"],
        "period_label": ["1880-1883"],
        "year": ["1883"],
        "text": ["En kvinna stod där."],
    })
    monkeypatch.setattr(ea, "load_model", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Mock connection error")))
    with pytest.raises(BaselineDataError, match="XLM-R"):
        load_frozen_usage_embeddings(usage, config, tmp_path)