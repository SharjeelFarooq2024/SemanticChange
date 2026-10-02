from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .baseline_a_subtask1 import _join_subtask1, _period_distributions, _valid_clustering_metrics
from .common import (
    load_baseline_bundle,
    load_config,
    load_frozen_usage_embeddings,
    prepare_output_dirs,
    set_seed,
    write_json,
)


def run(project_root: Path | None = None) -> dict:
    root = project_root or Path(__file__).resolve().parents[2]
    config = load_config(root)
    set_seed(int(config["seed"]))
    prepare_output_dirs(root, config)
    bundle = load_baseline_bundle(root, config)
    data, diagnostics = _join_subtask1(bundle["usage"].copy(), bundle["subtask1"])
    embeddings, embedding_metadata = load_frozen_usage_embeddings(bundle["usage"], config, root)
    data = data.merge(embeddings[["sentence_id", "word", "embedding"]], on=["sentence_id", "word"], how="left", validate="one_to_one")
    if data["embedding"].isna().any():
        raise ValueError("Some Subtask 1 records have no matching cached embedding")
    settings = config["classical"]["subtask1"]
    assignments = []
    metrics = []
    for word, group in data.groupby("word", sort=True):
        matrix = np.vstack(group["embedding"].map(np.asarray).to_numpy())
        n_clusters = min(int(settings["n_clusters"]), len(group))
        cluster = np.zeros(len(group), dtype=int) if n_clusters == 1 else KMeans(n_clusters=n_clusters, n_init=10, random_state=config["seed"]).fit_predict(matrix)
        result = group.copy()
        result["cluster"] = cluster
        assignments.append(result)
        metrics.append({"word": word, "records": len(result), **_valid_clustering_metrics(result)})
    result = pd.concat(assignments, ignore_index=True)
    output = root / config["outputs"]["pretrained"]
    saved = result.copy()
    saved["sense_labels"] = saved["sense_labels"].map(json.dumps)
    saved.to_csv(output / "subtask1_cluster_assignments.csv", index=False)
    _period_distributions(result).to_csv(output / "subtask1_period_distributions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "subtask1_metrics.csv", index=False)
    write_json(output / "embedding_metadata.json", {**embedding_metadata, "tokenizer_name": config["pretrained"]["model_name"], "cache_dir": config["pretrained"]["cache_dir"], "frozen": True, "clustering_method": "kmeans on mean-pooled embeddings", "requested_clusters": settings["n_clusters"], "multi_label_policy": "Multi-valued labels are preserved and excluded from ARI, NMI, and purity.", "data_diagnostics": diagnostics})
    return {"metrics": metrics, "embedding_metadata": embedding_metadata, "diagnostics": diagnostics}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))