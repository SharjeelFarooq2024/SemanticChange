from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .common import (
    join_subtask2_labels,
    load_baseline_bundle,
    load_config,
    load_frozen_usage_embeddings,
    prepare_output_dirs,
    set_seed,
    write_json,
)


def _metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def run(project_root: Path | None = None) -> dict:
    root = project_root or Path(__file__).resolve().parents[2]
    config = load_config(root)
    set_seed(int(config["seed"]))
    prepare_output_dirs(root, config)
    bundle = load_baseline_bundle(root, config)
    labeled, diagnostics = join_subtask2_labels(bundle["usage"], bundle["subtask2"])
    embeddings, embedding_metadata = load_frozen_usage_embeddings(bundle["usage"], config, root)
    data = labeled.merge(embeddings[["sentence_id", "word", "embedding"]], on=["sentence_id", "word"], how="left", validate="one_to_one")
    if data["embedding"].isna().any():
        raise ValueError("Some labeled Subtask 2 records have no matching cached embedding")
    train, test = train_test_split(data, test_size=config["evaluation"]["test_size"], random_state=config["evaluation"]["random_state"], stratify=data["label_value"])
    x_train = np.vstack(train["embedding"].map(np.asarray).to_numpy())
    x_test = np.vstack(test["embedding"].map(np.asarray).to_numpy())
    models = {
        "frozen_xlmr_logistic_regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=config["classical"]["max_iter"], class_weight=config["classical"]["class_weight"], random_state=config["seed"]))]),
        "frozen_xlmr_linear_svm": Pipeline([("scale", StandardScaler()), ("model", LinearSVC(class_weight=config["classical"]["class_weight"], random_state=config["seed"]))]),
    }
    runs = []
    predictions = []
    for name, model in models.items():
        model.fit(x_train, train["label_value"])
        prediction = model.predict(x_test)
        frame = test[["sentence_id", "word", "period_label", "label_value"]].copy()
        frame["model"] = name
        frame["prediction"] = prediction
        predictions.append(frame)
        runs.append({"model": name, "representation": "frozen_xlm_roberta_mean_pooled_embedding", "train_size": len(train), "test_size": len(test), **_metrics(test["label_value"], prediction)})
    output = root / config["outputs"]["pretrained"]
    metrics = pd.DataFrame(runs)
    metrics.to_csv(output / "subtask2_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "subtask2_predictions.csv", index=False)
    write_json(output / "subtask2_metadata.json", {**embedding_metadata, "tokenizer_name": config["pretrained"]["model_name"], "cache_dir": config["pretrained"]["cache_dir"], "frozen": True, "classifier_models": list(models), "data_diagnostics": diagnostics, "split": config["evaluation"]})
    return {"metrics": runs, "embedding_metadata": embedding_metadata, "diagnostics": diagnostics}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))