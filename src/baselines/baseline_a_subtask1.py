from __future__ import annotations

import json
from ast import literal_eval
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from .classical_features import build_feature_frame, get_numeric_features
from .common import load_baseline_bundle, load_config, prepare_output_dirs, set_seed, write_json


def parse_sense_labels(value: Any) -> list[str]:
    """Normalize a gold label into a list without inventing a single sense."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = literal_eval(text)
        except (ValueError, SyntaxError):
            return [text]
        if isinstance(parsed, (list, tuple, set)):
            return [str(item) for item in parsed]
        return [str(parsed)]
    return [str(value)]


def cluster_purity(true_labels: list[str], predicted: np.ndarray) -> float:
    if not true_labels:
        return float("nan")
    frame = pd.DataFrame({"gold": true_labels, "cluster": predicted})
    counts = frame.groupby("cluster")["gold"].value_counts()
    return float(counts.groupby(level=0).max().sum() / len(frame))


def _valid_clustering_metrics(group: pd.DataFrame) -> dict[str, Any]:
    eligible = group["singleton_sense"].notna()
    evaluated = group.loc[eligible]
    true = evaluated["singleton_sense"].astype(str).tolist()
    predicted = evaluated["cluster"].to_numpy()
    result: dict[str, Any] = {
        "evaluation_records": len(evaluated),
        "multi_label_records_excluded": int((~eligible).sum()),
        "ari": np.nan,
        "nmi": np.nan,
        "purity": np.nan,
        "evaluation_note": "Metrics use singleton-label records only; multi-label records are preserved but excluded.",
    }
    if len(evaluated) >= 2 and len(set(true)) >= 2 and len(set(predicted)) >= 2:
        result["ari"] = adjusted_rand_score(true, predicted)
        result["nmi"] = normalized_mutual_info_score(true, predicted)
        result["purity"] = cluster_purity(true, predicted)
    else:
        result["evaluation_note"] += " At least two gold senses and two predicted clusters are required."
    return result


def cluster_target_word(group: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict[str, Any]]:
    settings = config["classical"]["subtask1"]
    method = str(settings["clustering_method"]).lower()
    if method != "kmeans":
        raise ValueError(f"Unsupported Subtask 1 clustering method: {method}. Supported method: kmeans")
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(config["classical"]["word_ngram_range"]),
        lowercase=True,
        min_df=config["classical"]["min_df"],
        max_features=config["classical"]["max_features"],
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(group["text"].fillna(""))
    used_lexical = bool(settings.get("use_lexical_features", False))
    if used_lexical:
        num_cols = get_numeric_features(config)
        features = build_feature_frame(group, config)
        lexical = StandardScaler(with_mean=False).fit_transform(features[num_cols].astype(float))
        matrix = hstack([matrix, lexical], format="csr")
    requested_clusters = int(settings["n_clusters"])
    n_clusters = min(requested_clusters, len(group))
    if n_clusters < 1:
        raise ValueError("Subtask 1 requires at least one cluster")
    if n_clusters == 1:
        assignments = np.zeros(len(group), dtype=int)
    else:
        assignments = KMeans(n_clusters=n_clusters, n_init=10, random_state=config["seed"]).fit_predict(matrix)
    result = group.copy()
    result["cluster"] = assignments
    return result, {"requested_clusters": requested_clusters, "actual_clusters": n_clusters, "tfidf_features": int(len(vectorizer.vocabulary_)), "lexical_features_used": used_lexical}


def _join_subtask1(usage: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {"sentence_id", "word", "label"}
    if not required.issubset(labels.columns):
        raise ValueError("Subtask 1 must contain sentence_id, word, and label columns")
    label_frame = labels[["sentence_id", "word", "label"]].copy()
    label_frame["sense_labels"] = label_frame["label"].map(parse_sense_labels)
    diagnostics = {
        "usage_records": len(usage),
        "label_records": len(labels),
        "malformed_or_missing_labels": int(label_frame["sense_labels"].map(len).eq(0).sum()),
        "multi_label_records": int(label_frame["sense_labels"].map(len).gt(1).sum()),
        "duplicate_label_keys": int(label_frame.duplicated(["sentence_id", "word"], keep=False).sum()),
    }
    label_frame = label_frame.drop_duplicates(["sentence_id", "word"], keep="first")
    merged = usage.merge(label_frame[["sentence_id", "word", "label", "sense_labels"]], on=["sentence_id", "word"], how="left", indicator=True)
    diagnostics["unmatched_usage_records"] = int((merged["_merge"] != "both").sum())
    merged = merged.loc[merged["_merge"] == "both"].drop(columns="_merge")
    merged["singleton_sense"] = merged["sense_labels"].map(lambda labels: labels[0] if len(labels) == 1 else np.nan)
    diagnostics["matched_records"] = len(merged)
    return merged, diagnostics


def _period_distributions(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    clusters = assignments.groupby(["word", "period_label", "cluster"], dropna=False).size().rename("count").reset_index()
    for row in clusters.to_dict("records"):
        total = int(assignments[(assignments["word"] == row["word"]) & (assignments["period_label"] == row["period_label"])].shape[0])
        rows.append({"distribution_type": "cluster", "word": row["word"], "period_label": row["period_label"], "category": row["cluster"], "count": row["count"], "proportion": row["count"] / total})
    singleton = assignments.dropna(subset=["singleton_sense"])
    senses = singleton.groupby(["word", "period_label", "singleton_sense"], dropna=False).size().rename("count").reset_index()
    for row in senses.to_dict("records"):
        total = int(singleton[(singleton["word"] == row["word"]) & (singleton["period_label"] == row["period_label"])].shape[0])
        rows.append({"distribution_type": "singleton_gold_sense", "word": row["word"], "period_label": row["period_label"], "category": row["singleton_sense"], "count": row["count"], "proportion": row["count"] / total})
    return pd.DataFrame(rows)


def _save_cluster_plot(assignments: pd.DataFrame, path: Path, config: dict | None = None) -> None:
    words = sorted(assignments["word"].unique())
    columns = 2
    rows = max(1, int(np.ceil(len(words) / columns)))
    figure, axes = plt.subplots(rows, columns, figsize=(14, 4 * rows), squeeze=False)
    ngram_range = tuple(config["classical"]["word_ngram_range"]) if config and "classical" in config else (1, 2)
    seed = int(config.get("seed", 42)) if config else 42
    for axis, word in zip(axes.flat, words):
        group = assignments.loc[assignments["word"] == word]
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, lowercase=True)
        matrix = vectorizer.fit_transform(group["text"].fillna(""))
        if matrix.shape[1] >= 2 and len(group) >= 2:
            coordinates = TruncatedSVD(n_components=2, random_state=seed).fit_transform(matrix)
        else:
            coordinates = np.column_stack([np.arange(len(group)), np.zeros(len(group))])
        axis.scatter(coordinates[:, 0], coordinates[:, 1], c=group["cluster"], cmap="tab10", s=18)
        axis.set_title(word)
        axis.set_xlabel("context component 1")
        axis.set_ylabel("context component 2")
    for axis in axes.flat[len(words):]:
        axis.set_visible(False)
    figure.suptitle("Subtask 1 per-word TF-IDF cluster projections", y=1.0)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _save_clustering_quality_plot(metrics_df: pd.DataFrame, path: Path) -> None:
    """Save a per-word ARI and purity comparison bar chart.

    ARI near 0 indicates random-chance agreement with gold senses; purity measures
    the fraction of each cluster dominated by a single gold sense. Both are plotted
    side-by-side so it is easy to see which words are cluster-separable vs. not.
    Multi-label records are excluded from evaluation and their count is annotated.
    """
    df = metrics_df.copy()
    df = df.dropna(subset=["ari"])  # words without computable metrics
    if df.empty:
        return
    words = df["word"].tolist()
    x = np.arange(len(words))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(words) * 0.55 + 1)))

    # Panel 1: ARI (can be negative)
    ari_colors = ["#2980b9" if v >= 0 else "#c0392b" for v in df["ari"]]
    axes[0].barh(x, df["ari"], color=ari_colors, edgecolor="none")
    axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(words, fontsize=9)
    axes[0].set_xlabel("Adjusted Rand Index", fontsize=10)
    axes[0].set_title("Subtask 1 ARI per word\n(singleton labels only; blue≥0, red<0)", fontsize=10)
    for xi, row in zip(x, df.itertuples()):
        axes[0].annotate(
            f"n={row.evaluation_records}",
            xy=(max(row.ari, 0) + 0.002, xi),
            va="center", fontsize=7, color="#555555",
        )

    # Panel 2: Purity (0-1)
    axes[1].barh(x, df["purity"], color="#8e44ad", edgecolor="none")
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(words, fontsize=9)
    axes[1].set_xlabel("Cluster purity", fontsize=10)
    axes[1].set_xlim(0, 1.05)
    axes[1].set_title("Subtask 1 cluster purity per word\n(proportion of dominant sense per cluster)", fontsize=10)
    for xi, row in zip(x, df.itertuples()):
        excluded = getattr(row, "multi_label_records_excluded", 0)
        if excluded > 0:
            axes[1].annotate(
                f"excl.={excluded}",
                xy=(row.purity + 0.01, xi),
                va="center", fontsize=7, color="#888888",
            )

    fig.suptitle(
        "Classical TF-IDF KMeans clustering quality (Subtask 1)\n"
        "Interpretation: ARI and purity are evaluated on singleton-labeled records only.",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(project_root: Path | None = None) -> dict:
    root = project_root or Path(__file__).resolve().parents[2]
    config = load_config(root)
    set_seed(int(config["seed"]))
    prepare_output_dirs(root, config)
    bundle = load_baseline_bundle(root, config)
    data, diagnostics = _join_subtask1(bundle["usage"].copy(), bundle["subtask1"])
    assignment_frames = []
    metric_rows = []
    model_details = {}
    for word, group in data.groupby("word", sort=True):
        assigned, details = cluster_target_word(group.reset_index(drop=True), config)
        assignment_frames.append(assigned)
        metric_rows.append({"word": word, "records": len(group), **_valid_clustering_metrics(assigned)})
        model_details[word] = details
    assignments = pd.concat(assignment_frames, ignore_index=True)
    output = root / config["outputs"]["classical"]
    saved = assignments.copy()
    saved["sense_labels"] = saved["sense_labels"].map(json.dumps)
    saved.to_csv(output / "subtask1_cluster_assignments.csv", index=False)
    _period_distributions(assignments).to_csv(output / "subtask1_period_distributions.csv", index=False)
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output / "subtask1_metrics.csv", index=False)
    _save_cluster_plot(assignments, output / "subtask1_cluster_plot.png", config)
    _save_clustering_quality_plot(metrics_frame, output / "subtask1_clustering_quality.png")
    write_json(output / "subtask1_metadata.json", {"seed": config["seed"], "task": "diachronic word sense induction", "representation": "per-word word-level TF-IDF with optional standardized lexical features", "clustering_method": config["classical"]["subtask1"]["clustering_method"], "requested_clusters": config["classical"]["subtask1"]["n_clusters"], "multi_label_policy": "Multi-valued labels are preserved in sense_labels and excluded from ARI, NMI, and purity; singleton labels are used only for valid evaluation.", "interpretation_warning": "Clusters are exploratory and may reflect target identity, time period, genre, corpus source, or lexical context rather than pure senses.", "data_diagnostics": diagnostics, "per_word_model_details": model_details})
    return {"metrics": metric_rows, "diagnostics": diagnostics}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))