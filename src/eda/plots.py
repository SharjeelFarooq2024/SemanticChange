from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10


def save_figure(fig, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_dataset_composition(usage_df: pd.DataFrame, output_path: str | Path):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    word_counts = usage_df["word"].value_counts().sort_values(ascending=False)
    pos_counts = usage_df["pos"].value_counts().sort_values(ascending=False)
    sns.barplot(x=word_counts.index, y=word_counts.values, ax=axes[0], palette="viridis")
    axes[0].set_title("Usage counts by target word")
    axes[0].set_xlabel("Target word")
    axes[0].set_ylabel("Usage count")
    axes[0].tick_params(axis="x", rotation=45)
    sns.barplot(x=pos_counts.index, y=pos_counts.values, ax=axes[1], palette="magma")
    axes[1].set_title("Usage counts by POS")
    axes[1].set_xlabel("POS")
    axes[1].set_ylabel("Usage count")
    return save_figure(fig, output_path)


def plot_temporal_distribution(usage_df: pd.DataFrame, output_path: str | Path):
    set_style()
    period_counts = usage_df.groupby("period_label").size().reset_index(name="usage_count")
    period_counts = period_counts.sort_values("period_label")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(period_counts["period_label"], period_counts["usage_count"], marker="o", linewidth=2)
    ax.set_title("Usage counts across time periods")
    ax.set_xlabel("Period")
    ax.set_ylabel("Usage count")
    ax.tick_params(axis="x", rotation=45)
    return save_figure(fig, output_path)


def plot_word_usage_distribution(usage_df: pd.DataFrame, output_path: str | Path):
    set_style()
    counts = usage_df["word"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=counts.values, y=counts.index, orient="h", palette="viridis", ax=ax)
    ax.set_title("Usage counts per target word")
    ax.set_xlabel("Usage count")
    ax.set_ylabel("Target word")
    return save_figure(fig, output_path)


def plot_context_length(usage_df: pd.DataFrame, output_path: str | Path):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(usage_df["text_length_chars"].dropna(), bins=30, ax=axes[0], color="steelblue")
    axes[0].set_title("Sentence character length distribution")
    axes[0].set_xlabel("Characters")
    axes[0].set_ylabel("Count")
    sns.histplot(usage_df["token_count"].dropna(), bins=30, ax=axes[1], color="darkorange")
    axes[1].set_title("Whitespace token count distribution")
    axes[1].set_xlabel("Tokens")
    axes[1].set_ylabel("Count")
    return save_figure(fig, output_path)


def plot_lexical_characteristics(usage_df: pd.DataFrame, output_path: str | Path):
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.boxplot(data=usage_df, x="word", y="type_token_ratio", ax=axes[0, 0])
    axes[0, 0].set_title("Type-token ratio by target word")
    axes[0, 0].set_xlabel("Target word")
    axes[0, 0].set_ylabel("TTR")
    axes[0, 0].tick_params(axis="x", rotation=45)
    sns.boxplot(data=usage_df, x="word", y="avg_token_length", ax=axes[0, 1])
    axes[0, 1].set_title("Average token length by target word")
    axes[0, 1].set_xlabel("Target word")
    axes[0, 1].set_ylabel("Characters")
    axes[0, 1].tick_params(axis="x", rotation=45)
    sns.boxplot(data=usage_df, x="word", y="stopword_ratio", ax=axes[1, 0])
    axes[1, 0].set_title("Stopword ratio by target word")
    axes[1, 0].set_xlabel("Target word")
    axes[1, 0].set_ylabel("Stopword ratio")
    axes[1, 0].tick_params(axis="x", rotation=45)
    sns.boxplot(data=usage_df, x="word", y="punctuation_ratio", ax=axes[1, 1])
    axes[1, 1].set_title("Punctuation density by target word")
    axes[1, 1].set_xlabel("Target word")
    axes[1, 1].set_ylabel("Punctuation ratio")
    axes[1, 1].tick_params(axis="x", rotation=45)
    return save_figure(fig, output_path)


def plot_embedding_structure(pca_df: pd.DataFrame, output_path: str | Path):
    set_style()
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("tab10", n_colors=pca_df["word"].nunique())
    for i, (word, group) in enumerate(pca_df.groupby("word")):
        ax.scatter(group["pca_1"], group["pca_2"], s=20, label=word, color=palette[i])
    ax.set_title("PCA projection of contextual embeddings")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Target word")
    return save_figure(fig, output_path)


def plot_temporal_embedding_similarity(similarity_df: pd.DataFrame, output_path: str | Path):
    set_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    similarity_df = similarity_df.sort_values(["word", "period_from"])
    for word, group in similarity_df.groupby("word"):
        ax.plot(group["period_to"], group["similarity"], marker="o", label=word)
    ax.set_title("Adjacent-period embedding similarity by target word")
    ax.set_xlabel("Next period")
    ax.set_ylabel("Cosine similarity")
    ax.legend(loc="best")
    return save_figure(fig, output_path)


def plot_definition_usage_similarity(sim_df: pd.DataFrame, output_path: str | Path):
    set_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    if not sim_df.empty:
        labels = sim_df["label"].astype(str)
        sns.histplot(sim_df["cosine_similarity"], bins=25, ax=ax)
        ax.set_title("Definition-to-usage cosine similarity distribution")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Count")
    else:
        ax.text(0.5, 0.5, "No definition-usage similarity data available", ha="center", va="center")
        ax.set_axis_off()
    return save_figure(fig, output_path)


def plot_period_word_heatmap(usage_df: pd.DataFrame, output_path: str | Path):
    set_style()
    pivot = (
        usage_df.groupby(["word", "period_label"])
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt="d", linewidths=0.3, ax=ax)
    ax.set_title("Usage counts by target word and period")
    ax.set_xlabel("Period")
    ax.set_ylabel("Target word")
    return save_figure(fig, output_path)


def plot_period_lexical_diversity(usage_df: pd.DataFrame, output_path: str | Path):
    set_style()
    period_stats = (
        usage_df.groupby("period_label")
        .agg(
            avg_ttr=("type_token_ratio", "mean"),
            avg_char_len=("text_length_chars", "mean"),
            avg_token_len=("avg_token_length", "mean"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.lineplot(data=period_stats, x="period_label", y="avg_ttr", marker="o", ax=axes[0])
    axes[0].set_title("Mean type-token ratio by period")
    axes[0].set_xlabel("Period")
    axes[0].set_ylabel("Mean TTR")
    axes[0].tick_params(axis="x", rotation=45)

    sns.lineplot(data=period_stats, x="period_label", y="avg_char_len", marker="o", color="darkorange", ax=axes[1])
    axes[1].set_title("Mean context length by period")
    axes[1].set_xlabel("Period")
    axes[1].set_ylabel("Mean characters")
    axes[1].tick_params(axis="x", rotation=45)
    return save_figure(fig, output_path)


def build_eda_pdf(fig_paths: list[str | Path], pdf_path: str | Path):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    for fig_path in fig_paths:
        image = ImageReader(str(fig_path))
        img_width, img_height = image.getSize()
        ratio = min((width - 40) / img_width, (height - 40) / img_height)
        new_width = img_width * ratio
        new_height = img_height * ratio
        x = (width - new_width) / 2
        y = (height - new_height) / 2
        c.drawImage(image, x, y, width=new_width, height=new_height)
        c.showPage()
    c.save()
    return str(pdf_path)
