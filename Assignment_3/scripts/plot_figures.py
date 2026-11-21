"""Generate publication-quality figures for Assignment 3 - Part 1.

Produces 5 high-resolution (300 DPI) figures for the research report:
1. 01_model_benchmark_official_f1.png: Model comparison with 95% bootstrap error bars.
2. 02_threshold_tuning_tradeoff.png: Threshold cutoff K sweep showing train vs test performance.
3. 03_per_word_f1_heatmap.png: Per-word performance breakdown across all systems.
4. 04_rrf_k_sensitivity.png: Sensitivity of IR ranking metrics to RRF smoothing parameter k.
5. 05_paired_differences_forest_plot.png: Forest plot of paired differences with 95% CIs.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
A3_ROOT = REPO_ROOT / "Assignment_3"
TABLES_DIR = A3_ROOT / "outputs" / "tables"
FIG_DIR = A3_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})


def plot_model_benchmark() -> None:
    t1 = pd.read_csv(TABLES_DIR / "table1_multi_seed_subtask2_benchmark.csv")
    t1 = t1[t1["system"] != "majority"].sort_values("mean_official_f1", ascending=True)

    names = t1["display_name"].tolist()
    means = t1["mean_official_f1"].to_numpy()
    err_lo = means - t1["ci95_off_lo"].to_numpy()
    err_hi = t1["ci95_off_hi"].to_numpy() - means
    errors = np.vstack([err_lo, err_hi])

    colors = []
    for phase in t1["phase"]:
        if "Pretrained" in phase:
            colors.append("#2980b9")
        elif "Ablation" in phase:
            colors.append("#8e44ad")
        elif "Classical" in phase:
            colors.append("#27ae60")
        elif "Assignment 2" in phase:
            colors.append("#e67e22")
        else:
            colors.append("#7f8c8d")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    y_pos = np.arange(len(names))
    ax.barh(y_pos, means, xerr=errors, capsize=4, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Official Subtask 2 Metric ($F_{1,\\mathrm{official}}$)", fontweight="bold")
    ax.set_title("Subtask 2 System Benchmark Across 5 Random Seeds (Mean $\\pm$ 95% Bootstrap CI)", fontweight="bold", pad=12)
    ax.set_xlim(0, 0.55)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2980b9", edgecolor="black", label="Pretrained (Frozen XLM-R)"),
        Patch(facecolor="#e67e22", edgecolor="black", label="Assignment 2 Retrieval"),
        Patch(facecolor="#27ae60", edgecolor="black", label="Classical ML"),
        Patch(facecolor="#8e44ad", edgecolor="black", label="Morphological Ablation"),
        Patch(facecolor="#7f8c8d", edgecolor="black", label="Reference / Baseline"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_model_benchmark_official_f1.png", dpi=300)
    plt.close(fig)
    print("Saved 01_model_benchmark_official_f1.png")


def plot_threshold_tuning() -> None:
    # Plot tuning curve for BM25, Dense, Hybrid from Assignment 2 evaluation sweeps
    a2_eval = REPO_ROOT / "Assignment_2" / "outputs" / "evaluation"
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    styles = {"sweep_bm25.csv": ("BM25", "#d35400", "--"),
              "sweep_dense.csv": ("Dense (E5-small)", "#2980b9", "-."),
              "sweep_hybrid.csv": ("Hybrid (RRF)", "#27ae60", "-")}

    for fname, (label, color, ls) in styles.items():
        fpath = a2_eval / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            k_col = [c for c in df.columns if "k" in c.lower()][0]
            f1_col = [c for c in df.columns if "f1" in c.lower()][0]
            ax.plot(df[k_col], df[f1_col], label=label, color=color, linestyle=ls, linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Rank Decision Cutoff ($K$)", fontweight="bold")
    ax.set_ylabel("Macro-$F_1$ Score", fontweight="bold")
    ax.set_title("Operating Point Sensitivity Across Decision Cutoffs $K \\in [5, 100]$", fontweight="bold", pad=10)
    ax.axvline(50, color="gray", linestyle=":", linewidth=1, label="Naive $K=50$ Baseline")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_threshold_tuning_tradeoff.png", dpi=300)
    plt.close(fig)
    print("Saved 02_threshold_tuning_tradeoff.png")


def plot_per_word_heatmap() -> None:
    pw_path = TABLES_DIR / "table5_per_word_robustness.csv"
    if not pw_path.exists():
        return
    pw = pd.read_csv(pw_path)
    # Focus on key systems
    key_sys = [
        "majority", "all_positive", "tfidf_logistic_regression",
        "word_char_tfidf_linear_svm", "frozen_xlmr_logistic_regression",
        "bm25_train_tuned", "dense_train_tuned", "hybrid_train_tuned",
    ]
    sub = pw[pw["system"].isin(key_sys)].copy()
    sub_map = {
        "majority": "Majority (always 0)",
        "all_positive": "Always 1",
        "tfidf_logistic_regression": "TF-IDF + LR",
        "word_char_tfidf_linear_svm": "Word+char SVM",
        "frozen_xlmr_logistic_regression": "Frozen XLM-R + LR",
        "bm25_train_tuned": "BM25 (K*)",
        "dense_train_tuned": "Dense (K*)",
        "hybrid_train_tuned": "Hybrid (K*)",
    }
    sub["system_name"] = sub["system"].map(sub_map)
    piv = sub.pivot(index="word", columns="system_name", values="mean")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "Mean Official $F_1$"},
                linewidths=0.5, linecolor="gray", ax=ax)
    ax.set_title("Subtask 2 Per-Word Performance ($F_1$) Across Target Lemmas", fontweight="bold", pad=12)
    ax.set_xlabel("System", fontweight="bold")
    ax.set_ylabel("Target Word Lemma", fontweight="bold")
    plt.xticks(rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_per_word_f1_heatmap.png", dpi=300)
    plt.close(fig)
    print("Saved 03_per_word_f1_heatmap.png")


def plot_rrf_sensitivity() -> None:
    a4 = pd.read_csv(TABLES_DIR / "table4_ablation_summary.csv")
    rf = a4[a4["ablation_axis"] == "RRF Smoothing k"].copy()
    if rf.empty:
        return
    rf["k"] = rf["variant"].str.extract(r"k=(\d+)").astype(int)
    rf = rf.sort_values("k")

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
    ax.plot(rf["k"], rf["mrr"], marker="s", label="MRR", color="#2980b9", linewidth=2)
    ax.plot(rf["k"], rf["ndcg@10"], marker="o", label="nDCG@10", color="#27ae60", linewidth=2)
    ax.plot(rf["k"], rf["map"], marker="^", label="MAP", color="#e67e22", linewidth=2)

    ax.set_xlabel("RRF Smoothing Parameter $k$", fontweight="bold")
    ax.set_ylabel("Ranking Metric Score", fontweight="bold")
    ax.set_title("Sensitivity of Information Retrieval Quality to RRF Parameter $k$", fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="center right")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_rrf_k_sensitivity.png", dpi=300)
    plt.close(fig)
    print("Saved 04_rrf_k_sensitivity.png")


def plot_forest_paired_differences() -> None:
    t3 = pd.read_csv(TABLES_DIR / "table3_paired_statistical_significance.csv")
    t3_off = t3[t3["metric"] == "official_f1"].sort_values("mean_diff", ascending=True)

    comps = t3_off["comparison"].tolist()
    diffs = t3_off["mean_diff"].to_numpy()
    los = t3_off["ci95_lo"].to_numpy()
    his = t3_off["ci95_hi"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    y_pos = np.arange(len(comps))

    for y, d, lo, hi in zip(y_pos, diffs, los, his):
        c = "#27ae60" if lo > 0 else ("#c0392b" if hi < 0 else "#2980b9")
        ax.plot([lo, hi], [y, y], color=c, linewidth=2.5)
        ax.plot([lo, lo], [y - 0.15, y + 0.15], color=c, linewidth=1.5)
        ax.plot([hi, hi], [y - 0.15, y + 0.15], color=c, linewidth=1.5)
        ax.plot(d, y, "o", color="black", markersize=6)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comps)
    ax.set_xlabel("Mean Paired Difference in Official $F_1$ ($\\Delta = M_A - M_B$)", fontweight="bold")
    ax.set_title("Paired Difference Forest Plot (95% Bootstrap Confidence Intervals)", fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    ax.text(0.03, 0.08, "Green: CI > 0 ($A$ outperforms $B$)\nBlue: CI spans 0 (Statistically indistinguishable)",
            transform=ax.transAxes, fontsize=8.5, bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.8))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_paired_differences_forest_plot.png", dpi=300)
    plt.close(fig)
    print("Saved 05_paired_differences_forest_plot.png")


def main() -> None:
    plot_model_benchmark()
    plot_threshold_tuning()
    plot_per_word_heatmap()
    plot_rrf_sensitivity()
    plot_forest_paired_differences()
    print("All 5 publication figures successfully generated!")


if __name__ == "__main__":
    main()
