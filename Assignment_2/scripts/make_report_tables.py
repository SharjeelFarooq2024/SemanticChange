"""Generate LaTeX tables for the Assignment 2 report directly from the result CSVs.

Avoids hand-transcribing numbers. Writes report/assignment_2/tables/*.tex.

Usage (from SemanticChange/):
    python Assignment_2/scripts/make_report_tables.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "Assignment_2" / "outputs" / "evaluation"
OUT = ROOT / "report" / "assignment_2" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

NAMES = {
    "eda_features_logistic_regression": "TF-IDF + features + LR",
    "word_char_tfidf_linear_svm": "Word+char TF-IDF + SVM",
    "tfidf_logistic_regression": "TF-IDF + LR",
    "eda_features_random_forest": "TF-IDF + features + RF",
    "frozen_xlmr_logistic_regression": "Frozen XLM-R + LR",
    "frozen_xlmr_linear_svm": "Frozen XLM-R + SVM",
    "ablated_no_word_id_logistic_regression": "Ablated LR (no word id)",
    "tfidf_complement_nb": "TF-IDF + Complement NB",
    "majority": "Majority (always 0)",
    "all_positive": "Always 1",
    "bm25": "BM25",
    "dense": "Dense (E5-small)",
    "hybrid": "Hybrid (RRF)",
    "dense_xlmr_base": "Dense (XLM-R base)",
}


def ci(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]"


def grid(tex: str) -> str:
    """Boxed table: vertical column lines, full grid, bold header row."""
    import re
    lines = tex.strip().split("\n")
    m = re.match(r"\\begin\{tabular\}\{([^}]*)\}", lines[0])
    spec = "|" + "|".join(m.group(1)) + "|"
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\hline"]
    header_done = False
    for ln in lines[1:-1]:
        if ln.strip() in ("\\toprule", "\\midrule", "\\bottomrule", "\\hline"):
            continue
        if not header_done:
            cells = [c.strip() for c in ln.rstrip("\\").split("&")]
            cells = [c if c.startswith("\\textbf") else f"\\textbf{{{c}}}" for c in cells]
            ln = " & ".join(cells) + " \\\\"
            header_done = True
        out += [ln, "\\hline"]
    out.append(lines[-1])
    return "\n".join(out) + "\n"


def main() -> None:
    h = pd.read_csv(EVAL / "heldout_subtask2_benchmark.csv")
    setting_label = {"assignment1": "A1", "trivial": "--", "naive_k50": "A2, $K=50$"}
    lines = []
    for _, r in h.iterrows():
        name = NAMES.get(r.system, r.system)
        if r.setting == "train_tuned":
            kind, name = "A2", f"{name}, $K^*{{=}}{int(r.k)}$"
        elif r.setting == "naive_k50":
            kind, name = "A2", f"{name}, $K{{=}}50$"
        else:
            kind = setting_label[r.setting]
        lines.append(f"{kind} & {name} & {r.official_f1:.3f} & {ci(r.off_ci_lo, r.off_ci_hi)} & {r.pooled_macro_f1:.3f} \\\\")
    (OUT / "heldout.tex").write_text(grid(
        "\\begin{tabular}{llccc}\n\\toprule\nSet & Model & Official F1 & 95\\% CI & Pooled F1 \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"))

    b = pd.read_csv(EVAL / "bootstrap_ranking_ci.csv")
    lines = []
    for sysname, g in b.groupby("system", sort=False):
        cells = [f"{NAMES.get(sysname, sysname)}"]
        for m in ("mrr", "map", "ndcg@10"):
            r = g[g.metric == m].iloc[0]
            cells.append(f"{r['mean']:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]")
        lines.append(" & ".join(cells) + " \\\\")
    (OUT / "bootstrap.tex").write_text(grid(
        "\\begin{tabular}{llll}\n\\toprule\n\\textbf{System} & \\textbf{MRR} & \\textbf{MAP} & \\textbf{nDCG@10} \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"))

    p = pd.read_csv(EVAL / "paired_system_differences.csv")
    lines = []
    p = p[p.metric.isin(["map", "ndcg@10"]) & ~p.a.str.contains("xlmr") & ~p.b.str.contains("xlmr")]
    for _, r in p.iterrows():
        sig = "yes" if (r.ci_lo > 0 or r.ci_hi < 0) else "no"
        lines.append(f"{NAMES.get(r.a, r.a)} $-$ {NAMES.get(r.b, r.b)} & {r.metric} & {r.mean_diff:+.3f} & {ci(r.ci_lo, r.ci_hi)} & {int(r.a_wins)}/{int(r.n)} & {sig} \\\\")
    (OUT / "paired.tex").write_text(grid(
        "\\begin{tabular}{llcccc}\n\\toprule\n\\textbf{Comparison} & \\textbf{Metric} & \\textbf{Mean diff.} & \\textbf{95\\% CI} & \\textbf{Wins} & \\textbf{CI excludes 0} \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"))
    pw = pd.read_csv(EVAL / "heldout_per_word_f1.csv")
    piv = pw.pivot(index="word", columns="system", values="f1")
    meta = pw[pw.system == "dense"].set_index("word")[["n_test", "n_pos"]]
    lines = []
    for w in piv.index:
        lines.append(f"{w} & {int(meta.loc[w, 'n_test'])} & {int(meta.loc[w, 'n_pos'])} & {piv.loc[w, 'all_positive']:.2f} & {piv.loc[w, 'bm25']:.2f} & {piv.loc[w, 'dense']:.2f} & {piv.loc[w, 'hybrid']:.2f} \\\\")
    (OUT / "perword.tex").write_text(grid(
        "\\begin{tabular}{lrrcccc}\n\\toprule\nWord & Test & Pos. & Always 1 & BM25 & Dense & Hybrid \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"))

    sc = pd.read_csv(EVAL / "ablation_scope.csv").set_index("scope")
    fk = pd.read_csv(EVAL / "ablation_fusion_k.csv")
    fk.columns = [c.strip() for c in fk.columns]
    kcol = [c for c in fk.columns if c.lower() in ("k", "rrf_k", "fusion_k")]
    lines = [f"BM25, per-word pool & {sc.loc['per_word','mrr']:.3f} & {sc.loc['per_word','map']:.3f} & {sc.loc['per_word','ndcg@10']:.3f} \\\\",
             f"BM25, global pool & {sc.loc['global','mrr']:.3f} & {sc.loc['global','map']:.3f} & {sc.loc['global','ndcg@10']:.3f} \\\\",
             "\\midrule"]
    for _, r in fk.iterrows():
        kv = int(r[kcol[0]]) if kcol else int([10, 30, 60, 100][_])
        lines.append(f"Hybrid, RRF $k={kv}$ & {r['mrr']:.3f} & {r['map']:.3f} & {r['ndcg@10']:.3f} \\\\")
    (OUT / "ablation.tex").write_text(grid(
        "\\begin{tabular}{lccc}\n\\toprule\nVariant & MRR & MAP & nDCG@10 \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"))
    print("wrote", sorted(x.name for x in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()
