"""Generate academic LaTeX tables for the Assignment 3 Research Report.

Directly parses result CSVs in outputs/tables/ and formats them using booktabs
with bold headers, proper mathematical notation, and consistent 3-decimal rounding.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
A3_ROOT = REPO_ROOT / "Assignment_3"
CSV_DIR = A3_ROOT / "outputs" / "tables"
TEX_DIR = REPO_ROOT / "report" / "assignment_3" / "tables"
TEX_DIR.mkdir(parents=True, exist_ok=True)


def booktabs(tex: str) -> str:
    """Format table with booktabs style."""
    lines = tex.strip().split("\n")
    return "\n".join(lines) + "\n"


def make_table1() -> None:
    df = pd.read_csv(CSV_DIR / "table1_multi_seed_subtask2_benchmark.csv")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Phase} & \textbf{Model / System} & \textbf{Official $F_1$} & \textbf{95\% CI} & \textbf{Pooled $F_1$} & \textbf{Accuracy} \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        name = r["display_name"]
        phase = "A2" if "Assignment 2" in r["phase"] else "A1"
        ci = f"[{r['ci95_off_lo']:.3f}, {r['ci95_off_hi']:.3f}]"
        lines.append(
            f"{phase} & {name} & {r['mean_official_f1']:.3f} $\\pm$ {r['std_official_f1']:.3f} & {ci} & {r['mean_pooled_macro_f1']:.3f} & {r['mean_accuracy']:.3f} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Subtask 2 classification benchmark across 5 random seeds (80/20 stratified split, $n_{\text{test}}=397$). Official metric is per-word macro-averaged positive-class $F_1$. Secondary metrics include pooled macro-$F_1$ and accuracy.}",
        r"\label{tab:subtask2_benchmark}",
        r"\end{table}",
    ])
    tex_path = TEX_DIR / "table1_benchmark.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    (CSV_DIR / "table1_benchmark.tex").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote table1_benchmark.tex")


def make_table2() -> None:
    df = pd.read_csv(CSV_DIR / "table2_retrieval_ranking_metrics.csv")
    piv = df.pivot(index="system", columns="metric", values="mean")
    piv_lo = df.pivot(index="system", columns="metric", values="ci95_lo")
    piv_hi = df.pivot(index="system", columns="metric", values="ci95_hi")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Retriever} & \textbf{MRR} & \textbf{MAP} & \textbf{nDCG@10} & \textbf{Recall@50} \\",
        r"\midrule",
    ]
    name_map = {"bm25": "BM25 (Sparse)", "dense": "Dense (E5-small)", "hybrid": "Hybrid (RRF, $k=60$)"}
    for s in ["bm25", "dense", "hybrid"]:
        if s in piv.index:
            mrr_str = f"{piv.loc[s, 'mrr']:.3f} [{piv_lo.loc[s, 'mrr']:.2f}, {piv_hi.loc[s, 'mrr']:.2f}]"
            map_str = f"{piv.loc[s, 'map']:.3f} [{piv_lo.loc[s, 'map']:.2f}, {piv_hi.loc[s, 'map']:.2f}]"
            ndcg_str = f"{piv.loc[s, 'ndcg@10']:.3f} [{piv_lo.loc[s, 'ndcg@10']:.2f}, {piv_hi.loc[s, 'ndcg@10']:.2f}]"
            rec_str = f"{piv.loc[s, 'recall@50']:.3f}"
            lines.append(f"{name_map.get(s, s)} & {mrr_str} & {map_str} & {ndcg_str} & {rec_str} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Information Retrieval ranking performance on the 10 target queries. Mean scores reported alongside 95\% bootstrap percentile confidence intervals in brackets.}",
        r"\label{tab:retrieval_metrics}",
        r"\end{table}",
    ])
    (TEX_DIR / "table2_ranking.tex").write_text("\n".join(lines), encoding="utf-8")
    (CSV_DIR / "table2_ranking.tex").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote table2_ranking.tex")


def make_table3() -> None:
    df = pd.read_csv(CSV_DIR / "table3_paired_statistical_significance.csv")
    df_off = df[df["metric"] == "official_f1"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Paired Comparison ($A$ vs $B$)} & \textbf{Mean Diff ($\Delta$)} & \textbf{95\% CI} & \textbf{$t$-test $p$} & \textbf{Permutation $p$} & \textbf{Sig.} \\",
        r"\midrule",
    ]
    for _, r in df_off.iterrows():
        sig = r"\textbf{Yes}" if r["significant_p05"] == "Yes" else "No"
        ci = f"[{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}]"
        lines.append(
            f"{r['comparison']} & {r['mean_diff']:+.3f} & {ci} & {r['t_pvalue']:.4f} & {r['permutation_pvalue']:.4f} & {sig} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Hypothesis testing of paired system differences in Official Subtask 2 $F_1$ across 5 random seeds. Significance assessed at $\alpha=0.05$ via paired Student's $t$-test and 10,000-resample permutation tests.}",
        r"\label{tab:paired_significance}",
        r"\end{table}",
    ])
    (TEX_DIR / "table3_significance.tex").write_text("\n".join(lines), encoding="utf-8")
    (CSV_DIR / "table3_significance.tex").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote table3_significance.tex")


def make_table4() -> None:
    df = pd.read_csv(CSV_DIR / "table4_ablation_summary.csv")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Ablation Axis} & \textbf{Variant} & \textbf{Official $F_1$} & \textbf{MRR} & \textbf{MAP} & \textbf{nDCG@10} \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        f1_str = f"{r['official_f1']:.3f}" if pd.notna(r["official_f1"]) else "--"
        mrr_str = f"{r['mrr']:.3f}" if pd.notna(r["mrr"]) else "--"
        map_str = f"{r['map']:.3f}" if pd.notna(r["map"]) else "--"
        ndcg_str = f"{r['ndcg@10']:.3f}" if pd.notna(r["ndcg@10"]) else "--"
        lines.append(f"{r['ablation_axis']} & {r['variant']} & {f1_str} & {mrr_str} & {map_str} & {ndcg_str} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Controlled ablation studies isolating candidate scoping, RRF smoothing parameter $k$, operating cutoff tuning, and target word identity.}",
        r"\label{tab:ablation_summary}",
        r"\end{table}",
    ])
    (TEX_DIR / "table4_ablations.tex").write_text("\n".join(lines), encoding="utf-8")
    (CSV_DIR / "table4_ablations.tex").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote table4_ablations.tex")


def make_table7() -> None:
    df = pd.read_csv(CSV_DIR / "table7_hardware_and_compute_cost.csv")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{System} & \textbf{Params (M)} & \textbf{Index Time (s)} & \textbf{Latency (ms)} & \textbf{RAM (MB)} & \textbf{Size (KB)} \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['system']} & {r['parameters_m']:.1f} & {r['indexing_time_s']:.2f} & {r['query_latency_ms']:.2f} & {r['memory_mb']:.0f} & {r['checkpoint_kb']:.1f} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Computational efficiency, execution latency, and resource footprints profiled on an Intel 8-core CPU platform (16 GB RAM). Latency denotes inference time per test query/usage.}",
        r"\label{tab:hardware_compute}",
        r"\end{table}",
    ])
    (TEX_DIR / "table7_hardware.tex").write_text("\n".join(lines), encoding="utf-8")
    (CSV_DIR / "table7_hardware.tex").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote table7_hardware.tex")


def main() -> None:
    make_table1()
    make_table2()
    make_table3()
    make_table4()
    make_table7()
    print("All LaTeX tables successfully created!")


if __name__ == "__main__":
    main()
