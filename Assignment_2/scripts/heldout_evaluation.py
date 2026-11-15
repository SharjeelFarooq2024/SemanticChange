"""Held-out, leakage-free re-scoring of the retrieval systems on the Assignment 1 split.

Fixes two issues in the original unified benchmark:
  1. Retrieval systems were scored on all 1,983 usages while the Assignment 1
     baselines were scored on a 20% stratified test split (n=397, seed 42).
  2. The rank cutoff K* was tuned on the same rows it was then scored on.

Here the Assignment 1 split is reproduced exactly (same loader, same
``train_test_split`` call). K* is tuned on the TRAIN rows only and applied to the
TEST rows. Percentile-bootstrap 95% CIs are reported for the test Macro-F1 and
for query-level ranking metrics.

Usage (from SemanticChange/):
    python Assignment_2/scripts/heldout_evaluation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.common import join_subtask2_labels, load_baseline_bundle, load_config  # noqa: E402
from Assignment_2.src.evaluation.metrics import average_precision, ndcg_at_k, reciprocal_rank  # noqa: E402

A2 = REPO_ROOT / "Assignment_2"
OUT = A2 / "outputs" / "evaluation"
SYSTEMS = ["bm25", "dense", "hybrid", "dense_xlmr_base"]
K_GRID = list(range(5, 105, 5))  # saved rankings keep only the top 100 per word (top_k=100), so K* <= 100
N_BOOT = 1000  # same as Assignment 1
SEED = 42


def a1_split(config):
    bundle = load_baseline_bundle(REPO_ROOT, config)
    labeled, _ = join_subtask2_labels(bundle["usage"], bundle["subtask2"])
    ev = config["evaluation"]
    train, test = train_test_split(
        labeled, test_size=ev["test_size"], random_state=ev["random_state"], stratify=labeled["label_value"]
    )
    return train, test


def pred_at_k(rows: pd.DataFrame, ranked: pd.DataFrame, k: int) -> np.ndarray:
    """Predict 1 iff the usage is within the top-k of its word's ranking."""
    r = ranked.set_index(["query_id", "doc_id"])["rank"]
    ranks = pd.Series(list(zip(rows["word"], rows["sentence_id"]))).map(r).to_numpy(dtype=float)
    return (np.nan_to_num(ranks, nan=1e9) <= k).astype(int)


def macro_f1(y, p):
    """Pooled macro-F1 over both classes (secondary metric)."""
    return f1_score(y, p, average="macro", zero_division=0)


def official_f1(widx, y, p, n_words):
    """Official Subtask 2 metric: positive-class F1 per word, averaged over words.

    F1_w = 2TP / (2TP + FP + FN); a word with no positives and no predicted
    positives contributes 0 (so the always-0 majority baseline scores 0).
    """
    y = np.asarray(y); p = np.asarray(p)
    tp = np.bincount(widx, weights=(y == 1) & (p == 1), minlength=n_words)
    fp = np.bincount(widx, weights=(y == 0) & (p == 1), minlength=n_words)
    fn = np.bincount(widx, weights=(y == 1) & (p == 0), minlength=n_words)
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros_like(den), where=den > 0)
    return float(f1.mean())


def boot_ci(stat, n, rng):
    vals = np.array([stat(rng.integers(0, n, n)) for _ in range(N_BOOT)])
    return np.percentile(vals, [2.5, 97.5])


def main() -> None:
    config = load_config(REPO_ROOT)
    train, test = a1_split(config)
    print(f"train={len(train)} test={len(test)} test_pos={int(test.label_value.sum())}")

    # Sanity: identical split to the Assignment 1 baselines.
    a1_pred = pd.read_csv(REPO_ROOT / "outputs/baselines/classical/subtask2_predictions.csv")
    a1_ids = set(a1_pred[a1_pred.model == "majority"]["sentence_id"])
    assert a1_ids == set(test["sentence_id"]), "test split does not match Assignment 1 split"
    print("OK: test split identical to Assignment 1 baselines")

    rng = np.random.default_rng(SEED)
    words = sorted(train["word"].unique())
    w_of = {w: i for i, w in enumerate(words)}
    nw = len(words)
    wtr = train["word"].map(w_of).to_numpy()
    wte = test["word"].map(w_of).to_numpy()
    ytr = train["label_value"].to_numpy()
    ytest = test["label_value"].to_numpy()

    def add_row(rows, system, setting, k, p):
        """Score predictions p on the test rows with bootstrap CIs (official metric first)."""
        n = len(ytest)
        boots = []
        for _ in range(N_BOOT):
            i = rng.integers(0, n, n)
            boots.append((official_f1(wte[i], ytest[i], p[i], nw), macro_f1(ytest[i], p[i])))
        boots = np.array(boots)
        o_lo, o_hi = np.percentile(boots[:, 0], [2.5, 97.5])
        m_lo, m_hi = np.percentile(boots[:, 1], [2.5, 97.5])
        rows.append(dict(system=system, setting=setting, k=k,
                         official_f1=official_f1(wte, ytest, p, nw), off_ci_lo=o_lo, off_ci_hi=o_hi,
                         pooled_macro_f1=macro_f1(ytest, p), mac_ci_lo=m_lo, mac_ci_hi=m_hi,
                         accuracy=float((ytest == p).mean())))

    rows = []
    for sysname in SYSTEMS:
        path = A2 / "predictions" / sysname / "ranked.csv"
        if not path.exists():
            print("skip (missing):", sysname)
            continue
        ranked = pd.read_csv(path)
        # K* is chosen on TRAIN rows only, maximizing the OFFICIAL metric.
        sweep = [(k, official_f1(wtr, ytr, pred_at_k(train, ranked, k), nw)) for k in K_GRID]
        k_star = max(sweep, key=lambda t: t[1])[0]
        add_row(rows, sysname, "naive_k50", 50, pred_at_k(test, ranked, 50))
        add_row(rows, sysname, "train_tuned", k_star, pred_at_k(test, ranked, k_star))
    # Trivial reference: predict 1 for every usage (positive-class F1 rewards this on skewed words).
    add_row(rows, "all_positive", "trivial", np.nan, np.ones(len(ytest), dtype=int))
    # Assignment 1 baselines on the very same rows, with the same bootstrap.
    for f in ("classical", "pretrained"):
        pp = REPO_ROOT / f"outputs/baselines/{f}/subtask2_predictions.csv"
        if not pp.exists():
            continue
        a = pd.read_csv(pp)
        for model, g in a.groupby("model"):
            g = g.set_index("sentence_id").loc[test["sentence_id"]]
            add_row(rows, model, "assignment1", np.nan, g["prediction"].to_numpy())
    res = pd.DataFrame(rows).drop_duplicates(["system", "setting"]).sort_values("official_f1", ascending=False)
    res.to_csv(OUT / "heldout_subtask2_benchmark.csv", index=False)
    print(res.round(3).to_string(index=False))

    # Per-word official F1 on the test rows (descriptive).
    pw = []
    base_pred = {"all_positive": np.ones(len(ytest), dtype=int)}
    for sysname in ("bm25", "dense", "hybrid"):
        path = A2 / "predictions" / sysname / "ranked.csv"
        if path.exists():
            kk = int(res[(res.system == sysname) & (res.setting == "train_tuned")]["k"].iloc[0])
            base_pred[sysname] = pred_at_k(test, pd.read_csv(path), kk)
    for name, p in base_pred.items():
        for w in words:
            m = (test["word"] == w).to_numpy()
            pw.append(dict(system=name, word=w, n_test=int(m.sum()), n_pos=int(ytest[m].sum()),
                           f1=official_f1(np.zeros(m.sum(), dtype=int), ytest[m], p[m], 1)))
    pd.DataFrame(pw).to_csv(OUT / "heldout_per_word_f1.csv", index=False)

    # Query-level (n=10 words) bootstrap of ranking metrics + paired hybrid-vs-dense difference.
    qrels = pd.read_csv(A2 / "predictions" / "qrels.csv")
    per_q = {}
    for sysname in SYSTEMS:
        path = A2 / "predictions" / sysname / "ranked.csv"
        if not path.exists():
            continue
        ranked = pd.read_csv(path)
        recs = {}
        for q, g in ranked.groupby("query_id"):
            order = g.sort_values("rank")["doc_id"].tolist()
            rel = set(qrels[(qrels.query_id == q) & (qrels.relevance == 1)]["doc_id"])
            recs[q] = (reciprocal_rank(order, rel), average_precision(order, rel), ndcg_at_k(order, rel, 10))
        per_q[sysname] = pd.DataFrame(recs, index=["mrr", "map", "ndcg@10"]).T.sort_index()
    qrows = []
    for sysname, df in per_q.items():
        for m in df.columns:
            v = df[m].to_numpy()
            lo, hi = boot_ci(lambda i: v[i].mean(), len(v), rng)
            qrows.append(dict(system=sysname, metric=m, mean=v.mean(), ci_lo=lo, ci_hi=hi, n_queries=len(v)))
    pd.DataFrame(qrows).to_csv(OUT / "bootstrap_ranking_ci.csv", index=False)
    print(pd.DataFrame(qrows).round(3).to_string(index=False))

    drows = []
    for a, b in (("hybrid", "dense"), ("dense", "bm25"), ("hybrid", "bm25"), ("dense", "dense_xlmr_base")):
        if a in per_q and b in per_q:
            for m in ("mrr", "map", "ndcg@10"):
                d = per_q[a][m].to_numpy() - per_q[b][m].to_numpy()
                lo, hi = boot_ci(lambda i: d[i].mean(), len(d), rng)
                drows.append(dict(a=a, b=b, metric=m, mean_diff=d.mean(), ci_lo=lo, ci_hi=hi,
                                  a_wins=int((d > 0).sum()), n=len(d)))
    pd.DataFrame(drows).to_csv(OUT / "paired_system_differences.csv", index=False)
    print(pd.DataFrame(drows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
