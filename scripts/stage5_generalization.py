"""Stage 5: Generalization checks (LOLO and LOPO) for Baseline SVM and Dense XLM-R systems."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Assignment_3/.deps"))

import time
import pandas as pd

from eval_harness import leave_one_lemma_out, leave_one_period_out
from scripts.stage1_measure import load_data, BaselineSVMWrapper
from scripts.stage3_definition_matching import (
    DenseFeatureStore,
    DenseDefinitionModel,
    EnsembleDenseSVM,
)


def _summarize_and_save(
    data: pd.DataFrame,
    model_label: str,
    model_factory,
    lolo_path: Path,
    lopo_path: Path,
) -> None:
    print(f"\n=== [{model_label}] Leave-One-Lemma-Out (LOLO) ===")
    t0 = time.time()
    lolo_res = leave_one_lemma_out(data, model_factory, seed=42)
    print(f"Mean LOLO F1: {lolo_res['mean_lolo_f1']:.6f} (took {time.time()-t0:.1f}s)")

    lolo_rows = []
    for word, score in lolo_res["per_lemma_f1"].items():
        sub = data[data["word"] == word]
        pos_cnt = int(sub["label"].sum())
        neg_cnt = len(sub) - pos_cnt
        min_cnt = min(pos_cnt, neg_cnt)
        lolo_rows.append({
            "held_out_lemma": word,
            "total_support": len(sub),
            "minority_support": min_cnt,
            "unstable_flag": bool(min_cnt < 10),
            "lolo_macro_f1": float(score),
        })
    df_lolo = pd.DataFrame(lolo_rows)
    df_lolo.to_csv(lolo_path, index=False)
    print(f"Saved LOLO results to {lolo_path}")

    print(f"\n=== [{model_label}] Leave-One-Period-Out (LOPO) ===")
    t0 = time.time()
    lopo_res = leave_one_period_out(data, model_factory, seed=42)
    print(f"Mean LOPO F1: {lopo_res['mean_lopo_f1']:.6f} (took {time.time()-t0:.1f}s)")

    lopo_rows = []
    for period, score in lopo_res["per_period_f1"].items():
        sub = data[data["period_label"] == period]
        lopo_rows.append({
            "held_out_period": period,
            "total_support": len(sub),
            "n_words_present": int(sub["word"].nunique()),
            "lopo_official_f1": float(score),
        })
    df_lopo = pd.DataFrame(lopo_rows)
    df_lopo.to_csv(lopo_path, index=False)
    print(f"Saved LOPO results to {lopo_path}")


def main():
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    data = load_data().reset_index(drop=True)

    _summarize_and_save(
        data,
        "Baseline Linear SVM",
        lambda s: BaselineSVMWrapper(s),
        results_dir / "lolo_results.csv",
        results_dir / "lopo_results.csv",
    )

    store_docs = DenseFeatureStore(encoder="xlmr", mode="docs")
    _summarize_and_save(
        data,
        "Dense XLM-R (Docs, C=0.1)",
        lambda s: DenseDefinitionModel(store_docs, seed=s, C=0.1),
        results_dir / "lolo_results_dense_xlmr.csv",
        results_dir / "lopo_results_dense_xlmr.csv",
    )

    store_both = DenseFeatureStore(encoder="xlmr", mode="both")
    _summarize_and_save(
        data,
        "Exploratory Blend (Dual-Ctx Dense + LemmaSVM)",
        lambda s: EnsembleDenseSVM(store_both, alpha=0.8, C=0.05, per_lemma_weights=True, seed=s),
        results_dir / "lolo_results_ensemble.csv",
        results_dir / "lopo_results_ensemble.csv",
    )

    print("\nStage 5 generalization audit completed cleanly.")


if __name__ == "__main__":
    main()
