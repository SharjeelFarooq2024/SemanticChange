"""Stage 5: Generalization checks (LOLO and LOPO) on the adopted Calibrated SVM system."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Assignment_3/.deps"))

import json
import time
import pandas as pd
import numpy as np

from eval_harness import leave_one_lemma_out, leave_one_period_out, official_f1
from scripts.stage1_measure import load_data, BaselineSVMWrapper
from Assignment_3.src.runtime import ROOT

def main():
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    data = load_data()

    print("--- [1/2] Leave-One-Lemma-Out (LOLO) Cross-Validation ---")
    t0 = time.time()
    lolo_res = leave_one_lemma_out(data, lambda s: BaselineSVMWrapper(s), seed=42)
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
    lolo_path = results_dir / "lolo_results.csv"
    df_lolo.to_csv(lolo_path, index=False)
    print(f"Saved LOLO results to {lolo_path}")
    print(df_lolo)

    print("\n--- [2/2] Leave-One-Period-Out (LOPO) Cross-Validation ---")
    t0 = time.time()
    lopo_res = leave_one_period_out(data, lambda s: BaselineSVMWrapper(s), seed=42)
    print(f"Mean LOPO F1: {lopo_res['mean_lopo_f1']:.6f} (took {time.time()-t0:.1f}s)")

    lopo_rows = []
    for period, score in lopo_res["per_period_f1"].items():
        sub = data[data["period_label"] == period]
        lopo_rows.append({
            "held_out_period": period,
            "total_support": len(sub),
            "lopo_official_f1": float(score),
        })
    df_lopo = pd.DataFrame(lopo_rows)
    lopo_path = results_dir / "lopo_results.csv"
    df_lopo.to_csv(lopo_path, index=False)
    print(f"Saved LOPO results to {lopo_path}")
    print(df_lopo)

    print("\nStage 5 completed successfully.")

if __name__ == "__main__":
    main()
