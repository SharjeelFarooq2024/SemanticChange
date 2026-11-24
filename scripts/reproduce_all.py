"""Master reproduction script for SemEval-2027 Task 3, Subtask 2.
Runs the complete audited and validated pipeline from Stage 1 to Stage 6.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = sys.executable

STAGES = [
    ("Stage 1 - Measure Baseline vs Hybrid & Leakage", ROOT / "scripts/stage1_measure.py"),
    ("Stage 2 - Cheap Calibration (Paired Bootstrap)", ROOT / "scripts/stage2_calibrate.py"),
    ("Stage 3 - Definition Matching & Paired Bootstrap", ROOT / "scripts/stage3_definition_matching.py"),
    ("Stage 5 - Generalization Checks (LOLO & LOPO)", ROOT / "scripts/stage5_generalization.py"),
    ("Stage 6 - Final Nested CV (Baseline Linear SVM)", ROOT / "scripts/stage6_final_nested_cv.py"),
    ("Stage 6b - Final Nested CV (Dense XLM-R Definition Head)", ROOT / "scripts/stage6_nested_cv_dense_xlmr.py"),
    ("Prediction - Calibrated Dense XLM-R Inference", ROOT / "scripts/predict.py"),
]

def run_stage(title: str, script_path: Path):
    print("=" * 80)
    print(f">>> Running {title}")
    print(f"    Script: {script_path}")
    print("=" * 80)
    t0 = time.time()
    res = subprocess.run([PYTHON_EXE, str(script_path)], cwd=str(ROOT))
    if res.returncode != 0:
        print(f"ERROR: {title} failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f">>> Finished {title} in {time.time()-t0:.1f}s\n")

def main():
    print("Starting full pipeline reproduction...")
    t_start = time.time()
    for title, script_path in STAGES:
        run_stage(title, script_path)
    print("=" * 80)
    print(f"All stages successfully reproduced in {time.time()-t_start:.1f}s.")
    print("Results are available in the 'results/' directory.")
    print("=" * 80)

if __name__ == "__main__":
    main()
