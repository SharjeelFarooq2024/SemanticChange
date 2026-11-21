"""Export scope: historical source stays locally but is not a final deliverable."""
from Assignment_3.src.runtime import ROOT

RETIRED_FILES = {
    "Assignment_3/scripts/run_hybrid_improvement.py", "Assignment_3/scripts/infer_improved_hybrid.py",
    "Assignment_3/scripts/run_neural_pilot.py", "Assignment_3/scripts/package_advanced_trials.py",
    "Assignment_3/scripts/write_part2_report.py", "Assignment_3/scripts/make_latex_tables.py",
    "Assignment_3/scripts/plot_figures.py", "Assignment_3/configs/hybrid_improvement.json",
    "Assignment_3/configs/neural_pilot.json", "Assignment_3/configs/part2_analysis.json",
    "Assignment_3/configs/error_example_notes.json", "Assignment_3/tests/test_neural_pilot.py",
}


def current_deliverable(path):
    return path.is_file() and str(path.relative_to(ROOT)).replace("\\", "/") not in RETIRED_FILES
