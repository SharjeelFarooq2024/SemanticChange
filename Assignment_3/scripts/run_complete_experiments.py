"""Execute the complete code/experiment package; never generates a paper or report."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from Assignment_3.src.runtime import Measurements, dump_json, load_config, logger, prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    config = load_config()
    output = ROOT / config["output_dir"]
    log = logger(output)
    start = time.perf_counter()
    if not args.analysis_only:
        measurements = Measurements(output, config["sample_memory_interval_s"])
        prepare(config, output, log, measurements, args.allow_download)
        from Assignment_3.src.experiments import run_experiments
        run_experiments(config, output, log, measurements)
    from Assignment_3.src.analysis import analyze
    analyze(config, output, log)
    dump_json(output / "execution_summary.json", {"command": "run_complete_experiments.py", "wall_seconds_this_invocation": time.perf_counter()-start,
              "wall_time_scope": "this invocation before packaging; encoder preparation may be a separately logged invocation",
              "analysis_only": args.analysis_only, "reports_generated": False, "presentation_generated": False})
    from Assignment_3.src.package import finalize
    finalize(config, output, log)
    log.info("Complete. Artifacts: %s", output)


if __name__ == "__main__":
    main()
