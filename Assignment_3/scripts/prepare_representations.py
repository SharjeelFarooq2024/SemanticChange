"""Generate exact-revision frozen representations, with measured resource use."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from Assignment_3.src.runtime import Measurements, load_config, logger, prepare


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config()
    output = ROOT / config["output_dir"]
    log = logger(output)
    prepare(config, output, log, Measurements(output, config["sample_memory_interval_s"]), args.allow_download, args.force)
