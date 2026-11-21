"""Package additive advanced trials with the unchanged earlier experiment bundle."""
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from Assignment_3.src.runtime import dump_json, sha256


def main():
    base = ROOT / "Assignment_3/improved_hybrid_package.zip"
    metadata = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    if sha256(base) != metadata["sha256"]:
        raise ValueError("Earlier experiment package integrity failed")
    files = [Path(__file__).resolve(), ROOT / "Assignment_3/RUNNING_ADVANCED.md",
             ROOT / "Assignment_3/src/advanced_hybrid.py"]
    for name in ["advanced_hybrid", "neural_pilot"]:
        files += [ROOT / f"Assignment_3/configs/{name}.json",
                  ROOT / f"Assignment_3/scripts/run_{name}.py",
                  ROOT / f"Assignment_3/tests/test_{name}.py"]
        output = ROOT / f"Assignment_3/runs/{name}"
        (output / "execution_snapshot.log").write_bytes((output / "execution.log").read_bytes())
        files += [p for p in output.rglob("*") if p.is_file() and p.name != "execution.log"]
    files.append(ROOT / "Assignment_3/scripts/infer_advanced_hybrid.py")
    mapping = {str(p.relative_to(ROOT)).replace("\\", "/"): p for p in sorted(set(files))}
    manifest = ROOT / "Assignment_3/runs/advanced_trials_manifest.json"
    dump_json(manifest, {"base_package_sha256": metadata["sha256"],
        "files_sha256": {name: sha256(path) for name, path in mapping.items()},
        "official_metric_unchanged": True, "neural_pilot_scope": "inner_validation_only",
        "pretrained_weights_included": False,
        "pretrained_weights": "Pinned revisions and local/download instructions are in the base package.",
        "reports_generated": False})
    mapping[str(manifest.relative_to(ROOT)).replace("\\", "/")] = manifest
    archive = ROOT / "Assignment_3/advanced_trials_package.zip"
    with zipfile.ZipFile(base) as previous, zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as target:
        for item in previous.infolist():
            if item.filename not in mapping:
                target.writestr(item, previous.read(item.filename))
        for name, path in mapping.items():
            target.write(path, name)
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None:
            raise AssertionError("ZIP integrity failed")
        for name, path in mapping.items():
            if package.read(name) != path.read_bytes():
                raise AssertionError(f"Packaged artifact mismatch: {name}")
    results = json.loads((ROOT / "Assignment_3/runs/advanced_hybrid/result_summary.json").read_text(encoding="utf-8"))
    dump_json(archive.with_suffix(".json"), {"bytes": archive.stat().st_size, "sha256": sha256(archive),
        "base_package_preserved": True, "advanced_checkpoints": 5, "neural_pilot_checkpoints": 1,
        "official_f1_target_reached": results["target_reached"]})
    print(f"Verified {archive.name}: {archive.stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
