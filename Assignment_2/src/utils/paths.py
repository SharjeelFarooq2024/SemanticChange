"""Path resolution helpers for Assignment 2.

All relative paths in configuration files are resolved against either the
repository root (shared Assignment 1 data) or the Assignment_2 root
(artifacts, predictions, logs).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

# Assignment_2/src/utils/paths.py -> parents[0]=utils, [1]=src, [2]=Assignment_2, [3]=repo root
ASSIGNMENT2_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ASSIGNMENT2_ROOT.parent
CONFIG_DIR = ASSIGNMENT2_ROOT / "configs"
ARTIFACTS_DIR = ASSIGNMENT2_ROOT / "artifacts"
PREDICTIONS_DIR = ASSIGNMENT2_ROOT / "predictions"
LOGS_DIR = ASSIGNMENT2_ROOT / "logs"


def ensure_dir(path: Path | str) -> Path:
    """Create a directory (and parents) if missing; return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_repo_path(relative: str | Path) -> Path:
    """Resolve a path expressed relative to the repository root."""
    path = Path(relative)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_a2_path(relative: str | Path) -> Path:
    """Resolve a path expressed relative to the Assignment_2 root."""
    path = Path(relative)
    return path if path.is_absolute() else ASSIGNMENT2_ROOT / path


def load_config(name: str) -> Dict[str, Any]:
    """Load a YAML configuration file from Assignment_2/configs/.

    ``name`` may be given with or without the ``.yaml`` suffix.
    """
    import yaml

    filename = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")
    return config
