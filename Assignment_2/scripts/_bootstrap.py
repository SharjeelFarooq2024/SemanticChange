"""Shared bootstrap for Assignment 2 standalone scripts.

Ensures the repository root is importable so ``Assignment_2.*`` and the
Assignment 1 ``src`` package both resolve correctly, regardless of the
current working directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

# scripts/_bootstrap.py -> parents[0]=scripts, [1]=Assignment_2, [2]=repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
