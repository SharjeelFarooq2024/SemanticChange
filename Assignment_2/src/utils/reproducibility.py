"""Reproducibility helpers (seed control), mirroring Assignment 1 conventions."""
from __future__ import annotations

import random

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed ``random`` and ``numpy``; seed ``torch`` when it is available."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency elsewhere
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - CPU-only environment
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
