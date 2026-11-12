"""Logging helpers: one console logger plus an optional per-run file log."""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from .paths import LOGS_DIR, ensure_dir

LOGGER_NAME = "assignment2"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str | None = None, log_file: bool = False) -> logging.Logger:
    """Return the shared Assignment 2 logger.

    Args:
        name: Optional suffix to distinguish module loggers.
        log_file: When True, also write to ``logs/run_<timestamp>.log``.
    """
    logger = logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(console)

    if log_file:
        ensure_dir(LOGS_DIR)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(LOGS_DIR / f"run_{stamp}.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(file_handler)

    return logger
