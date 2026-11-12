"""Text preprocessing for retrieval.

Design rules (temporal retrieval task):
  - Non-destructive: original text and IDs are always preserved elsewhere;
    this module returns normalized *views* of the text.
  - No stopword removal, no punctuation/digit stripping, no temporal
    expression removal — dates, years, and periods are signal here.
  - The same configuration is applied to queries and documents so both
    sides of retrieval are processed identically.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Optional

DEFAULT_CONFIG: Dict[str, Any] = {
    "normalize_unicode": True,   # NFC composition (Swedish combining marks)
    "lowercase": False,          # default off; BM25 tokenizer may enable its own
    "strip": True,
    "collapse_whitespace": True,
    "strip_control_chars": True,
}

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")


class TextPreprocessor:
    """Configurable, reproducible text normalizer."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    def __call__(self, text: Optional[str]) -> str:
        return self.process(text)

    def process(self, text: Optional[str]) -> str:
        """Return the normalized form of ``text`` (never raises on None)."""
        if text is None:
            return ""
        result = str(text)
        if self.config["normalize_unicode"]:
            result = unicodedata.normalize("NFC", result)
        if self.config["strip_control_chars"]:
            result = _CONTROL_RE.sub(" ", result)
        if self.config["collapse_whitespace"]:
            result = _WS_RE.sub(" ", result)
        if self.config["strip"]:
            result = result.strip()
        if self.config["lowercase"]:
            result = result.lower()
        return result

    def describe(self) -> Dict[str, Any]:
        """Return the active configuration (stored in caches for compatibility checks)."""
        return dict(self.config)
