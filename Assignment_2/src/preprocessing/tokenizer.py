"""Unicode-aware tokenizer shared by all lexical (BM25) systems.

Keeps Swedish letters (å ä ö) and digits, because temporal expressions
(years, periods) must not be discarded. No stopword list is applied.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .text_preprocessor import TextPreprocessor

DEFAULT_PATTERN = r"[\w]+"
DEFAULT_LOWERCASE = True


class Tokenizer:
    """Deterministic regex tokenizer with configurable lowercase handling."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        pattern = config.get("pattern", DEFAULT_PATTERN)
        self.lowercase = bool(config.get("lowercase", DEFAULT_LOWERCASE))
        self.pattern = pattern
        self._regex = re.compile(pattern, flags=re.UNICODE)
        # Preprocessing applied before tokenization (whitespace/unicode only;
        # lowercase is owned by this tokenizer so BM25 config controls it).
        self._preprocessor = TextPreprocessor(
            {"lowercase": False, "strip": True, "collapse_whitespace": True}
        )

    def tokenize(self, text: Optional[str]) -> List[str]:
        """Tokenize ``text`` into a list of lexical tokens."""
        normalized = self._preprocessor.process(text)
        if not normalized:
            return []
        if self.lowercase:
            normalized = normalized.lower()
        return self._regex.findall(normalized)

    def describe(self) -> Dict[str, Any]:
        return {"pattern": self.pattern, "lowercase": self.lowercase}
