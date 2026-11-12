"""Pretrained dense encoders for query/document embedding.

Two encoder profiles are configured in ``dense_config.yaml``:

- ``e5_small``  — primary: ``intfloat/multilingual-e5-small``, trained
  contrastively for query<->passage retrieval, multilingual (incl. Swedish).
- ``xlmr_base`` — ablation: frozen ``FacebookAI/xlm-roberta-base`` mean-pool,
  mirroring the Assignment 1 feature extractor (not retrieval-trained).

The model is loaded lazily, frozen (``requires_grad_(False)``), kept in
eval mode, and run under ``torch.no_grad()``. Pooling is mean pooling for
both profiles; E5-style prefixes and L2 normalization are applied per
configuration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..utils.logging_utils import get_logger

SUPPORTED_ENCODERS = ("e5_small", "xlmr_base")


class EncoderError(RuntimeError):
    """Raised when a pretrained encoder cannot be loaded or misused."""


def _mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    pooled = torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(
        mask.sum(dim=1), min=1e-9
    )
    return pooled


class DenseEncoder:
    """Frozen pretrained encoder with batching, prefixes, and normalization."""

    def __init__(
        self,
        encoder_name: str,
        models_config: Dict[str, Any],
        device: str = "auto",
        model_cache_dir: Optional[Path] = None,
        log=None,
    ) -> None:
        if encoder_name not in models_config:
            raise EncoderError(
                f"Unknown encoder '{encoder_name}'. Available: {sorted(models_config)}"
            )
        self.encoder_name = encoder_name
        self.settings: Dict[str, Any] = dict(models_config[encoder_name])
        self.device_pref = device or "auto"
        self.model_cache_dir = Path(model_cache_dir) if model_cache_dir else None
        self.log = log or get_logger("encoder")

        self._model = None
        self._tokenizer = None
        self._device = None

    # ------------------------------------------------------------------ #
    # Identity (used for cache validation)
    # ------------------------------------------------------------------ #
    def identity(self, doc_fingerprint: Optional[str] = None) -> Dict[str, Any]:
        """Return the encoder identity dict stored alongside embeddings."""
        identity: Dict[str, Any] = {
            "encoder_key": self.encoder_name,
            "model_name": str(self.settings["model_name"]),
            "max_length": int(self.settings["max_length"]),
            "pooling": str(self.settings["pooling"]),
            "normalize": bool(self.settings["normalize"]),
            "query_prefix": str(self.settings.get("query_prefix", "")),
            "passage_prefix": str(self.settings.get("passage_prefix", "")),
        }
        if doc_fingerprint is not None:
            identity["doc_fingerprint"] = doc_fingerprint
        return identity

    # ------------------------------------------------------------------ #
    # Lazy model loading
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_name = str(self.settings["model_name"])
        if self.device_pref == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device_pref)

        cache_dir = str(self.model_cache_dir) if self.model_cache_dir else None
        # If the configured cache does not contain this model, fall back to the
        # default HuggingFace cache (avoids re-downloading weights that are
        # already present there, e.g. the Assignment 1 XLM-R ablation).
        if cache_dir is not None:
            model_folder = f"models--{model_name.replace('/', '--')}"
            if not (Path(cache_dir) / model_folder).exists():
                cache_dir = None
        self.log.info("Loading encoder '%s' on %s ...", model_name, device)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        except Exception as exc:  # network/cache/model problems
            raise EncoderError(
                f"Failed to load encoder '{model_name}'. Check model files and "
                f"network/cache access: {exc}"
            ) from exc

        model.to(device)
        model.eval()
        model.requires_grad_(False)  # frozen: inference only

        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self.log.info(
            "Encoder ready: %s | params=%.1fM | device=%s",
            model_name,
            sum(p.numel() for p in model.parameters()) / 1e6,
            device,
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def dimension(self) -> int:
        """Embedding dimension (requires the model to be loaded)."""
        self._ensure_loaded()
        return int(self._model.config.hidden_size)

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #
    def encode(
        self,
        texts: Sequence[str],
        kind: str = "passage",
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Embed ``texts`` as ``kind`` = ``'query'`` or ``'passage'``.

        Returns a float32 array of shape ``(len(texts), dimension)``.
        """
        if kind not in ("query", "passage"):
            raise EncoderError(f"kind must be 'query' or 'passage', got '{kind}'")
        self._ensure_loaded()
        import torch

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        batch_size = int(batch_size or self.settings.get("batch_size", 32))
        max_length = int(self.settings["max_length"])
        prefix = str(self.settings.get("query_prefix" if kind == "query" else "passage_prefix", ""))
        normalize = bool(self.settings["normalize"])

        chunks: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = [prefix + str(text) for text in texts[start : start + batch_size]]
            encoded = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = self._model(**encoded)
                pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                if normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(chunks) if len(chunks) > 1 else chunks[0]
