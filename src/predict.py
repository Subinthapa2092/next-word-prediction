"""Inference: predict_next(text, k=5).

Two behaviours, like a phone keyboard:
  * text ends with a space   -> suggest the NEXT word     ("i want to go " -> to, back, ...)
  * text ends mid-word       -> COMPLETE the current word ("i want to g"  -> go, get, ...)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import config
from src.preprocess import Vocabulary, tokenize


class Predictor:
    def __init__(self, model, vocab: Vocabulary, seq_len: int | None = None):
        self.model = model
        self.vocab = vocab
        self.seq_len = seq_len or int(model.input_shape[1])
        # never suggest the special tokens
        self._blocked = np.array([config.PAD_ID, config.UNK_ID])
        self._prefix_cache: dict[str, np.ndarray] = {}

    @classmethod
    def load(cls, model_path: Path | str | None = None, vocab_path: Path | str | None = None) -> "Predictor":
        import keras

        model_path = Path(model_path or config.MODELS_DIR / f"{config.PRODUCTION_MODEL}.keras")
        vocab_path = Path(vocab_path or config.VOCAB_PATH)
        for path in (model_path, vocab_path):
            if not path.exists():
                raise FileNotFoundError(f"{path} not found. Train a model first: python -m src.train --model lstm")
        return cls(keras.models.load_model(model_path, compile=False), Vocabulary.load(vocab_path))

    # -- helpers -----------------------------------------------------------
    def _encode_context(self, tokens: list[str]) -> np.ndarray:
        ids = self.vocab.encode(tokens)[-self.seq_len:]
        ids = [config.PAD_ID] * (self.seq_len - len(ids)) + ids
        return np.asarray([ids], dtype=np.int32)

    def _probabilities(self, tokens: list[str]) -> np.ndarray:
        probs = np.asarray(self.model(self._encode_context(tokens), training=False))[0].astype("float64")
        probs[self._blocked] = 0.0
        return probs

    def _prefix_mask(self, prefix: str) -> np.ndarray:
        if prefix not in self._prefix_cache:
            self._prefix_cache[prefix] = np.array(
                [w.startswith(prefix) for w in self.vocab.itos], dtype="float64"
            )
        return self._prefix_cache[prefix]

    # -- public API --------------------------------------------------------
    def predict_next(self, text: str, k: int = config.TOP_K) -> list[dict]:
        """Return up to k suggestions as [{"word": str, "probability": float}, ...]."""
        if k < 1:
            raise ValueError("k must be at least 1")
        text = text or ""
        tokens = tokenize(text)
        completing = bool(tokens) and not text[-1].isspace()

        if completing:
            prefix, context = tokens[-1], tokens[:-1]
            probs = self._probabilities(context) * self._prefix_mask(prefix)
            total = probs.sum()
            if total > 0:
                probs = probs / total  # renormalise over words that match the prefix
        else:
            probs = self._probabilities(tokens)

        k = min(k, len(probs))
        top = np.argpartition(-probs, k - 1)[:k]
        top = top[np.argsort(-probs[top])]
        return [
            {"word": self.vocab.itos[i], "probability": round(float(probs[i]), 6)}
            for i in top
            if probs[i] > 0
        ]
