"""Token counting for chunk sizing.

Chunking must respect the dense embedding model's subword-token limit (bge-small hard-
truncates at 512), but loading the real tokenizer needs a (network) model fetch. This
module exposes a ``TokenCounter`` seam so ingestion counts real subword tokens when the
tokenizer is available and degrades to a word-based heuristic when it isn't — ingest
never hard-fails just because the tokenizer couldn't be fetched (offline build, etc.).
"""

from __future__ import annotations

import logging
import math
from typing import Protocol

logger = logging.getLogger(__name__)

# English words expand to ~1.3-1.8 subword tokens each; 1.6 is a deliberately slight
# over-estimate so the heuristic errs toward smaller (safe) chunks, never overflow.
_HEURISTIC_TOKENS_PER_WORD = 1.6


class TokenCounter(Protocol):
    """Counts model tokens for a piece of text, excluding special tokens.

    Implementations may expose ``special_tokens_per_sequence`` (tokens the model adds
    around each input); chunking reserves that many once per chunk, 0 if absent.
    """

    def count(self, text: str) -> int: ...


# [CLS] + [SEP]: what bge (and BERT-family models generally) wrap around each input.
_BERT_SPECIAL_TOKENS = 2


class HeuristicTokenCounter:
    """Word-count-based token estimate; no model, always available."""

    special_tokens_per_sequence = _BERT_SPECIAL_TOKENS

    def count(self, text: str) -> int:
        words = len(text.split())
        return math.ceil(words * _HEURISTIC_TOKENS_PER_WORD)


class HFTokenCounter:
    """Counts subword tokens with the dense model's own tokenizer, lazy-loaded once.

    The tokenizer is fetched on the first ``count`` call, not at construction, so
    building the counter is free and a fetch failure degrades to the heuristic (logged
    once) rather than raising — a document must still ingest without network access.
    """

    special_tokens_per_sequence = _BERT_SPECIAL_TOKENS

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._tokenizer: object | None = None
        self._fallback = HeuristicTokenCounter()
        self._degraded = False

    def count(self, text: str) -> int:
        if self._degraded:
            return self._fallback.count(text)
        if self._tokenizer is None and not self._load():
            return self._fallback.count(text)
        # Without special tokens: [CLS]/[SEP] wrap the whole embedded string once, and
        # chunking reserves them once in its budget. Counting them per call added 2 to
        # every sentence (and every word, when an oversized sentence is split word by
        # word), leaving windows well short of the budget they were sized for.
        encode = self._tokenizer.encode(  # type: ignore[union-attr]
            text, add_special_tokens=False
        )
        return len(encode.ids)

    def _load(self) -> bool:
        try:
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_pretrained(self._model_name)
            return True
        except Exception as exc:  # noqa: BLE001 - any failure must degrade, not crash ingest
            self._degraded = True
            logger.warning(
                "Could not load tokenizer for %s (%s); falling back to heuristic token "
                "counting for chunk sizing.",
                self._model_name,
                exc,
            )
            return False


def make_token_counter(model_name: str) -> TokenCounter:
    """Build the token counter for a dense embedding model name."""

    return HFTokenCounter(model_name)
