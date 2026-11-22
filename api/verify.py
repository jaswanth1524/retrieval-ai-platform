"""Checking a text against the documents, sentence by sentence (``POST
/questions/verify``): for each sentence that claims something, the passage and the
words in it that support it — or that none does, or that the passage found states other
numbers. The same hybrid search and rerank a question uses, and the same span and
number checks an answer's citations get; no LLM call."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from api.citations import (
    _ANSWER_SENTENCE_RE,
    _CITATION_RE,
    SPAN_MIN_SHARED_WORDS,
    _best_span,
    _claim_numbers,
    _content_words,
    _source_numbers,
    _span_index,
)
from api.reranking import RerankedChunk

VerifyStatus = Literal["supported", "numbers_differ", "unsupported", "no_claim"]

# Sentences checked in one request (each is a search and a rerank).
MAX_VERIFY_SENTENCES = 20
# Passages looked through for each sentence.
VERIFY_PASSAGES = 3


@dataclass(frozen=True)
class SentenceCheck:
    sentence: str
    status: VerifyStatus
    chunk: RerankedChunk | None = None
    quote: str | None = None
    # Numbers the sentence states that the passage found doesn't.
    missing_numbers: tuple[str, ...] = ()


def sentences_of(text: str) -> list[str]:
    """``text``'s sentences (lines and sentence ends), citation markers kept."""

    return [piece.strip() for piece in _ANSWER_SENTENCE_RE.split(text) if piece.strip()]


def check_sentence(
    sentence: str, search: Callable[[str], Sequence[RerankedChunk]]
) -> SentenceCheck:
    """One sentence against the passages ``search`` finds for it."""

    plain = " ".join(_CITATION_RE.sub(" ", sentence).split())
    words = _content_words(plain)
    if len(words) < SPAN_MIN_SHARED_WORDS:
        return SentenceCheck(sentence, "no_claim")
    numbers = _claim_numbers(plain)
    differing: SentenceCheck | None = None
    for chunk in list(search(plain))[:VERIFY_PASSAGES]:
        found = _best_span(words, _span_index(chunk.text))
        if found is None:
            continue
        _, start, end = found
        quote = chunk.text[start:end].strip()
        missing = tuple(sorted(numbers - _source_numbers(chunk.text)))
        if not missing:
            return SentenceCheck(sentence, "supported", chunk, quote)
        if differing is None:
            differing = SentenceCheck(sentence, "numbers_differ", chunk, quote, missing)
    return differing or SentenceCheck(sentence, "unsupported")


class VerifyAbandoned(Exception):
    """The client left: the sentences not yet checked are not searched for."""


def verify_text(
    text: str,
    search: Callable[[str], Sequence[RerankedChunk]],
    should_stop: Callable[[], bool] | None = None,
) -> tuple[list[SentenceCheck], bool]:
    """Each sentence of ``text`` checked (at most MAX_VERIFY_SENTENCES), and whether
    the text had more than that. ``should_stop`` (the client has gone) is asked before
    each sentence's search: ``VerifyAbandoned``."""

    sentences = sentences_of(text)
    checks: list[SentenceCheck] = []
    for sentence in sentences[:MAX_VERIFY_SENTENCES]:
        if should_stop is not None and should_stop():
            raise VerifyAbandoned
        checks.append(check_sentence(sentence, search))
    return checks, len(sentences) > MAX_VERIFY_SENTENCES


__all__ = ["SentenceCheck", "VerifyAbandoned", "check_sentence", "sentences_of", "verify_text"]
