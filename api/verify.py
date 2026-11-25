"""Checking a text against the documents, sentence by sentence (``POST
/questions/verify``): for each sentence that claims something, the passage and the
words in it that support it — or that none does, or that the passage found states other
numbers. The same hybrid search and rerank a question uses, and the same span and
number checks an answer's citations get; no LLM call."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from api.citations import (
    _ANSWER_SENTENCE_RE,
    _CITATION_RE,
    SPAN_MIN_SHARED_WORDS,
    SourceCitation,
    _best_span,
    _claim_numbers,
    _content_words,
    _source_numbers,
    _span_index,
    cited_numbers,
    only_declines,
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
    sentence: str,
    search: Callable[[str], Sequence[RerankedChunk]],
    widen: Callable[[RerankedChunk], str] | None = None,
    passages: int = VERIFY_PASSAGES,
) -> SentenceCheck:
    """One sentence against the passages ``search`` finds for it. ``widen`` gives a
    passage's text with its neighbours: a number missing from the passage alone is
    looked for there before the sentence is called ``numbers_differ`` (a question's
    prompt has that text, so its answers state such numbers)."""

    plain = " ".join(_CITATION_RE.sub(" ", sentence).split())
    words = _content_words(plain)
    if len(words) < SPAN_MIN_SHARED_WORDS:
        return SentenceCheck(sentence, "no_claim")
    numbers = _claim_numbers(plain)
    differing: list[SentenceCheck] = []
    chunks = list(search(plain))[:passages]
    for chunk in chunks:
        found = _best_span(words, _span_index(chunk.text))
        if found is None:
            continue
        _, start, end = found
        quote = chunk.text[start:end].strip()
        missing = tuple(sorted(numbers - _source_numbers(chunk.text)))
        if not missing:
            return SentenceCheck(sentence, "supported", chunk, quote)
        differing.append(SentenceCheck(sentence, "numbers_differ", chunk, quote, missing))
    # No passage states the numbers itself: one whose neighbours do still supports it
    # (looked at only now, so a passage holding the numbers is the one quoted).
    if widen is not None:
        for check in differing:
            assert check.chunk is not None
            if not set(check.missing_numbers) - _source_numbers(widen(check.chunk)):
                return replace(check, status="supported", missing_numbers=())
        if not differing:
            # No passage has the words: the text around one may (a question's prompt
            # had it, so an answer says what a neighbour holds).
            for chunk in chunks:
                wide = widen(chunk)
                found = _best_span(words, _span_index(wide))
                if found is None:
                    continue
                _, start, end = found
                quote = wide[start:end].strip()
                missing = tuple(sorted(numbers - _source_numbers(wide)))
                if not missing:
                    return SentenceCheck(sentence, "supported", chunk, quote)
                differing.append(SentenceCheck(sentence, "numbers_differ", chunk, quote, missing))
    return differing[0] if differing else SentenceCheck(sentence, "unsupported")


class VerifyAbandoned(Exception):
    """The client left: the sentences not yet checked are not searched for."""


def verify_text(
    text: str,
    search: Callable[[str], Sequence[RerankedChunk]],
    should_stop: Callable[[], bool] | None = None,
    widen: Callable[[RerankedChunk], str] | None = None,
) -> tuple[list[SentenceCheck], bool]:
    """Each sentence of ``text`` checked (at most MAX_VERIFY_SENTENCES), and whether
    the text had more than that. ``should_stop`` (the client has gone) is asked before
    each sentence's search: ``VerifyAbandoned``. A sentence repeated is searched once."""

    sentences = sentences_of(text)
    checks: list[SentenceCheck] = []
    seen: dict[str, SentenceCheck] = {}
    for sentence in sentences[:MAX_VERIFY_SENTENCES]:
        same = " ".join(sentence.split()).casefold()
        if same in seen:
            checks.append(replace(seen[same], sentence=sentence))
            continue
        if should_stop is not None and should_stop():
            raise VerifyAbandoned
        seen[same] = check_sentence(sentence, search, widen)
        checks.append(seen[same])
    return checks, len(sentences) > MAX_VERIFY_SENTENCES


def self_check(
    answer: str,
    sources: Sequence[SourceCitation],
    widen: Callable[[RerankedChunk], str] | None = None,
) -> list[SentenceCheck]:
    """An answer's sentences checked against its own sources — no search, no LLM: each
    citing sentence against the passages it cites, an uncited one against all of them.
    What the answer says that its passages don't (``unsupported``) or says with other
    numbers (``numbers_differ``), sentence by sentence (at most MAX_VERIFY_SENTENCES)."""

    if not sources:
        return []
    by_number = {source.source_number: source for source in sources}
    passages = {source.source_number: _as_chunk(source) for source in sources}
    checks: list[SentenceCheck] = []
    for sentence in sentences_of(answer)[:MAX_VERIFY_SENTENCES]:
        # "The documents don't cover the expense policy." claims nothing to look for:
        # flagged as unsupported, it read as a mistake in a partial answer (gpt-4o-mini).
        if only_declines(sentence):
            checks.append(SentenceCheck(sentence, "no_claim"))
            continue
        cited = sorted(cited_numbers(sentence, max(by_number)) & by_number.keys())
        chosen = [passages[number] for number in cited] or list(passages.values())
        checks.append(
            check_sentence(sentence, lambda _, c=chosen: c, widen, passages=len(chosen))  # type: ignore[misc]
        )
    return checks


def recheck(
    answer: str,
    cited: dict[int, Sequence[RerankedChunk] | Callable[[str], Sequence[RerankedChunk]]],
    widen: Callable[[RerankedChunk], str] | None = None,
) -> list[SentenceCheck]:
    """An answer checked again against what its sources are now: ``cited`` maps each
    source number to its passage as indexed now, or to a search within its document
    (the passage changed), or to nothing (the document is gone). Each sentence against
    the sources it cites, an uncited one against all of them."""

    numbers = sorted(cited)
    checks: list[SentenceCheck] = []
    for sentence in sentences_of(answer)[:MAX_VERIFY_SENTENCES]:
        wanted = sorted(cited_numbers(sentence, max(numbers, default=0)) & set(numbers))

        def search(plain: str, wanted: list[int] = wanted or numbers) -> list[RerankedChunk]:
            found: list[RerankedChunk] = []
            for number in wanted:
                source = cited[number]
                found.extend(source(plain) if callable(source) else source)
            return found

        checks.append(check_sentence(sentence, search, widen, passages=MAX_RECHECK_PASSAGES))
    return checks


# Passages one sentence of a recheck is held against (its cited ones, or a changed
# document's closest few).
MAX_RECHECK_PASSAGES = 20


def _as_chunk(source: SourceCitation) -> RerankedChunk:
    return RerankedChunk(
        point_id=source.chunk_id,
        filename=source.filename,
        page=source.page,
        section=source.section,
        chunk_id=source.chunk_id,
        text=source.text,
        retrieval_score=0.0,
        rerank_score=source.rerank_score or 0.0,
    )


__all__ = [
    "SentenceCheck",
    "VerifyAbandoned",
    "check_sentence",
    "recheck",
    "self_check",
    "sentences_of",
    "verify_text",
]
