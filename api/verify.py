"""Checking a text against the documents, sentence by sentence (``POST
/questions/verify``): for each sentence that claims something, the passage and the
words in it that support it — or that none does, or that the passage found states other
numbers. The same hybrid search and rerank a question uses, and the same span and
number checks an answer's citations get; no LLM call."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
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
    _header_numbers,
    _source_numbers,
    _span_index,
    _SpanIndex,
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


# Chinese and Japanese end a sentence with no space after it, and write no spaces
# between words: an answer in them was one "sentence" of one "word" (no_claim).
# The end mark, then the citation markers and closing quotes that belong to the sentence
# ("…28天假期。[1]」"): the split goes after them, or "[1]" was a sentence of its own.
_CJK_SENTENCE_END_RE = re.compile(r"[。！？]+(?:[」』）)”’]|\s*" + _CITATION_RE.pattern + r")*")
_CJK_RUN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


# A sentence ended inside a bracket or quote: "(… [2].) You can ask …" was one
# sentence, checked against both sources and flagged. Only for checking: the citation
# code's own splitter (_ANSWER_SENTENCE_RE) is left as it is.
_CLOSED_SENTENCE_RE = re.compile(r"(?<=[.!?][)\]\"'\u201d\u2019])\s+")


def sentences_of(text: str) -> list[str]:
    """``text``'s sentences (lines and sentence ends), citation markers kept."""

    pieces = (
        part
        for piece in _ANSWER_SENTENCE_RE.split(text)
        for closed in _CLOSED_SENTENCE_RE.split(piece)
        for part in _cjk_split(closed)
    )
    return [piece.strip() for piece in pieces if piece.strip()]


def _cjk_split(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for end in _CJK_SENTENCE_END_RE.finditer(text):
        # Before a space too: "。[1] …" has none the other splitter splits at.
        if end.end() < len(text):
            parts.append(text[start : end.end()])
            start = end.end()
    parts.append(text[start:])
    return parts


def _cjk_bigrams(text: str) -> set[str]:
    """Each pair of adjacent Chinese/Japanese characters: the words of a text that
    writes none apart (empty for any other text, which is matched as before)."""

    return {run[i : i + 2] for run in _CJK_RUN_RE.findall(text) for i in range(len(run) - 1)}


def _apart(text: str) -> str:
    """``text`` with its runs of characters blanked, leaving what is written between
    them as words of its own: ``\w+`` reads "使用Kubernetes部署" and "28天" as one."""

    return _CJK_RUN_RE.sub(" ", text)


class _CjkSpanIndex(_SpanIndex):
    """A passage's sentences with their character pairs as words, and the Latin words
    and numbers written against them."""

    def words(self, start: int, end: int) -> set[str]:
        segment = self.text[start:end]
        return _content_words(_apart(segment)) | _cjk_bigrams(segment)


def _claim_words(plain: str) -> tuple[set[str], Callable[[str], _SpanIndex]]:
    bigrams = _cjk_bigrams(plain)
    if not bigrams:
        return _content_words(plain), _span_index
    return _content_words(_apart(plain)) | bigrams, _CjkSpanIndex


def check_sentence(
    sentence: str,
    search: Callable[[str], Sequence[RerankedChunk]],
    widen: Callable[[RerankedChunk], str] | None = None,
    passages: int = VERIFY_PASSAGES,
    together: bool = False,
) -> SentenceCheck:
    """One sentence against the passages ``search`` finds for it. ``together``: the
    passages are the ones the sentence cites, and its numbers may come from any of them
    (as ``unsupported_numbers`` counts them): "the current policy says fourteen days
    [1]; the archived 2019 one … [2]". ``widen`` gives a
    passage's text with its neighbours: a number missing from the passage alone is
    looked for there before the sentence is called ``numbers_differ`` (a question's
    prompt has that text, so its answers state such numbers)."""

    plain = " ".join(_CITATION_RE.sub(" ", sentence).split())
    words, index_of = _claim_words(plain)
    # "There are two values in the provided documents:" introduces what follows (a
    # list on the next lines, checked line by line); it states nothing itself.
    if len(words) < SPAN_MIN_SHARED_WORDS or plain.endswith((":", "：")):
        return SentenceCheck(sentence, "no_claim")
    cjk = index_of is _CjkSpanIndex
    # One reading of numbers for the sentence and the passage alike ("3.5" spaced and
    # unspaced read differently with two).
    numbers = _claim_numbers(_apart(plain) if cjk else plain)

    def stated(text: str, chunk: RerankedChunk) -> set[str]:
        # With what the prompt's source line showed (its file name and section): "the
        # archived 2019 policy" of travel_policy_2019_archived.md, as unsupported_numbers
        # already allows. gpt-5-mini names documents that way.
        return _source_numbers(_apart(text) if cjk else text) | _header_numbers(chunk)

    differing: list[SentenceCheck] = []
    chunks = list(search(plain))[:passages]
    for chunk in chunks:
        found = _best_span(words, index_of(chunk.text))
        if found is None:
            continue
        _, start, end = found
        quote = chunk.text[start:end].strip()
        missing = tuple(sorted(numbers - stated(chunk.text, chunk)))
        if not missing:
            return SentenceCheck(sentence, "supported", chunk, quote)
        differing.append(SentenceCheck(sentence, "numbers_differ", chunk, quote, missing))
    if together and differing:
        everywhere = set().union(*(stated(chunk.text, chunk) for chunk in chunks))
        if numbers <= everywhere:
            return replace(differing[0], status="supported", missing_numbers=())
    # No passage states the numbers itself: one whose neighbours do still supports it
    # (looked at only now, so a passage holding the numbers is the one quoted).
    if widen is not None:
        for check in differing:
            assert check.chunk is not None
            if not set(check.missing_numbers) - stated(widen(check.chunk), check.chunk):
                return replace(check, status="supported", missing_numbers=())
        if not differing:
            # No passage has the words: the text around one may (a question's prompt
            # had it, so an answer says what a neighbour holds).
            for chunk in chunks:
                wide = widen(chunk)
                found = _best_span(words, index_of(wide))
                if found is None:
                    continue
                _, start, end = found
                quote = wide[start:end].strip()
                missing = tuple(sorted(numbers - stated(wide, chunk)))
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
    whole: Mapping[str, RerankedChunk] | None = None,
) -> list[SentenceCheck]:
    """An answer's sentences checked against its own sources — no search, no LLM: each
    citing sentence against the passages it cites, an uncited one against all of them.
    What the answer says that its passages don't (``unsupported``) or says with other
    numbers (``numbers_differ``), sentence by sentence (at most MAX_VERIFY_SENTENCES).
    ``whole`` holds the cited passages in full by chunk id: a source's own ``text`` is
    a short excerpt, which a claim backed further down its passage isn't found in."""

    if not sources:
        return []
    by_number = {source.source_number: source for source in sources}
    passages = {
        source.source_number: _as_chunk(source, (whole or {}).get(source.chunk_id))
        for source in sources
    }
    checks: list[SentenceCheck] = []
    for sentence in sentences_of(answer)[:MAX_VERIFY_SENTENCES]:
        # "The documents don't cover the expense policy." claims nothing to look for:
        # flagged as unsupported, it read as a mistake in a partial answer (gpt-4o-mini).
        if only_declines(sentence):
            checks.append(SentenceCheck(sentence, "no_claim"))
            continue
        cited = sorted(cited_numbers(sentence, max(by_number)) & by_number.keys())
        if not cited and _CITATION_RE.search(sentence):
            # It cites only sources the answer doesn't have ("[9]" of five): nothing
            # backs it, where it used to be checked against every passage as uncited.
            checks.append(SentenceCheck(sentence, "unsupported"))
            continue
        chosen = [passages[number] for number in cited] or list(passages.values())
        checks.append(
            check_sentence(
                sentence,
                lambda _, c=chosen: c,  # type: ignore[misc]
                widen,
                passages=len(chosen),
                together=len(cited) > 1,
            )
        )
    return checks


def recheck(
    answer: str,
    cited: dict[int, Sequence[RerankedChunk] | Callable[[str], Sequence[RerankedChunk]]],
    widen: Callable[[RerankedChunk], str] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[list[SentenceCheck], bool]:
    """An answer checked again against what its sources are now: ``cited`` maps each
    source number to its passage as indexed now, or to a search within its document
    (the passage changed), or to nothing (the document is gone). Each sentence against
    the sources it cites, an uncited one against all of them (at most
    MAX_VERIFY_SENTENCES), and whether the answer had more sentences than that.
    ``should_stop`` (the client has gone) is asked before each sentence:
    ``VerifyAbandoned``."""

    numbers = sorted(cited)
    sentences = sentences_of(answer)
    checks: list[SentenceCheck] = []
    for sentence in sentences[:MAX_VERIFY_SENTENCES]:
        if should_stop is not None and should_stop():
            raise VerifyAbandoned
        wanted = sorted(cited_numbers(sentence, max(numbers, default=0)) & set(numbers))
        if not wanted and _CITATION_RE.search(sentence) and not only_declines(sentence):
            # It cites only sources there are none of: as self_check says, unbacked.
            checks.append(SentenceCheck(sentence, "unsupported"))
            continue

        def search(plain: str, wanted: list[int] = wanted or numbers) -> list[RerankedChunk]:
            found: list[RerankedChunk] = []
            seen: set[str] = set()
            for number in wanted:
                source = cited[number]
                for chunk in source(plain) if callable(source) else source:
                    # Two sources in one changed document share its search's passages.
                    if chunk.chunk_id not in seen:
                        seen.add(chunk.chunk_id)
                        found.append(chunk)
            return found

        checks.append(check_sentence(sentence, search, widen, passages=MAX_RECHECK_PASSAGES))
    return checks, len(sentences) > MAX_VERIFY_SENTENCES


# Passages one sentence of a recheck is held against (its cited ones, or a changed
# document's closest few).
MAX_RECHECK_PASSAGES = 20


def _as_chunk(source: SourceCitation, stored: RerankedChunk | None = None) -> RerankedChunk:
    if stored is not None:
        return replace(stored, rerank_score=source.rerank_score or 0.0)
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
