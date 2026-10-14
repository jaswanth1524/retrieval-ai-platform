"""Citations: which sources an answer cites, and the one-shot retry when it cites none."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from api.chunking import sentence_spans
from api.llm import (
    ChatGenerator,
    ChatMessage,
    GenerationError,
    last_finish_reason,
    reset_finish_reason,
)
from api.metrics import citation_retries_total
from api.reranking import RerankedChunk
from api.settings import AppSettings

# One bracketed citation marker: [3], and the list/range forms models also write —
# [1, 2], [1-3], [1–3], [1, 3-4]. Matching only [3] counted "[1, 2]" as uncited, which
# cost a retry LLM call and could end with no sources at all.
# Numbers are capped at 6 digits: int() refuses a string past 4300 digits, so one
# degenerate "[999…9]" in an answer turned a finished stream into a server error.
_CITATION_RE = re.compile(
    r"\[(\d{1,6}(?:\s*[-–]\s*\d{1,6})?(?:\s*,\s*\d{1,6}(?:\s*[-–]\s*\d{1,6})?)*)\]"
)


_CITATION_RANGE_SEP = re.compile(r"\s*[-–]\s*")


CITATION_EXCERPT_MAX_CHARS = 200


CITATION_RETRY_REMINDER = (
    "Your previous answer included no [n] citation markers. Rewrite it, keeping the "
    "same substance, and cite the specific numbered sources that support each claim "
    "(for example [1], [2]). If the context does not actually support an answer, say "
    "the provided documents do not contain enough information."
)


# Phrases that signal the model already declared the context insufficient — a valid
# no-citation answer that must not trigger a citation retry. Matched against
# _normalize_negation'd text, so only spelled-out forms need to be listed here;
# contractions ("don't", "can't") are expanded to match before comparison.
_INSUFFICIENCY_HINTS = (
    "enough information",
    "insufficient",
    "do not contain",
    "does not contain",
    "cannot answer",
)


def _normalize_negation(text: str) -> str:
    """Expand contractions so hint matching doesn't require every surface form.

    Order matters: ``can't`` is irregular (the generic rule alone would produce
    "ca not"), so it's special-cased before the generic ``n't`` -> " not" rule runs.
    The generic rule also mangles irregular contractions outside our hint vocabulary
    (``won't`` -> "wo not", ``shan't`` -> "sha not") but none of _INSUFFICIENCY_HINTS
    is reachable through a mangled stem, so those false expansions are harmless.
    """

    text = text.replace("’", "'")  # curly apostrophe, common in LLM output
    text = re.sub(r"\bcan't\b", "cannot", text)
    return re.sub(r"n't\b", " not", text)


@dataclass(frozen=True)
class CitationSpan:
    """The part of a cited chunk an answer's claim rests on.

    ``start``/``end`` index the chunk's full text (the ``text`` that
    ``GET /documents/{filename}/content`` returns for it) in code points; ``quote`` is
    that slice, so a client can find it even where its string indexing differs.
    """

    start: int
    end: int
    quote: str


@dataclass(frozen=True)
class SourceCitation:
    """Source metadata returned with each generated answer."""

    source_number: int
    filename: str
    page: int
    section: str
    chunk_id: str
    text: str
    # Filled in once the answer exists (attach_spans); empty before generation.
    spans: tuple[CitationSpan, ...] = ()
    # The last page of a passage that crosses a page break; None on one page.
    page_end: int | None = None


def citation_from_dict(data: dict[str, Any]) -> SourceCitation:
    """A citation back from its ``dataclasses.asdict`` form (a ``done`` frame's sources)."""

    fields = {key: value for key, value in data.items() if key != "spans"}
    spans = tuple(CitationSpan(**span) for span in data.get("spans") or ())
    return SourceCitation(**fields, spans=spans)


def source_citations(chunks: Sequence[RerankedChunk]) -> list[SourceCitation]:
    """Return source metadata for chunks included in the prompt context."""

    return [
        SourceCitation(
            source_number=index,
            filename=chunk.filename,
            page=chunk.page,
            section=chunk.section,
            chunk_id=chunk.chunk_id,
            text=_truncate_excerpt(chunk.text),
            page_end=chunk.page_end,
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


def _truncate_excerpt(text: str, limit: int = CITATION_EXCERPT_MAX_CHARS) -> str:
    """Collapse whitespace and truncate a chunk's text to a short citation excerpt."""

    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "..."


def cited_sources(
    answer: str,
    sources: Sequence[SourceCitation],
) -> list[SourceCitation]:
    """Return only the sources the answer actually cites via ``[n]`` markers.

    An answer with zero ``[n]`` markers is treated as ungrounded/insufficient — it
    gets zero sources rather than every candidate chunk attached regardless of
    relevance.
    """

    available = {source.source_number for source in sources}
    referenced = cited_numbers(answer, max(available, default=0))
    used = referenced & available
    return [source for source in sources if source.source_number in used]


def cited_numbers(text: str, max_number: int) -> set[int]:
    """Every source number ``text`` cites, with ranges expanded.

    A range is clamped to ``max_number`` (the highest source number on offer), so a
    stray ``[1-100000]`` costs nothing; a backwards range like ``[3-1]`` is ignored.
    """

    numbers: set[int] = set()
    for marker in _CITATION_RE.findall(text):
        for part in marker.split(","):
            bounds = _CITATION_RANGE_SEP.split(part.strip())
            start = int(bounds[0])
            end = int(bounds[-1])
            if start <= end:
                numbers.update(range(start, min(end, max_number) + 1))
    return numbers


# Exact spans. No model involved: each answer sentence's content words are matched
# against the cited chunk's sentences (and pairs of them, for a claim that spans two),
# and the best-covering one is quoted. Below these bars nothing is claimed, rather than
# highlighting a passage that merely shares a word or two with the answer.
SPAN_MIN_SHARED_WORDS = 2
SPAN_MIN_COVERAGE = 0.3
MAX_SPANS_PER_SOURCE = 3
_WORD_RE = re.compile(r"\w+")
_STOPWORDS = frozenset(
    "a an and are as at be been but by can could do does for from had has have how if in "
    "into is it its may might must no not of on or our shall should so such than that the "
    "their them then there these they this those to was we were what when where which "
    "who whom why will with would you your".split()
)
# Answer sentences: after a sentence end, and at every line break (lists, headings).
_ANSWER_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
_LEADING_MARKERS_RE = re.compile(r"(?:\s*" + _CITATION_RE.pattern + r")+")


_NUMBER_UNITS = {
    word: value
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_NUMBER_TENS = {
    word: value * 10
    for value, word in enumerate(
        "twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2
    )
}
_NUMBER_SCALES = {"hundred": 100, "thousand": 1000}


def _words(text: str) -> list[str]:
    """Lower-cased words, with spelled-out numbers as digits: a document saying
    "forty-five euros" and an answer saying "45 euros" share the word "45"."""

    words: list[str] = []
    total = current = 0
    in_number = False
    last_unit = False
    previous_end = 0

    def flush() -> None:
        nonlocal total, current, in_number, last_unit
        if in_number:
            words.append(str(total + current))
        total = current = 0
        in_number = last_unit = False

    for match in _WORD_RE.finditer(text):
        word = match.group(0).lower()
        # A number runs across spaces and hyphens only: "five, six" is two numbers.
        if in_number and text[previous_end : match.start()].strip(" -"):
            flush()
        previous_end = match.end()
        if word in _NUMBER_UNITS and last_unit:
            flush()  # "one two": two numbers, not three
        if word in _NUMBER_UNITS or word in _NUMBER_TENS:
            current += _NUMBER_UNITS.get(word, 0) + _NUMBER_TENS.get(word, 0)
            in_number = True
            last_unit = word in _NUMBER_UNITS
        elif word in _NUMBER_SCALES and in_number:
            last_unit = False
            if _NUMBER_SCALES[word] == 1000:
                total, current = (total + current) * 1000, 0
            else:
                current = max(current, 1) * 100
        elif word == "and" and in_number:
            continue  # "one hundred and five"
        else:
            flush()
            words.append(word)
    flush()
    return words


def _content_words(text: str) -> set[str]:
    return {
        word
        for word in _words(text)
        if word not in _STOPWORDS and (len(word) > 1 or word.isdigit())
    }


def _claims_by_source(answer: str, max_number: int) -> dict[int, list[set[str]]]:
    """For each cited source number, the content words of every sentence citing it."""

    claims: dict[int, list[set[str]]] = {}
    previous: set[str] | None = None
    for piece in _ANSWER_SENTENCE_RE.split(answer):
        if not piece.strip():
            continue
        # "...twenty days. [2] Next..." splits after the period: a marker opening a
        # piece belongs to the sentence before it.
        leading = _LEADING_MARKERS_RE.match(piece)
        if leading is not None and previous is not None:
            for number in cited_numbers(leading.group(0), max_number):
                claims.setdefault(number, []).append(previous)
            piece = piece[leading.end() :]
        words = _content_words(_CITATION_RE.sub(" ", piece))
        for number in cited_numbers(piece, max_number):
            claims.setdefault(number, []).append(words)
        previous = words
    return claims


def _best_span(
    claim: set[str], text: str, sentences: Sequence[tuple[int, int]]
) -> tuple[tuple[float, int, int], int, int] | None:
    # A pair only when each of its sentences carries part of the claim by itself:
    # otherwise one shared word ("staff") pulled an unrelated sentence into the quote.
    relevant = [
        len(claim & _content_words(text[start:end])) >= SPAN_MIN_SHARED_WORDS
        for start, end in sentences
    ]
    candidates = [*sentences]
    candidates += [
        (sentences[i][0], sentences[i + 1][1])
        for i in range(len(sentences) - 1)
        if relevant[i] and relevant[i + 1]
    ]
    best: tuple[tuple[float, int, int], int, int] | None = None
    for start, end in candidates:
        words = _content_words(text[start:end])
        shared = len(claim & words)
        if shared < SPAN_MIN_SHARED_WORDS:
            continue
        coverage = shared / len(claim)
        if coverage < SPAN_MIN_COVERAGE:
            continue
        # Most of the claim covered, then the tightest passage, then the earliest.
        key = (coverage, -len(words), -start)
        if best is None or key > best[0]:
            best = (key, start, end)
    return best


def attach_spans(
    answer: str, cited: Sequence[SourceCitation], chunks: Sequence[RerankedChunk]
) -> list[SourceCitation]:
    """``cited`` with the passage of each source its citing sentences rest on.

    ``chunks`` are the prompt's sources in order (source ``n`` is ``chunks[n - 1]``).
    The 200-character excerpt a citation carries is the chunk's start, which often
    isn't the sentence the answer used; the span is.
    """

    claims = _claims_by_source(answer, len(chunks))
    result: list[SourceCitation] = []
    for source in cited:
        index = source.source_number - 1
        if not 0 <= index < len(chunks) or source.source_number not in claims:
            result.append(source)
            continue
        text = chunks[index].text
        sentences = sentence_spans(text)
        found: dict[tuple[int, int], tuple[float, int, int]] = {}
        for claim in claims[source.source_number]:
            if not claim:
                continue
            best = _best_span(claim, text, sentences)
            if best is not None:
                key, start, end = best
                found[(start, end)] = max(key, found.get((start, end), key))
        kept: list[tuple[int, int]] = []
        for start, end in sorted(found, key=lambda span: found[span], reverse=True):
            # A pair that already contains a kept sentence (or the reverse) is one passage.
            if any(start < other_end and other_start < end for other_start, other_end in kept):
                continue
            kept.append((start, end))
            if len(kept) == MAX_SPANS_PER_SOURCE:
                break
        spans = tuple(CitationSpan(start, end, text[start:end]) for start, end in sorted(kept))
        result.append(replace(source, spans=spans) if spans else source)
    return result


def needs_citation_retry(answer: str, source_count: int) -> bool:
    """True when an answer should be retried for missing citations.

    Only when sources were available, the answer contains no ``[n]`` marker, and the
    answer did not already declare the context insufficient (a valid uncited answer).
    """

    if source_count <= 0 or _CITATION_RE.search(answer):
        return False
    return not _states_insufficiency(answer)


# How well an answer is grounded, sent with every answer. A string, not an enum, so a
# new value can be added without breaking a client: an unknown one reads as "answered".
ANSWER_STATUSES = ("answered", "insufficient_context", "no_context", "uncited")


def answer_status(answer: str, cited_count: int, source_count: int) -> str:
    """``no_context`` when nothing cleared the rerank bar, ``insufficient_context`` when
    the model said the sources don't cover it and cited none of them, ``uncited`` when
    it answered without citing any source (even after the citation retry, or citing
    only numbers that aren't sources), else ``answered``.

    Both conditions, as for the citation retry: "the documents don't cover X, but Y
    [1]" is a partial answer, and a hint word alone would have flagged it. An uncited
    answer used to read as ``answered``, so nothing told the reader it had no sources.
    """

    if source_count <= 0:
        return "no_context"
    if cited_count == 0:
        return "insufficient_context" if _states_insufficiency(answer) else "uncited"
    return "answered"


def _states_insufficiency(answer: str) -> bool:
    lowered = _normalize_negation(answer.lower())
    return any(hint in lowered for hint in _INSUFFICIENCY_HINTS)


def retry_uncited_answer(
    prompt_messages: Sequence[ChatMessage],
    first_answer: str,
    generator: ChatGenerator,
    settings: AppSettings,
) -> tuple[str, list[ChatMessage]] | None:
    """One-shot retry that reminds the model to cite.

    Returns ``(cited_answer, messages_actually_sent)``, or None when the retry fails or
    still produces no citations — in which case the caller keeps the original answer.
    Never loops.

    Takes the already-built grounded prompt rather than rebuilding it from
    ``query``/``context_chunks``/``history``: the retry must continue the *same*
    conversation the first answer came from, and reconstructing it here was a second
    derivation that could silently drift from the one generation actually used.
    """

    messages: list[ChatMessage] = [
        *prompt_messages,
        {"role": "assistant", "content": first_answer},
        {"role": "user", "content": CITATION_RETRY_REMINDER},
    ]
    try:
        retried = generator.complete(messages, settings).strip()
    except GenerationError:
        return None
    if not retried or not _CITATION_RE.search(retried):
        return None
    return retried, messages


@dataclass(frozen=True)
class CitationOutcome:
    """Result of citation filtering, including what actually produced the answer.

    ``prompt_messages`` is the message list of the call the returned ``answer`` came
    from — the retry's continuation when ``retry_used`` is True, the original prompt
    otherwise. Traces record this rather than the first prompt, so a debug trace can
    never show a prompt the model was not asked.
    """

    answer: str
    cited: list[SourceCitation]
    retry_used: bool
    prompt_messages: list[ChatMessage]
    # Wall-clock of the retry's LLM round trip, 0.0 when it didn't fire. Measured here
    # rather than by the callers because this is the only place that knows whether the
    # call happened at all — and it happens in both response modes.
    retry_ms: float = 0.0


def will_retry_citations(
    answer: str, all_sources: Sequence[SourceCitation], settings: AppSettings
) -> bool:
    """Whether ``finalize_citations`` will make its retry call for this answer.

    Split out so the streaming path can announce the retry *before* it blocks, without
    restating the condition — two copies would drift the first time either changed.
    """

    return (
        not cited_sources(answer, all_sources)
        and settings.citation_retry_enabled
        and needs_citation_retry(answer, len(all_sources))
    )


def finalize_citations(
    prompt_messages: Sequence[ChatMessage],
    answer: str,
    all_sources: Sequence[SourceCitation],
    generator: ChatGenerator,
    settings: AppSettings,
) -> CitationOutcome:
    """Filter ``answer`` to its cited sources, retrying once if none are cited.

    Shared by both response modes (sync ``generate_grounded_answer`` and the streaming
    path in ``RagPipeline.answer_stream``) so the retry-trigger condition can't drift
    between them. ``prompt_messages`` is the prompt that produced ``answer``; the
    returned outcome carries whichever prompt produced its final answer.

    The ``not cited`` guard is redundant with ``needs_citation_retry`` — that returns
    False whenever the answer contains any ``[n]`` marker, and ``cited`` is only
    non-empty when a marker matched — so it never changes the outcome. It's kept as a
    cheap, explicit statement of the intent ("only retry when we ended up with zero
    sources"). Note the deliberate gap it does *not* close: an answer citing ``[9]``
    when only 3 sources exist yields empty ``cited`` but no retry, because the marker
    check in ``needs_citation_retry`` short-circuits first. Retrying there would mean
    re-prompting a model that did cite, just badly, so it's left alone.
    """

    cited = cited_sources(answer, all_sources)
    final_messages = list(prompt_messages)
    retry_used = False
    retry_ms = 0.0
    if will_retry_citations(answer, all_sources, settings):
        # Timed around the call itself, not around the `if` — a retry that fails or
        # still comes back uncited cost just as much wall-clock as one that worked, and
        # attributing it to generate_ms is what made this stage invisible before.
        retry_start = time.monotonic()
        first_finish = last_finish_reason()
        citation_retries_total.inc()
        retried = retry_uncited_answer(prompt_messages, answer, generator, settings)
        retry_ms = (time.monotonic() - retry_start) * 1000
        if retried is not None:
            answer, final_messages = retried
            cited = cited_sources(answer, all_sources)
            retry_used = True
        else:
            # The first answer stands, and so does how it ended (truncated or not).
            reset_finish_reason(first_finish)
    return CitationOutcome(
        answer=answer,
        cited=cited,
        retry_used=retry_used,
        prompt_messages=final_messages,
        retry_ms=retry_ms,
    )
