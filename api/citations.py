"""Citations: which sources an answer cites, and the one-shot retry when it cites none."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

from api.llm import ChatGenerator, ChatMessage, GenerationError
from api.reranking import RerankedChunk
from api.settings import AppSettings

# One bracketed citation marker: [3], and the list/range forms models also write —
# [1, 2], [1-3], [1–3], [1, 3-4]. Matching only [3] counted "[1, 2]" as uncited, which
# cost a retry LLM call and could end with no sources at all.
_CITATION_RE = re.compile(r"\[(\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*)\]")


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
class SourceCitation:
    """Source metadata returned with each generated answer."""

    source_number: int
    filename: str
    page: int
    section: str
    chunk_id: str
    text: str


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


def needs_citation_retry(answer: str, source_count: int) -> bool:
    """True when an answer should be retried for missing citations.

    Only when sources were available, the answer contains no ``[n]`` marker, and the
    answer did not already declare the context insufficient (a valid uncited answer).
    """

    if source_count <= 0 or _CITATION_RE.search(answer):
        return False
    lowered = _normalize_negation(answer.lower())
    return not any(hint in lowered for hint in _INSUFFICIENCY_HINTS)


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
        retried = retry_uncited_answer(prompt_messages, answer, generator, settings)
        retry_ms = (time.monotonic() - retry_start) * 1000
        if retried is not None:
            answer, final_messages = retried
            cited = cited_sources(answer, all_sources)
            retry_used = True
    return CitationOutcome(
        answer=answer,
        cited=cited,
        retry_used=retry_used,
        prompt_messages=final_messages,
        retry_ms=retry_ms,
    )
