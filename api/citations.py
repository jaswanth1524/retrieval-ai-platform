"""Citations: which sources an answer cites, and the one-shot retry when it cites none."""

from __future__ import annotations

import functools
import logging
import re
import time
import unicodedata
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
from api.metrics import citation_backfills_total, citation_retries_total
from api.prompting import (
    CHARS_PER_TOKEN_ESTIMATE,
    clip_text,
    clip_to_weight,
    estimate_prompt_tokens,
    prompt_token_budget,
)
from api.reranking import RerankedChunk
from api.settings import AppSettings

logger = logging.getLogger(__name__)

# One bracketed citation marker: [3], and the list/range forms models also write —
# [1, 2], [1-3], [1–3], [1, 3-4]. Matching only [3] counted "[1, 2]" as uncited, which
# cost a retry LLM call and could end with no sources at all.
# Numbers are capped at 6 digits: int() refuses a string past 4300 digits, so one
# degenerate "[999…9]" in an answer turned a finished stream into a server error.
_CITATION_RE = re.compile(
    r"\[(\d{1,6}(?:\s*[-–]\s*\d{1,6})?(?:\s*,\s*\d{1,6}(?:\s*[-–]\s*\d{1,6})?)*)\]"
)


_CITATION_RANGE_SEP = re.compile(r"\s*[-–]\s*")


# A marker with the space before it ("... days [1]."), so stripping leaves "... days.".
_MARKER_WITH_SPACE_RE = re.compile(r"[ \t]*" + _CITATION_RE.pattern)


def strip_citation_markers(text: str) -> str:
    """``text`` without its ``[n]`` citation markers (lists, ranges and runs too).

    For earlier answers sent back as conversation history: their numbers named the
    sources of the turn they answered, not this one's.
    """

    return _MARKER_WITH_SPACE_RE.sub("", text).strip()


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
    # The same for the multilingual profile's languages (es, de, fr, pt, it), written
    # without accents: both sides are compared accent-stripped (_fold), so "información"
    # and "informacion" both match.
    "suficiente informacion",
    "informacion suficiente",
    "no contienen",
    "no contiene",
    "no puedo responder",
    "nicht genugend informationen",
    "nicht genug informationen",
    "keine ausreichenden informationen",
    "enthalten nicht",
    "enthalt nicht",
    "nicht beantworten",
    "pas suffisamment d'informations",
    "pas assez d'informations",
    "informations insuffisantes",
    "ne contiennent pas",
    "ne contient pas",
    "ne peux pas repondre",
    "informacoes suficientes",
    "informacao suficiente",
    "nao contem",
    "nao posso responder",
    "informazioni sufficienti",
    "non contengono",
    "non contiene",
    "non posso rispondere",
)


def _fold(text: str) -> str:
    """Lower-cased and accent-stripped ("Información" -> "informacion")."""

    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


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
    # The cross-encoder's score for the passage (normalized to [0, 1]).
    rerank_score: float | None = None


def citation_from_dict(data: dict[str, Any]) -> SourceCitation:
    """A citation back from its ``dataclasses.asdict`` form (a ``done`` frame's sources)."""

    fields = {key: value for key, value in data.items() if key != "spans"}
    spans = tuple(CitationSpan(**span) for span in data.get("spans") or ())
    return SourceCitation(**fields, spans=spans)


def source_citations(
    chunks: Sequence[RerankedChunk], *, scored: bool = True
) -> list[SourceCitation]:
    """Return source metadata for chunks included in the prompt context. ``scored``
    False (a summary reads passages in order, never reranked) leaves out the score."""

    return [
        SourceCitation(
            source_number=index,
            filename=chunk.filename,
            page=chunk.page,
            section=chunk.section,
            chunk_id=chunk.chunk_id,
            text=_truncate_excerpt(chunk.text),
            page_end=chunk.page_end,
            rerank_score=chunk.rerank_score if scored else None,
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
# The same for the other languages answers come in (es, de, fr, pt, it): their
# articles, prepositions, conjunctions, pronouns and auxiliaries. Without them a
# Spanish sentence and an unrelated source sentence shared "el", "de", "que" — enough
# for a backfilled citation. Less the ones that are also everyday English words ("son",
# "plus", "par", "no"…), so English answers match as before.
_STOPWORDS_OTHER = frozenset(
    word
    for words in (
        # Spanish
        (
            "el la los las un una unos unas lo le les de del al en con por para sin sobre "
            "entre hasta desde y o u pero que qué como cómo es son está están ser fue ha "
            "han hay se su sus este esta estos estas ese esa más muy también cuando donde "
            "cada todo todos toda todas puede pueden debe"
        ).split(),
        # German
        (
            "der das den dem des ein eine einen einem einer eines und oder aber von zu zum "
            "zur mit auf für im ins bei aus nach über unter ist sind war wird werden wurde "
            "hat haben kann können muss müssen sie es er ihr ihre sich nicht kein keine "
            "auch noch dass wenn wie jede jeder jedes alle"
        ).split(),
        # French
        (
            "le les une des du au aux et ou mais dans pour avec sans sur sous par entre est "
            "sont été être avoir ont il elle ils elles se ce cette ces qui que dont ne pas "
            "plus aussi chaque tout tous toute toutes peut doit leur leurs sa ses son"
        ).split(),
        # Portuguese
        (
            "o os um uma uns umas do da dos das no na nos nas ao aos em com por pela pelo "
            "para sem sobre entre e ou mas que é são foi ser estar tem têm se seu sua seus "
            "suas este esta esse essa mais muito também cada todo todos pode deve"
        ).split(),
        # Italian
        (
            "il lo gli i uno di da dei degli delle della dello nel nella nei sul sulla con "
            "per tra fra ed ma che è sono essere ha hanno si suo sua suoi sue questo questa "
            "quello quella più anche ogni tutto tutti può deve non"
        ).split(),
    )
    for word in words
) - {"son", "come", "also", "plus", "hay", "sin", "par", "todo", "war", "no", "on", "in"}
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
        if word not in _STOPWORDS
        and word not in _STOPWORDS_OTHER
        and (len(word) > 1 or word.isdigit())
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


class _SpanIndex:
    """A source text's sentences, with the content words of each sentence (or pair of
    sentences) worked out once: every claim of an answer is matched against every
    source, and recomputing them per claim cost ~90 ms of CPU per backfill."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.sentences = sentence_spans(text)
        self._words: dict[tuple[int, int], set[str]] = {}

    def words(self, start: int, end: int) -> set[str]:
        found = self._words.get((start, end))
        if found is None:
            found = self._words[(start, end)] = _content_words(self.text[start:end])
        return found


# A Markdown heading flattened into a chunk's text ("## Receipts Submit receipts…"),
# and a setext underline left after its title ("Receipts ===== Submit…").
_HEADING_MARKER_RE = re.compile(r"#{1,6}\s+")
_SETEXT_UNDERLINE_RE = re.compile(r"\s*[=-]{3,}\s+")


def _past_heading(text: str, start: int, end: int, section: str) -> int:
    """Where a quote should start: past a heading's ``#`` marker, and past the heading's
    title when it is the chunk's section label (a heading met mid-chunk keeps its
    title: nothing says where that title ends). The quote is still ``text[start:end]``,
    so only ``start`` moves, and never past the claim itself."""

    moved = start
    marker = _HEADING_MARKER_RE.match(text, moved)
    if marker is not None:
        moved = marker.end()
    label = section.strip()
    if label and text.startswith(label, moved):
        after = moved + len(label)
        underline = _SETEXT_UNDERLINE_RE.match(text, after)
        if underline is not None:
            after = underline.end()
        elif marker is not None and after < end and text[after].isspace():
            after += 1
            while after < end and text[after].isspace():
                after += 1
        else:
            after = moved
        moved = after
    # A span that was nothing but the heading keeps it rather than becoming empty.
    return moved if moved < end and text[moved:end].strip() else start


@functools.lru_cache(maxsize=64)
def _span_index(text: str) -> _SpanIndex:
    return _SpanIndex(text)


def _best_span(
    claim: set[str], index: _SpanIndex
) -> tuple[tuple[float, int, int], int, int] | None:
    sentences = index.sentences
    # A pair only when each of its sentences carries part of the claim by itself:
    # otherwise one shared word ("staff") pulled an unrelated sentence into the quote.
    relevant = [
        len(claim & index.words(start, end)) >= SPAN_MIN_SHARED_WORDS for start, end in sentences
    ]
    candidates = [*sentences]
    candidates += [
        (sentences[i][0], sentences[i + 1][1])
        for i in range(len(sentences) - 1)
        # Never across a heading: the sentence before it belongs to another section.
        if relevant[i]
        and relevant[i + 1]
        and not _HEADING_MARKER_RE.match(index.text, sentences[i + 1][0])
    ]
    best: tuple[tuple[float, int, int], int, int] | None = None
    for start, end in candidates:
        words = index.words(start, end)
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
        spans_index = _span_index(text)
        found: dict[tuple[int, int], tuple[float, int, int]] = {}
        for claim in claims[source.source_number]:
            if not claim:
                continue
            best = _best_span(claim, spans_index)
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
        section = chunks[index].section
        trimmed = [(_past_heading(text, start, end, section), end) for start, end in sorted(kept)]
        spans = tuple(CitationSpan(start, end, text[start:end]) for start, end in trimmed)
        result.append(replace(source, spans=spans) if spans else source)
    return result


# A search result marks at most this many of its sentences.
MAX_QUERY_SPANS = 2


def query_spans(query: str, text: str, section: str = "") -> tuple[CitationSpan, ...]:
    """The sentences of ``text`` that share the most content words with a search query,
    to highlight in its result: at least two shared words (one for a query of fewer
    than three), best first, at most ``MAX_QUERY_SPANS``, returned in text order.
    Unlike an answer's spans there is no coverage floor: a query is a few keywords,
    not a claim to support."""

    claim = _content_words(query)
    if not claim:
        return ()
    needed = 1 if len(claim) < 3 else SPAN_MIN_SHARED_WORDS
    index = _span_index(text)
    scored = []
    for start, end in index.sentences:
        shared = len(claim & index.words(start, end))
        if shared >= needed:
            scored.append((shared, -(end - start), start, end))
    chosen = sorted(scored, reverse=True)[:MAX_QUERY_SPANS]
    spans = []
    for _, _, start, end in sorted(chosen, key=lambda item: item[2]):
        start = _past_heading(text, start, end, section)
        spans.append(CitationSpan(start, end, text[start:end]))
    return tuple(spans)


def needs_citation_retry(answer: str, source_count: int) -> bool:
    """True when an answer should be retried for missing citations.

    Only when sources were available, the answer contains no ``[n]`` marker, and the
    answer did not already declare the context insufficient (a valid uncited answer).
    """

    if source_count <= 0 or _CITATION_RE.search(answer):
        return False
    return not states_insufficiency(answer)


# How well an answer is grounded, sent with every answer. A string, not an enum, so a
# new value can be added without breaking a client: an unknown one reads as "answered".
ANSWER_STATUSES = ("answered", "insufficient_context", "no_context", "uncited")


def answer_status(answer: str, cited_count: int, source_count: int) -> str:
    """``no_context`` when nothing cleared the rerank bar, ``insufficient_context`` when
    the model said the sources don't cover it — citing none of them, or citing them
    only for that (``declines_to_answer``) — ``uncited`` when it answered without citing
    any source (even after the citation retry, or citing only numbers that aren't
    sources), else ``answered``.

    A hint word alone doesn't decide it, as for the citation retry: "the documents
    don't cover X, but Y [1]" is a partial answer. An uncited answer used to read as
    ``answered``, so nothing told the reader it had no sources.
    """

    if source_count <= 0:
        return "no_context"
    if cited_count == 0:
        if states_insufficiency(answer) or only_declines(answer):
            return "insufficient_context"
        return "uncited"
    return "insufficient_context" if declines_to_answer(answer) else "answered"


# How a model says the sources don't answer it, beyond _INSUFFICIENCY_HINTS (which
# also decide whether an uncited answer is retried, so they stay narrow).
_DECLINE_HINTS = (
    *_INSUFFICIENCY_HINTS,
    "not specified",
    "does not specify",
    "do not specify",
    "not mentioned",
    "does not mention",
    "do not mention",
    "does not say",
    "do not say",
    "does not state",
    "do not state",
    "does not provide",
    "do not provide",
    "does not address",
    "do not address",
    "does not discuss",
    "do not discuss",
    "does not include",
    "do not include",
    "unrelated",
    "not related",
    "no information",
    "none of these",
    "none of the",
)
# A sentence about what the sources are, not what they say: "[1] and [2] discuss
# inference engineering, while [3] covers an expense policy", "The context focuses on
# travel". After an opening refusal it explains the refusal; it answers nothing.
_ABOUT_THE_SOURCES_RE = re.compile(
    r"^\s*(?:\[\d[^\]]*\][\s,]*(?:(?:and|,)\s*\[\d[^\]]*\][\s,]*)*|(?:the |these |other )?"
    r"(?:provided |given |retrieved |available )?"
    r"(?:context|documents?|sources?|passages?|excerpts?|texts?|sections?)\b)"
    r"[^.]*?\b(?:discuss(?:es)?|cover(?:s)?|focus(?:es)?|describe(?:s)?|is about|are about|"
    r"pertain(?:s)?|relate(?:s)?|deal(?:s)? with|concern(?:s)?|talk(?:s)? about)\b"
)
# "contains no sick leave information", "gives no further details", "no mention of".
_NO_INFORMATION_RE = re.compile(r"\bno\b(?:\s+[\w-]+){0,4}?\s+(?:information|details|mention)\b")
# Where a sentence turns ("…, but …", "However, …"): what follows is what it ends up
# saying — something the sources do give, or (again) that they don't.
_CONTRAST_RE = re.compile(
    r"(?:,|;)\s*(?:but|however|although|though|except)\b|^\s*(?:but|however|although|though|"
    r"that said|still|nevertheless|nonetheless|instead)\b"
)
_LETTER_RE = re.compile(r"[^\W\d_]")
# What an opening refusal is about: the sources. "None of the contractors are eligible
# [1]" and "If not specified otherwise, notice is 30 days [1]" use the same words as a
# refusal and are answers.
_SOURCES_SUBJECT_RE = re.compile(
    r"\b(?:documents?|context|sources?|passages?|excerpts?|texts?|information|provided|"
    r"policy|policies|handbook|manual|guide|file|pdf)\b"
)
# A figure ("180 euros", "3.5"), not a digit inside a name ("B2B", "GPT4").
_FIGURE_RE = re.compile(r"(?<![^\W\d_])\d")


def _finds_nothing(clause: str) -> bool:
    return any(hint in clause for hint in _DECLINE_HINTS) or bool(_NO_INFORMATION_RE.search(clause))


def _declines(sentence: str, *, about_the_sources: bool = False) -> bool:
    """The sentence ends by saying the sources don't have it: "The documents don't give
    the rate" or "The context covers travel, but does not mention dividends" — not "…,
    but days carry over until March". ``about_the_sources``: a sentence that only says
    what the sources are about counts too (one with a figure in it gives something)."""

    folded = _normalize_negation(_fold(sentence))
    text = strip_citation_markers(folded)
    if about_the_sources and _ABOUT_THE_SOURCES_RE.search(folded) and not _FIGURE_RE.search(text):
        return True
    return _finds_nothing(_CONTRAST_RE.split(text)[-1])


def declines_to_answer(answer: str) -> bool:
    """True for an answer that cites sources only to say they don't answer it.

    "The provided documents do not contain enough information to determine how many
    sick days contractors get [1][2]." read as ``answered``: the badge said nothing, and
    the refusal went back as history, where it primed the next question to be refused
    too. A refusal opens with it; an answer that gives something after it — a cited
    sentence that doesn't decline too, or a "but"/"however" — is a partial answer.
    """

    sentences = [answer[start:end] for start, end in sentence_spans(answer)]
    # A trailing run of bare markers ("… ticker. [1] [3] [4]") says nothing either way.
    sentences = [s for s in sentences if _LETTER_RE.search(strip_citation_markers(s))]
    if not sentences or not opens_with_a_refusal(sentences[0]):
        return False
    return all(
        _declines(sentence, about_the_sources=True)
        for sentence in sentences[1:]
        if cited_numbers(sentence, _DECLINE_CITATION_CHECK_MAX)
        or _CONTRAST_RE.search(_fold(sentence.strip()))
    )


# An aside: "The document gives sixteen weeks of leave — it does not specify days".
_ASIDE_RE = re.compile(r"\s[-–—]\s|\s*[–—]\s*|;")


def opens_with_a_refusal(sentence: str) -> bool:
    """``sentence`` says the sources don't hold the answer: it declines, and about the
    documents ("The provided documents do not contain enough information …", "The
    policy does not specify …") — not after first giving something ("The document
    states parents get sixteen weeks of leave — it does not specify the days [1]",
    "Mileage is thirty cents per kilometre, but the policy does not give the currency"
    are answers with a caveat)."""

    if not (
        _declines(sentence) and _SOURCES_SUBJECT_RE.search(_fold(strip_citation_markers(sentence)))
    ):
        return False
    # Where the sentence first turns: an aside, or a contrast ("…, but the document does
    # not specify the currency"). What it says before that decides it.
    text = _normalize_negation(_fold(sentence))
    first = _CONTRAST_RE.split(_ASIDE_RE.split(text, maxsplit=1)[0], maxsplit=1)[0]
    if first == text or not _LETTER_RE.search(strip_citation_markers(first)):
        return True
    return _declines(first, about_the_sources=True)


def only_declines(answer: str) -> bool:
    """Every sentence of ``answer`` says the sources don't have it ("The documents do not
    mention fuel cards."), so an uncited answer is a refusal — unlike "The documents do
    not mention fuel cards. Mileage is 30 cents.", which answers part of it."""

    sentences = [answer[start:end] for start, end in sentence_spans(answer)]
    sentences = [s for s in sentences if _LETTER_RE.search(strip_citation_markers(s))]
    return (
        bool(sentences)
        and opens_with_a_refusal(sentences[0])
        and all(_declines(s, about_the_sources=True) for s in sentences[1:])
    )


# Only whether a sentence cites anything matters, so ranges are clamped small.
_DECLINE_CITATION_CHECK_MAX = 1000


def states_insufficiency(answer: str) -> bool:
    lowered = _normalize_negation(_fold(answer))
    return any(hint in lowered for hint in _INSUFFICIENCY_HINTS)


# Below this much of the first answer, a retry has too little to rewrite: skipped.
_MIN_RETRY_ANSWER_TOKENS = 64


def retry_messages(
    prompt_messages: Sequence[ChatMessage], first_answer: str, settings: AppSettings
) -> list[ChatMessage] | None:
    """The retry's continuation of the grounded prompt, fitted to the context window.

    The prompt was fitted with room for one answer; the retry adds that answer and a
    reminder on top, which could run past the window, and a local model then silently
    drops the *front* of the prompt — the grounding rules. Over budget, the history
    goes first (``[system, *history, user]``: the sources are in the last message, so
    their numbering is untouched), then the first answer is clipped; when even that
    leaves too little of it, None (no retry). An unknown window changes nothing.
    """

    def build(head: Sequence[ChatMessage], answer: str) -> list[ChatMessage]:
        return [
            *head,
            {"role": "assistant", "content": answer},
            {"role": "user", "content": CITATION_RETRY_REMINDER},
        ]

    messages = build(prompt_messages, first_answer)
    budget = prompt_token_budget(settings)
    if budget is None or estimate_prompt_tokens(messages) <= budget:
        return messages
    head = (
        [*prompt_messages[:1], *prompt_messages[-1:]]
        if len(prompt_messages) > 2
        else list(prompt_messages)
    )
    messages = build(head, first_answer)
    if estimate_prompt_tokens(messages) <= budget:
        return messages
    spare = budget - estimate_prompt_tokens(build(head, ""))
    if spare < _MIN_RETRY_ANSWER_TOKENS:
        return None
    # clip_text may add one character ("…"); clipped by weight, so a CJK answer fits too.
    clipped = clip_text(
        clip_to_weight(first_answer, spare * CHARS_PER_TOKEN_ESTIMATE - 1),
        spare * CHARS_PER_TOKEN_ESTIMATE - 1,
    )
    if clipped != first_answer and not clipped.endswith("…"):
        clipped = f"{clipped}…"
    return build(head, clipped)


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

    messages = retry_messages(prompt_messages, first_answer, settings)
    if messages is None:
        logger.info(
            "Skipped the citation retry: the continuation would not fit the context window."
        )
        return None
    citation_retries_total.inc()
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
    # The citations were added by backfill_citations, not written by the model.
    backfilled: bool = False


def _citation_steps(settings: AppSettings) -> tuple[bool, bool]:
    """Whether an uncited answer is backfilled, and whether it gets the LLM retry
    (CITATION_RETRY_MODE; CITATION_RETRY_ENABLED=false turns both off)."""

    mode = settings.citation_retry_mode
    if not settings.citation_retry_enabled or mode == "off":
        return False, False
    if mode == "llm":
        return False, True
    if mode == "backfill":
        return True, False
    local = settings.llm_provider.lower().strip() in ("ollama", "openai_compatible")
    return local, True


def will_retry_citations(
    answer: str,
    all_sources: Sequence[SourceCitation],
    settings: AppSettings,
    *,
    weak_matches: bool = False,
) -> bool:
    """Whether ``finalize_citations`` will try to add citations to this answer.

    Split out so the streaming path can announce the retry *before* it blocks, without
    restating the condition — two copies would drift the first time either changed.

    ``weak_matches``: the prompt told the model to decline *without citing* when the
    nearest passages don't answer, so a refusal there is what was asked for — the retry
    asked for citations on it and cost a second generation for nothing.
    """

    if weak_matches:
        sentences = [answer[start:end] for start, end in sentence_spans(answer)]
        if sentences and opens_with_a_refusal(sentences[0]):
            return False
    return (
        not cited_sources(answer, all_sources)
        and any(_citation_steps(settings))
        and needs_citation_retry(answer, len(all_sources))
    )


def will_call_llm_for_citations(
    answer: str,
    all_sources: Sequence[SourceCitation],
    settings: AppSettings,
    *,
    weak_matches: bool = False,
    context_chunks: Sequence[RerankedChunk] | None = None,
) -> bool:
    """Whether ``finalize_citations`` will make the LLM retry call: it will add
    citations, and backfilling (when it applies) won't find them. The stream announces
    the call before it blocks; a backfill takes no noticeable time."""

    # Judged on the text finalize_citations judges: a marker that only appears in an
    # echoed source header line is stripped there, and would hide the retry here.
    answer = strip_echoed_source_headers(answer)
    if not will_retry_citations(answer, all_sources, settings, weak_matches=weak_matches):
        return False
    backfill, llm_retry = _citation_steps(settings)
    if backfill and context_chunks:
        filled = backfill_citations(answer, context_chunks)
        if filled is not None and cited_sources(filled, all_sources):
            return False
    return llm_retry


# Backfilling cites on the match alone, so its bar is above the span highlighter's:
# most of a sentence's content words, and at least three of them, in one source.
BACKFILL_MIN_SHARED_WORDS = 3
BACKFILL_MIN_COVERAGE = 0.5
_SENTENCE_END_RE = re.compile(r"[.!?。！？]+[\"'”’)\]]*\s*$")


def backfill_citations(answer: str, chunks: Sequence[RerankedChunk]) -> str | None:
    """``answer`` with ``[n]`` added to each sentence a source plainly states, or None
    when no sentence matched one (or the answer already cites).

    The LLM citation retry is a second full generation — about a minute on a local
    model — to add markers to an answer that is usually right. Here each sentence is
    matched against the prompt's sources the way citation spans are (content words,
    spelled-out numbers as digits, one or two adjacent source sentences) and gets the
    best one's number. A sentence saying the sources lack something is left alone.
    """

    if not chunks or _CITATION_RE.search(answer):
        return None
    # Asked twice per answer (will_call_llm_for_citations, then finalize_citations).
    return _backfill(answer, tuple(chunk.expanded_text or chunk.text for chunk in chunks))


@functools.lru_cache(maxsize=16)
def _backfill(answer: str, texts: tuple[str, ...]) -> str | None:
    sources = [_span_index(text) for text in texts]
    pieces: list[str] = []
    last = 0
    matched = False
    for start, end in sentence_spans(answer):
        sentence = answer[start:end]
        claim = _content_words(sentence)
        if len(claim) < BACKFILL_MIN_SHARED_WORDS or _declines(sentence):
            continue
        best: tuple[tuple[float, int, int], int] | None = None
        for number, source_index in enumerate(sources, start=1):
            found = _best_span(claim, source_index)
            if found is None:
                continue
            coverage = found[0][0]
            if (
                coverage < BACKFILL_MIN_COVERAGE
                or coverage * len(claim) < BACKFILL_MIN_SHARED_WORDS
            ):
                continue
            # At least one shared word that says something: a number or a word of four
            # or more letters, not only short words a stopword list missed.
            shared = claim & source_index.words(found[1], found[2])
            if not any(word.isdigit() or len(word) >= 4 for word in shared):
                continue
            if best is None or found[0][0] > best[0][0]:
                best = (found[0], number)
        if best is None:
            continue
        ending = _SENTENCE_END_RE.search(sentence)
        cut = start + (ending.start() if ending else len(sentence.rstrip()))
        pieces.append(answer[last:cut])
        pieces.append(f" [{best[1]}]")
        last = cut
        matched = True
    if not matched:
        return None
    pieces.append(answer[last:])
    return "".join(pieces)


# A line that copies a source's header from the prompt ("[1] filename=I-2.pdf; page=1-2;
# section=…"). The prompt says to cite inline and not to end with a references list;
# qwen2.5 appended one to answer after answer anyway, and it showed as a stray line.
_ECHOED_HEADER_RE = re.compile(r"^[ \t>*-]*(?:\[\d{1,6}\][ \t]*)+filename=[^\n]*(?:\n|$)", re.M)


def strip_echoed_source_headers(answer: str) -> str:
    """``answer`` without lines that repeat a source header; never emptied by it."""

    stripped = _ECHOED_HEADER_RE.sub("", answer).strip()
    return stripped or answer


def finalize_citations(
    prompt_messages: Sequence[ChatMessage],
    answer: str,
    all_sources: Sequence[SourceCitation],
    generator: ChatGenerator,
    settings: AppSettings,
    *,
    weak_matches: bool = False,
    context_chunks: Sequence[RerankedChunk] | None = None,
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

    ``context_chunks`` (the prompt's sources, in order) let an uncited answer be
    backfilled without an LLM call (CITATION_RETRY_MODE); ``weak_matches`` skips adding
    citations to a refusal the weak-match prompt asked to be uncited.
    """

    answer = strip_echoed_source_headers(answer)
    cited = cited_sources(answer, all_sources)
    final_messages = list(prompt_messages)
    retry_used = False
    retry_ms = 0.0
    backfill, llm_retry = _citation_steps(settings)
    if (
        backfill
        and context_chunks
        and will_retry_citations(answer, all_sources, settings, weak_matches=weak_matches)
    ):
        filled = backfill_citations(answer, context_chunks)
        if filled is not None and cited_sources(filled, all_sources):
            citation_backfills_total.inc()
            return CitationOutcome(
                answer=filled,
                cited=cited_sources(filled, all_sources),
                retry_used=False,
                prompt_messages=final_messages,
                backfilled=True,
            )
    if llm_retry and will_retry_citations(answer, all_sources, settings, weak_matches=weak_matches):
        # Timed around the call itself, not around the `if` — a retry that fails or
        # still comes back uncited cost just as much wall-clock as one that worked, and
        # attributing it to generate_ms is what made this stage invisible before.
        retry_start = time.monotonic()
        first_finish = last_finish_reason()
        retried = retry_uncited_answer(prompt_messages, answer, generator, settings)
        retry_ms = (time.monotonic() - retry_start) * 1000
        if retried is not None:
            answer, final_messages = retried
            answer = strip_echoed_source_headers(answer)
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
