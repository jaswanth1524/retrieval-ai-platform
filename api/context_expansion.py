"""Small-to-big context: widen each kept chunk with its ``chunk_ordinal`` neighbours for
the prompt (citations still point at the original chunk), and the shared pool that runs
per-query fan-out (query-expansion variants, neighbour scrolls) in parallel.

Split out of ``api.pipeline``, which re-exports these names.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from api.repository import VectorRepository
from api.request_id import submit_in_context
from api.reranking import RerankedChunk
from api.settings import AppSettings

# Query-expansion variants are embedded and searched side by side: four sequential
# embed+search round trips were the whole cost of turning expansion on. Neighbour
# expansion fetches its per-file scrolls here too. Shared by every question: 8 covers
# MAX_CONCURRENT_QUESTIONS' default 4 with two queries each (a follow-up and its rewrite).
_VARIANT_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="docrag-variants")
# The stored-tag lookup of a tag-scoped question runs on its own: when the listing is
# cold it scans the whole collection, and on the shared pool four such lookups held
# every worker — other questions' searches waited behind them (measured: a fan-out
# waited 1 s behind four 1 s lookups).
_TAG_LOOKUP_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="docrag-tags")


def _map_concurrently[T, R](function: Callable[[T], R], items: Sequence[T]) -> list[R]:
    """``[function(item) for item in items]``, in parallel when there is more than one."""

    if len(items) <= 1:
        return [function(item) for item in items]
    futures = [
        submit_in_context(_VARIANT_EXECUTOR, functools.partial(function, item)) for item in items
    ]
    return [future.result() for future in futures]


# Below this, a suffix/prefix match between adjacent chunks is more likely a
# coincidence (a shared "the"/"and") than the sentence overlap chunking wrote.
_MIN_OVERLAP_CHARS = 20
# Upper bound on characters per dense-model token, used only to cap how much text an
# overlap can span. Real text averages ~4; this is generous so a genuine overlap of
# long words is never missed.
_MAX_CHARS_PER_TOKEN = 10


def _strip_overlap(previous: str, following: str, overlap_tokens: int) -> str:
    """Drop the prefix of ``following`` that repeats the tail of ``previous``.

    Adjacent chunks share up to ``overlap_tokens`` of whole trailing sentences (chunks
    are space-joined sentences), so a word-boundary suffix/prefix match finds the text
    written twice. Returns "" when ``following`` lies entirely inside that tail, and
    ``following`` unchanged when no overlap is found (or chunking used none).

    The *shortest* match wins and the match is capped near the configured overlap:
    in repetitive text (identical CSV rows, log lines) several lengths match, and
    over-trimming would silently drop real content while under-trimming only resends
    a line the model already saw.
    """

    if overlap_tokens <= 0:
        return following
    limit = min(len(previous), len(following), overlap_tokens * _MAX_CHARS_PER_TOKEN)
    for end in range(_MIN_OVERLAP_CHARS, limit + 1):
        if end < len(following) and not following[end].isspace():
            continue
        start = len(previous) - end
        if start > 0 and not previous[start - 1].isspace():
            continue
        if previous.endswith(following[:end]):
            return following[end:].lstrip()
    return following


def _clip_head(text: str, max_chars: int) -> str:
    """The start of ``text``, at most ``max_chars`` long, cut at a word boundary.
    ``max_chars`` 0 keeps everything."""

    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = text[:max_chars]
    mid_word = not head[-1].isspace() and not text[max_chars].isspace()
    if mid_word and any(character.isspace() for character in head):
        head = head.rsplit(None, 1)[0]
    return head.rstrip()


def _clip_tail(text: str, max_chars: int) -> str:
    """The end of ``text``, at most ``max_chars`` long, cut at a word boundary."""

    if max_chars <= 0 or len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    mid_word = not tail[0].isspace() and not text[-max_chars - 1].isspace()
    if mid_word and any(character.isspace() for character in tail):
        tail = tail.split(None, 1)[1]
    return tail.lstrip()


def expand_with_neighbors(
    chunks: Sequence[RerankedChunk],
    repository: VectorRepository,
    settings: AppSettings,
) -> list[RerankedChunk]:
    """Expand each reranked chunk's generation context with its immediate neighbors.

    Small-to-big retrieval: rerank on tight chunks, but let the model generate from a
    wider window around each one. Chunks with no recorded ``chunk_ordinal`` (indexed
    before this field existed) are returned unchanged rather than erroring.

    Neighbours are fetched **once per distinct filename**, not once per chunk: the
    surviving chunks are usually a handful drawn from one or two documents, so a
    per-chunk round trip meant up to ``rerank_top_k`` sequential Qdrant scrolls on every
    question for data one query per file could return.

    Every prompt token costs prefill time, so no text is sent twice: a neighbour that is
    itself one of ``chunks`` (it gets its own block) or already appeared in a
    higher-ranked chunk's block is skipped, and the overlap chunking repeats at each
    boundary between consecutive ordinals is trimmed.

    ``context_neighbor_max_chars`` (0 = no cap) keeps only the text nearest the chunk:
    the last that many characters before it and the first that many after it. The
    sentences right at the boundary are what resolve a chunk that starts or ends
    mid-thought; the far side of a neighbour mostly adds prefill.
    """

    radius = int(settings.context_neighbor_radius)
    if radius <= 0:
        return list(chunks)
    overlap_tokens = int(settings.chunk_overlap_tokens)
    max_chars = int(settings.context_neighbor_max_chars)

    def wanted_ordinals(chunk: RerankedChunk) -> list[int]:
        assert chunk.chunk_ordinal is not None
        return [
            chunk.chunk_ordinal + delta
            for delta in range(-radius, radius + 1)
            if delta != 0 and chunk.chunk_ordinal + delta >= 1
        ]

    emitted: set[tuple[str, int]] = {
        (chunk.filename, chunk.chunk_ordinal) for chunk in chunks if chunk.chunk_ordinal is not None
    }

    # Union every chunk's neighbour ordinals per filename, then one fetch per filename.
    ordinals_by_file: dict[str, set[int]] = {}
    for chunk in chunks:
        if chunk.chunk_ordinal is None:
            continue
        ordinals_by_file.setdefault(chunk.filename, set()).update(
            ordinal
            for ordinal in wanted_ordinals(chunk)
            if (chunk.filename, ordinal) not in emitted
        )

    # (filename, ordinal) -> text, so each chunk can pick its own neighbours back out.
    # The per-file scrolls are independent round trips, so they run concurrently.
    wanted = [
        (filename, sorted(ordinals)) for filename, ordinals in ordinals_by_file.items() if ordinals
    ]
    fetched = _map_concurrently(
        lambda item: repository.fetch_neighbors(settings, item[0], item[1]), wanted
    )
    neighbor_text: dict[tuple[str, int], str] = {}
    for (filename, _), points in zip(wanted, fetched, strict=True):
        for point in points:
            payload = point.payload or {}
            ordinal = payload.get("chunk_ordinal")
            text = payload.get("text")
            if isinstance(ordinal, int) and not isinstance(ordinal, bool) and text:
                neighbor_text[(filename, ordinal)] = str(text)

    expanded: list[RerankedChunk] = []
    for chunk in chunks:
        if chunk.chunk_ordinal is None:
            expanded.append(chunk)
            continue
        neighbors: list[tuple[int, str]] = []
        for ordinal in wanted_ordinals(chunk):
            key = (chunk.filename, ordinal)
            if key in emitted or key not in neighbor_text:
                continue
            emitted.add(key)
            neighbors.append((ordinal, neighbor_text[key]))
        if not neighbors:
            expanded.append(chunk)
            continue

        parts = sorted([*neighbors, (chunk.chunk_ordinal, chunk.text)])
        before: list[str] = []
        center: list[str] = []
        after: list[str] = []
        previous: tuple[int, str] | None = None
        for ordinal, text in parts:
            if previous is not None and ordinal == previous[0] + 1:
                trimmed = _strip_overlap(previous[1], text, overlap_tokens)
            else:
                trimmed = text
            if trimmed:
                if ordinal < chunk.chunk_ordinal:
                    before.append(trimmed)
                elif ordinal == chunk.chunk_ordinal:
                    center.append(trimmed)
                else:
                    after.append(trimmed)
            previous = (ordinal, text)
        merged = [
            _clip_tail("\n\n".join(before), max_chars),
            *center,
            _clip_head("\n\n".join(after), max_chars),
        ]
        expanded.append(replace(chunk, expanded_text="\n\n".join(part for part in merged if part)))
    return expanded
