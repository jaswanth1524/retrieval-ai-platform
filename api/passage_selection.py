"""Which reranked passages an answer uses: diversity, the weak-match fallback, the
relative cut and the recency order. Moved out of ``api/pipeline.py``, which re-exports
these names.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from api.diversity import select_diverse
from api.reranking import (
    RerankedChunk,
    RerankOutcome,
    relative_cut,
)
from api.settings import AppSettings


@dataclass(frozen=True)
class _PassageSelection:
    outcome: RerankOutcome
    diversity_dropped: set[str]
    below_min_score: bool
    relative_cut_ids: set[str]


def _select_passages(
    outcome: RerankOutcome,
    settings: AppSettings,
    *,
    for_generation: bool,
    picked_only: bool = False,
) -> _PassageSelection:
    """The reranked passages kept for the prompt (or a search): near-duplicates dropped,
    the closest ones kept when none clears the floor, and the relative cut applied.

    With ``picked_only`` (a question about passages the user picked, ``chunk_ids``)
    every picked passage is kept, best first: the user chose them, and the context isn't
    neighbour-expanded, so a picked passage next to another was lost for good when the
    diversity filter dropped it as a neighbour, as was one under the score floor.
    """

    if picked_only:
        return _PassageSelection(
            RerankOutcome(scored=outcome.scored, kept=list(outcome.scored)), set(), False, set()
        )
    # A chunk next to a kept one is dropped on the promise that neighbour expansion
    # puts its text back — true only when expansion brings the whole neighbour.
    # With CONTEXT_NEIGHBOR_MAX_CHARS capping it, a well-scored passage was lost.
    diverse_outcome, diversity_dropped = select_diverse(
        outcome,
        settings,
        drop_neighbours=for_generation and int(settings.context_neighbor_max_chars) == 0,
    )
    # Nothing cleared rerank_min_score: the closest passages are used anyway,
    # flagged. A search (no answer) shows up to rerank_top_k of them; the cross-
    # encoder scores a bare keyword low against a passage that has it ("laptop":
    # 0.11 for the IT equipment policy, which it ranks first). An answer gets
    # rerank_fallback_count of them, and the prompt tells the model they match only
    # weakly, so it answers from them only when they hold the answer: the floor is
    # on one model's score scale, and a fixed one refused questions the documents
    # answer ("what are foundational models": 0.006 with MiniLM, 0.154 with jina).
    below_min_score = False
    fallback_count = min(
        int(settings.rerank_top_k),
        int(settings.rerank_fallback_count) if for_generation else int(settings.rerank_top_k),
    )
    if fallback_count > 0 and not diverse_outcome.kept and outcome.scored:
        # rerank_top_k too: select_diverse fills up to it from the scored list, so
        # an answer got 8 below-the-bar passages instead of rerank_fallback_count.
        unfloored = settings.model_copy(
            update={"rerank_min_score": 0.0, "rerank_top_k": fallback_count}
        )
        diverse_outcome, diversity_dropped = select_diverse(
            RerankOutcome(scored=outcome.scored, kept=outcome.scored[:fallback_count]),
            unfloored,
            drop_neighbours=False,
        )
        below_min_score = bool(diverse_outcome.kept)
    relative_cut_ids: set[str] = set()
    if for_generation and not below_min_score:
        trimmed, relative_cut_ids = relative_cut(
            diverse_outcome.kept,
            outcome.scored,
            float(settings.rerank_relative_floor),
        )
        diverse_outcome = RerankOutcome(
            scored=diverse_outcome.scored,
            kept=recency_ordered(trimmed, float(settings.recency_boost_weight)),
        )
    return _PassageSelection(diverse_outcome, diversity_dropped, below_min_score, relative_cut_ids)


def select_passages(
    scored: Sequence[RerankedChunk], settings: AppSettings, question: str = ""
) -> tuple[list[RerankedChunk], bool]:
    """The passages an answer would be given from ``scored`` (one rerank pass's scored
    candidates, best first) under ``settings``: the floor and top-K cut as the reranker
    applies them, then ``_select_passages``. With whether they are the weak-match
    fallback. The seam the eval harness's sweep replays settings through without
    retrieving or reranking again (``question`` decides the recency boost, as it does
    for an answer)."""

    settings = without_recency_for_the_past(settings, question)
    ordered = list(scored)
    floor = float(settings.rerank_min_score)
    kept = [chunk for chunk in ordered if chunk.rerank_score >= floor][: int(settings.rerank_top_k)]
    selection = _select_passages(
        RerankOutcome(scored=ordered, kept=kept), settings, for_generation=True
    )
    return list(selection.outcome.kept), selection.below_min_score


# A question about an earlier state of things: a year, or words naming an older version.
# The recency boost would put the newest document first — the one it isn't asking about.
# Measured (eval/corpus_recency, jina int8, 48-token chunks, RECENCY_BOOST_WEIGHT 1.0):
# questions about the past, file hit@1 0.33 boosted, 1.00 with the boost left out for
# them; the current-state set kept its gain (0.50 unboosted, 1.00 either way). Boosting
# only questions that say "current" or "latest" lost that gain (0.50): people rarely do.
_ABOUT_THE_PAST_RE = re.compile(
    r"\b(?:19|20)\d\d\b|\b(?:old|older|previous|previously|earlier|former|formerly|"
    r"original|originally|used to|prior|past|outdated|archived|superseded|historic|"
    r"historical)\b",
    re.IGNORECASE,
)


def asks_about_the_past(text: str) -> bool:
    """Whether a question asks about an earlier version or year (no recency boost)."""

    return bool(_ABOUT_THE_PAST_RE.search(text))


def without_recency_for_the_past(settings: AppSettings, *questions: str | None) -> AppSettings:
    """``settings`` with RECENCY_BOOST_WEIGHT 0 when any of ``questions`` (a rewrite and
    the raw follow-up) asks about the past; as they are otherwise."""

    if float(settings.recency_boost_weight) <= 0:
        return settings
    if any(question and asks_about_the_past(question) for question in questions):
        return settings.model_copy(update={"recency_boost_weight": 0.0})
    return settings


# How far back the recency boost looks: a document this much older than the newest kept
# one gets none of it. A fixed span, not the kept passages' own: documents uploaded
# seconds apart (a folder indexed in one go) must not be told apart by it.
RECENCY_HORIZON_SECONDS = 365 * 24 * 3600.0


def recency_ordered(chunks: Sequence[RerankedChunk], weight: float) -> list[RerankedChunk]:
    """``chunks`` ordered by rerank score plus ``weight`` x how recent each one's document
    is — 1 for the newest kept, falling to 0 a year (``RECENCY_HORIZON_SECONDS``) before
    it — RECENCY_BOOST_WEIGHT. The same passages, only reordered; unchanged when the
    weight is 0 or none is dated."""

    stamps = [chunk.uploaded_at for chunk in chunks if chunk.uploaded_at is not None]
    if weight <= 0 or len(stamps) < 2:
        return list(chunks)
    newest = max(stamps)

    def boosted(chunk: RerankedChunk) -> float:
        if chunk.uploaded_at is None:
            return chunk.rerank_score
        age = newest - chunk.uploaded_at
        return chunk.rerank_score + weight * max(0.0, 1.0 - age / RECENCY_HORIZON_SECONDS)

    return sorted(chunks, key=boosted, reverse=True)
