"""Tuning the relevance bar on a deployment's own questions.

RERANK_MIN_SCORE stays 0.15 because the measured margin between answerable and
unanswerable questions was too thin to change the default (pass 8). An operator can
tune it on their own corpus: ``score_report`` sets the best cross-encoder score of each
question answered against those that weren't, and suggests the bar that separates them
best — only from enough of each, since a bar fitted to a handful of questions is noise.
``trace_diff`` shows why two traces of a question went differently (Regenerate, or a
settings change): what moved in the ranking, and which settings differed.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from api.tracing import QueryTrace, TraceCandidate, TraceSummary

# Histogram bins over [0, 1]: the cross-encoder's score is a probability.
SCORE_BINS = 20
# Fewer traces than this on either side and no bar is suggested.
MIN_TRACES_PER_SIDE = 20
UNANSWERED_STATUSES = frozenset({"insufficient_context", "no_context"})


@dataclass(frozen=True)
class ScoreReport:
    current_min_score: float
    bin_edges: list[float]
    answered: list[int]
    unanswered: list[int]
    answered_count: int
    unanswered_count: int
    suggested_min_score: float | None
    # Why there is no suggestion (too few traces), or what the suggestion rests on.
    note: str
    # Traces whose top score lies between the current bar and the suggested one: the
    # questions the suggestion would decide differently.
    would_change: int


def _histogram(scores: Sequence[float]) -> list[int]:
    counts = [0] * SCORE_BINS
    for score in scores:
        counts[min(max(int(score * SCORE_BINS), 0), SCORE_BINS - 1)] += 1
    return counts


def _best_bar(answered: Sequence[float], unanswered: Sequence[float]) -> tuple[float, int]:
    """The bar with the fewest traces on the wrong side of it (answered below,
    unanswered at or above), placed midway between two neighbouring observed scores so
    it doesn't sit on a question's own score. Ties go to the lower bar, which refuses
    fewer questions."""

    scores = sorted({*answered, *unanswered})
    candidates = [0.0, *((low + high) / 2 for low, high in itertools.pairwise(scores)), 1.0]
    best: tuple[float, int] | None = None
    for bar in candidates:
        wrong = sum(score < bar for score in answered) + sum(score >= bar for score in unanswered)
        if best is None or wrong < best[1]:
            best = (bar, wrong)
    assert best is not None
    return best


def score_report(summaries: Sequence[TraceSummary], current_min_score: float) -> ScoreReport:
    """The top rerank score of each traced question, answered against not answered.

    Traces with no score (nothing was reranked) or no answer status (errors, traces
    from before statuses were kept) are left out. "uncited" answers count as answered:
    the passages were there.
    """

    answered: list[float] = []
    unanswered: list[float] = []
    for summary in summaries:
        if summary.top_rerank_score is None or summary.answer_status is None:
            continue
        side = unanswered if summary.answer_status in UNANSWERED_STATUSES else answered
        side.append(float(summary.top_rerank_score))
    edges = [round(index / SCORE_BINS, 4) for index in range(SCORE_BINS + 1)]
    suggested: float | None = None
    would_change = 0
    if len(answered) < MIN_TRACES_PER_SIDE or len(unanswered) < MIN_TRACES_PER_SIDE:
        note = (
            f"Needs at least {MIN_TRACES_PER_SIDE} answered and {MIN_TRACES_PER_SIDE} "
            f"unanswered questions with scores; there are {len(answered)} and "
            f"{len(unanswered)}."
        )
    else:
        bar, wrong = _best_bar(answered, unanswered)
        suggested = round(bar, 4)
        low, high = sorted((current_min_score, suggested))
        would_change = sum(low <= score < high for score in (*answered, *unanswered))
        note = (
            f"{wrong} of {len(answered) + len(unanswered)} traced questions are on the "
            "wrong side of the suggested bar. Answers are judged by the model, not by a "
            "person: check a few before changing RERANK_MIN_SCORE."
        )
    return ScoreReport(
        current_min_score=current_min_score,
        bin_edges=edges,
        answered=_histogram(answered),
        unanswered=_histogram(unanswered),
        answered_count=len(answered),
        unanswered_count=len(unanswered),
        suggested_min_score=suggested,
        note=note,
        would_change=would_change,
    )


@dataclass(frozen=True)
class CandidateChange:
    point_id: str
    filename: str
    page: int
    chunk_id: str
    # 1-based rank by rerank score among each trace's scored candidates; None when the
    # candidate wasn't in that trace (or wasn't scored).
    rank_a: int | None
    rank_b: int | None
    score_a: float | None
    score_b: float | None
    selected_a: bool
    selected_b: bool
    drop_reason_a: str | None
    drop_reason_b: str | None


@dataclass(frozen=True)
class SettingChange:
    name: str
    a: object
    b: object


@dataclass(frozen=True)
class TraceDiff:
    question_a: str
    question_b: str
    same_answer: bool
    settings: list[SettingChange]
    # Only candidates whose rank, selection or drop reason differ.
    candidates: list[CandidateChange]


def _ranked(candidates: Sequence[TraceCandidate]) -> dict[str, tuple[int, TraceCandidate]]:
    scored = sorted(
        (c for c in candidates if c.rerank_score is not None),
        key=lambda c: c.rerank_score or 0.0,
        reverse=True,
    )
    ranks = {c.point_id: (rank, c) for rank, c in enumerate(scored, start=1)}
    for candidate in candidates:
        ranks.setdefault(candidate.point_id, (0, candidate))
    return ranks


def trace_diff(a: QueryTrace, b: QueryTrace) -> TraceDiff:
    """What differs between two traces: their settings, and each candidate whose rank,
    score, selection for the prompt or drop reason moved."""

    settings: list[SettingChange] = []
    if a.config is not None and b.config is not None:
        for field in dataclasses.fields(a.config):
            value_a = getattr(a.config, field.name)
            value_b = getattr(b.config, field.name)
            if value_a != value_b:
                settings.append(SettingChange(field.name, value_a, value_b))
    ranked_a = _ranked(a.candidates)
    ranked_b = _ranked(b.candidates)
    changes: list[CandidateChange] = []
    for point_id in dict.fromkeys([*ranked_a, *ranked_b]):
        rank_a, cand_a = ranked_a.get(point_id, (None, None))
        rank_b, cand_b = ranked_b.get(point_id, (None, None))
        some = cand_a or cand_b
        assert some is not None
        change = CandidateChange(
            point_id=point_id,
            filename=some.filename,
            page=some.page,
            chunk_id=some.chunk_id,
            rank_a=rank_a or None,
            rank_b=rank_b or None,
            score_a=cand_a.rerank_score if cand_a else None,
            score_b=cand_b.rerank_score if cand_b else None,
            selected_a=bool(cand_a and cand_a.selected_for_context),
            selected_b=bool(cand_b and cand_b.selected_for_context),
            drop_reason_a=cand_a.drop_reason if cand_a else None,
            drop_reason_b=cand_b.drop_reason if cand_b else None,
        )
        if (
            change.rank_a != change.rank_b
            or change.selected_a != change.selected_b
            or change.drop_reason_a != change.drop_reason_b
        ):
            changes.append(change)
    return TraceDiff(
        question_a=a.question,
        question_b=b.question,
        same_answer=(a.answer or "").strip() == (b.answer or "").strip(),
        settings=settings,
        candidates=changes,
    )
