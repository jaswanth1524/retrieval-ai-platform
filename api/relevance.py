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

import bisect
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
    # Traces not counted: scored by another cross-encoder, or picked passages (which
    # skip the bar), or summaries.
    left_out: int = 0


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
    # Sorted once, each bar's count by bisection: the archive can hold thousands of
    # traces, and a count over both lists per bar was quadratic.
    ordered_answered = sorted(answered)
    ordered_unanswered = sorted(unanswered)
    best: tuple[float, int] | None = None
    for bar in candidates:
        wrong = bisect.bisect_left(ordered_answered, bar) + (
            len(ordered_unanswered) - bisect.bisect_left(ordered_unanswered, bar)
        )
        if best is None or wrong < best[1]:
            best = (bar, wrong)
    assert best is not None
    return best


def score_report(
    summaries: Sequence[TraceSummary],
    current_min_score: float,
    reranker_model: str | None = None,
) -> ScoreReport:
    """The top rerank score of each traced question, answered against not answered.

    Traces with no score (nothing was reranked) or no answer status (errors, traces
    from before statuses were kept) are left out. "uncited" answers count as answered:
    the passages were there. Counted apart (``left_out``): traces scored by another
    cross-encoder than ``reranker_model`` (scores don't compare across models),
    questions about picked passages (they skip the bar, so their low scores would pull
    it down) and summaries.
    """

    answered: list[float] = []
    unanswered: list[float] = []
    left_out = 0
    for summary in summaries:
        if summary.top_rerank_score is None or summary.answer_status is None:
            continue
        other_model = (
            reranker_model is not None
            and summary.reranker_model is not None
            and summary.reranker_model != reranker_model
        )
        if other_model or not summary.gated or summary.kind != "question":
            left_out += 1
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
        left_out=left_out,
    )


@dataclass(frozen=True)
class CandidateChange:
    point_id: str
    filename: str
    page: int
    # Told apart from another passage on the same page by its section.
    section: str
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
    # Only candidates whose rank, score (by SCORE_CHANGE_MIN or more), selection or
    # drop reason differ.
    candidates: list[CandidateChange]


# TraceConfig fields older traces lack (None there), compared only when both have them.
_RECORDED_LATER = frozenset(
    {
        "rerank_candidates",
        "rerank_relative_floor",
        "recency_boost_weight",
        "query_expansion_enabled",
        "reranker_onnx_file",
        "context_neighbor_max_chars",
        "context_neighbor_radius",
        "rerank_fallback_count",
        "reranker_model",
        "llm_model",
    }
)
# A score that moved less than this is the same (int8 scores shift with batching).
SCORE_CHANGE_MIN = 0.01


def _score_moved(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) >= SCORE_CHANGE_MIN


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
            # One trace from before the setting was recorded: not a difference.
            unrecorded = (value_a is None) != (value_b is None) and field.name in _RECORDED_LATER
            if value_a != value_b and not unrecorded:
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
            section=some.section,
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
            or _score_moved(change.score_a, change.score_b)
        ):
            changes.append(change)
    return TraceDiff(
        question_a=a.question,
        question_b=b.question,
        same_answer=(a.answer or "").strip() == (b.answer or "").strip(),
        settings=settings,
        candidates=changes,
    )
