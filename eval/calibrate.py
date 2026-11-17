"""``harness calibrate``: the rerank bar that best separates the questions the corpus
answers from the ones it doesn't, measured offline over labeled datasets.

Each question's best rerank score is read from retrieval (no LLM; a follow-up is
searched as typed). The answerable ones (labeled with relevant files, chunks or
passages) should clear the bar and the ``expect_no_context`` ones shouldn't; the report
gives how many each bar gets wrong, for the current RERANK_MIN_SCORE and the best one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BarReport:
    bar: float
    # Answerable questions below the bar (would be refused), unanswerable ones at or
    # above it (would be answered from weak passages).
    refused_answerable: int
    answered_unanswerable: int

    @property
    def wrong(self) -> int:
        return self.refused_answerable + self.answered_unanswerable


def judge(bar: float, answerable: Sequence[float], unanswerable: Sequence[float]) -> BarReport:
    return BarReport(
        bar,
        sum(score < bar for score in answerable),
        sum(score >= bar for score in unanswerable),
    )


def best_bar(answerable: Sequence[float], unanswerable: Sequence[float]) -> BarReport | None:
    """The bar with the fewest questions on the wrong side, placed midway between the
    neighbouring scores; of equally good ones, the one with the widest gap around it.
    None without questions of both kinds."""

    if not answerable or not unanswerable:
        return None
    scores = sorted({*answerable, *unanswerable})
    candidates = [scores[0] - 1e-6]
    candidates += [(low + high) / 2 for low, high in zip(scores, scores[1:], strict=False)]
    candidates.append(scores[-1] + 1e-6)

    def gap(bar: float) -> float:
        below = max((s for s in scores if s < bar), default=bar)
        above = min((s for s in scores if s >= bar), default=bar)
        return above - below

    return min(
        (judge(bar, answerable, unanswerable) for bar in candidates),
        key=lambda report: (report.wrong, -gap(report.bar)),
    )
