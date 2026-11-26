"""How much a metric's change between two runs of the same questions is noise.

A baseline gate compares a run's rate (``fact_hit_rate`` 0.917 vs 0.833) with an allowed
drop, but on 12 questions one answer moves a rate by 0.083: a "regression" is often one
question going the other way. A paired bootstrap resamples the questions (the same ones
from both runs) and gives the change's 95% interval: one that holds 0 is within noise.
Reported next to the gate's verdict; it doesn't change it.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

BOOTSTRAP_DRAWS = 2000


def paired_bootstrap(
    current: Sequence[float],
    baseline: Sequence[float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = 0,
) -> tuple[float, float, float]:
    """The mean change (current − baseline) over paired values, and its 95% interval."""

    if len(current) != len(baseline) or not current:
        raise ValueError("paired_bootstrap needs two equal, non-empty lists")
    differences = [a - b for a, b in zip(current, baseline, strict=True)]
    count = len(differences)
    rng = random.Random(seed)
    means = sorted(
        sum(differences[rng.randrange(count)] for _ in range(count)) / count for _ in range(draws)
    )
    low = means[int(0.025 * (draws - 1))]
    high = means[int(0.975 * (draws - 1))]
    return sum(differences) / count, low, high


# Per-question outcomes the answer rates are means of.
_ANSWER_OUTCOMES: dict[str, Callable[[Mapping[str, Any]], float | None]] = {
    "fact_hit_rate": lambda row: None if row.get("fact_hit") is None else float(row["fact_hit"]),
    "answered_rate": lambda row: float(row.get("answer_status") == "answered"),
    "error_rate": lambda row: float(row.get("error") is not None),
}


def answer_noise(
    current_rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]]
) -> dict[str, tuple[float, float, float]]:
    """For each answer rate both runs measured, the change and its 95% interval, over the
    questions both asked (matched by question text). Empty when they share none."""

    baseline_by_question = {str(row.get("question")): row for row in baseline_rows}
    pairs = [
        (row, baseline_by_question[str(row.get("question"))])
        for row in current_rows
        if str(row.get("question")) in baseline_by_question
    ]
    found: dict[str, tuple[float, float, float]] = {}
    for name, outcome in _ANSWER_OUTCOMES.items():
        values = [(outcome(a), outcome(b)) for a, b in pairs]
        kept = [(a, b) for a, b in values if a is not None and b is not None]
        if len(kept) >= 2:
            found[name] = paired_bootstrap([a for a, _ in kept], [b for _, b in kept])
    return found


__all__ = ["BOOTSTRAP_DRAWS", "answer_noise", "paired_bootstrap"]
