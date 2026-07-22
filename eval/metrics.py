"""Retrieval-quality metrics for DocRAG's eval harness.

Pure functions over ranked ID lists and relevant-ID sets — no models, no I/O — so
they run in CI as ordinary unit tests. The harness (``eval/harness.py``) produces
the ranked lists by driving the real pipeline; these functions score them.

A "ranked" list is the ordered identifiers a retrieval run returned (filenames or
chunk_ids, most-relevant first). "relevant" is the labeled ground-truth set for a
question. Metrics are computed independently at each requested cutoff ``k``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet


def reciprocal_rank(ranked: Sequence[str], relevant: AbstractSet[str]) -> float:
    """Reciprocal of the 1-based rank of the first relevant item, or 0.0 if none."""

    for index, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def hit_at_k(ranked: Sequence[str], relevant: AbstractSet[str], k: int) -> float:
    """1.0 if any relevant item appears in the top-k, else 0.0."""

    return 1.0 if any(item in relevant for item in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: AbstractSet[str], k: int) -> float:
    """Fraction of the relevant set found in the top-k. 0.0 when nothing is relevant."""

    if not relevant:
        return 0.0
    found = len({item for item in ranked[:k]} & set(relevant))
    return found / len(relevant)


def score_ranking(
    ranked: Sequence[str], relevant: AbstractSet[str], ks: Sequence[int]
) -> dict[str, float]:
    """Score one ranked list at every cutoff in ``ks`` plus MRR (cutoff-independent)."""

    scores: dict[str, float] = {"mrr": reciprocal_rank(ranked, relevant)}
    for k in ks:
        scores[f"hit@{k}"] = hit_at_k(ranked, relevant, k)
        scores[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
    return scores


def aggregate(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Mean of each metric across per-question score rows. Empty input -> empty dict.

    Assumes every row carries the same metric keys (they do — all come from
    ``score_ranking`` with the same ``ks``).
    """

    if not rows:
        return {}
    keys = list(rows[0].keys())
    count = len(rows)
    return {key: sum(row[key] for row in rows) / count for key in keys}


def compare_to_baseline(
    current: Mapping[str, float],
    baseline: Mapping[str, float],
    max_regression: float,
) -> list[str]:
    """Return human-readable regression messages for metrics that dropped too far.

    A metric regresses when ``current < baseline - max_regression``. Metrics present
    in the baseline but absent from ``current`` are skipped (shape changes are not
    regressions). Returns an empty list when nothing regressed.
    """

    regressions: list[str] = []
    for key, baseline_value in baseline.items():
        current_value = current.get(key)
        if current_value is None:
            continue
        if current_value < baseline_value - max_regression:
            regressions.append(
                f"{key}: {current_value:.4f} < baseline {baseline_value:.4f} "
                f"(allowed drop {max_regression:.4f})"
            )
    return regressions
