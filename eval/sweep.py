"""Offline sweep of the passage-selection settings (``harness sweep``).

Retrieval and the cross-encoder run once per question, with the settings as they are;
every combination of the swept settings is then replayed over the scored candidates
(``pipeline.select_passages``) — no model runs again, so a grid of dozens of
combinations costs about as much as one retrieval run.

Only settings applied *after* the rerank can be swept: the relevance bar, the relative
floor, the weak-match fallback, top-K, the recency boost and the diversity filter. What
decides which candidates are scored — the RRF k, the retrieval limits, the candidate
count, the models — is fixed by the run (the RRF k is a non-negotiable 60 anyway).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from api.pipeline import select_passages
from api.reranking import RerankedChunk
from api.settings import AppSettings
from eval.metrics import aggregate, score_ranking

# The settings a sweep may vary, and how a value is read from the command line.
SWEEPABLE: dict[str, Callable[[str], Any]] = {
    "rerank_min_score": float,
    "rerank_relative_floor": float,
    "rerank_fallback_count": int,
    "rerank_top_k": int,
    "recency_boost_weight": float,
    "context_diversity_max_similarity": float,
}


class SweepGridError(ValueError):
    """A grid that names a setting the sweep can't vary, or a value it can't read."""


def parse_grid(specs: Sequence[str]) -> dict[str, list[Any]]:
    """``["rerank_min_score=0.05,0.1", "rerank_top_k=4,6"]`` -> name -> values."""

    grid: dict[str, list[Any]] = {}
    for spec in specs:
        name, _, values = spec.partition("=")
        name = name.strip().lower()
        if name not in SWEEPABLE:
            raise SweepGridError(f"Can't sweep '{name}'; one of: {', '.join(sorted(SWEEPABLE))}.")
        try:
            parsed = [
                SWEEPABLE[name](value.strip()) for value in values.split(",") if value.strip()
            ]
        except ValueError as exc:
            raise SweepGridError(f"Bad value for {name}: {exc}") from exc
        if not parsed:
            raise SweepGridError(f"No values given for {name}.")
        grid[name] = parsed
    return grid


def combinations(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    names = sorted(grid)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(grid[n] for n in names))
    ]


def replay(
    scored: Sequence[RerankedChunk], settings: AppSettings, combo: Mapping[str, Any]
) -> tuple[list[str], list[str], bool]:
    """(filenames, chunk ids, weak-match) the reranked arm would return under ``combo``."""

    kept, below = select_passages(scored, settings.model_copy(update=dict(combo)))
    return [c.filename for c in kept], [c.chunk_id for c in kept], below


def score_combo(
    examples: Sequence[Any],
    scored_by_question: Sequence[Sequence[RerankedChunk]],
    passage_ids: Mapping[int, frozenset[str]],
    settings: AppSettings,
    combo: Mapping[str, Any],
    ks: Sequence[int],
) -> dict[str, dict[str, float]]:
    """The same aggregates ``run_retrieval_eval`` reports, for one combination."""

    filename_rows: list[dict[str, float]] = []
    chunk_rows: list[dict[str, float]] = []
    no_context_rows: list[dict[str, float]] = []
    for index, (example, scored) in enumerate(zip(examples, scored_by_question, strict=True)):
        filenames, chunk_ids, below = replay(scored, settings, combo)
        if example.expect_no_context:
            nothing = not chunk_ids or below
            no_context_rows.append({"no_context_rate": 1.0 if nothing else 0.0})
        if example.relevant_filenames:
            filename_rows.append(score_ranking(filenames, example.relevant_filenames, ks))
        relevant = example.relevant_chunk_ids | passage_ids.get(index, frozenset())
        if relevant:
            chunk_rows.append(score_ranking(chunk_ids, relevant, ks))
    levels = {"filename": aggregate(filename_rows), "chunk": aggregate(chunk_rows)}
    if no_context_rows:
        levels["unanswerable"] = aggregate(no_context_rows)
    return levels


def metric(levels: Mapping[str, Mapping[str, float]], name: str) -> float | None:
    """``levels["chunk"]["mrr"]`` for ``"chunk.mrr"``; None when the run has no such."""

    level, _, key = name.partition(".")
    value = levels.get(level, {}).get(key)
    return float(value) if value is not None else None


def env_lines(combo: Mapping[str, Any]) -> list[str]:
    return [f"{name.upper()}={value}" for name, value in sorted(combo.items())]


def format_table(rows: Sequence[Mapping[str, Any]], rank_by: str) -> str:
    """One line per combination, best first by ``rank_by``; the current settings marked."""

    names = sorted(rows[0]["settings"]) if rows else []
    lines = ["  ".join([*names, rank_by, "no_context_rate"])]
    for row in rows:
        value = metric(row["aggregate"], rank_by)
        refusal = metric(row["aggregate"], "unanswerable.no_context_rate")
        lines.append(
            "  ".join(
                [
                    *(str(row["settings"][name]) for name in names),
                    "-" if value is None else f"{value:.3f}",
                    "-" if refusal is None else f"{refusal:.3f}",
                ]
            )
            + ("  (current)" if row.get("current") else "")
        )
    return "\n".join(lines)
