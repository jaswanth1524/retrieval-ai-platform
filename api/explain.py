"""Where an expected passage was lost on its way through a search (``POST /search``
with ``explain`` and ``expect_chunk_id`` / ``expect_filename``).

Each stage of the search is checked in order and the first one that left the passage
out is named: neither search leg returned it, fusion ranked it past what the reranker
scores (RERANK_CANDIDATES), the cross-encoder scored it under the bar, the diversity
filter dropped it as a near-duplicate of a kept one, or the top-K cut left it out.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from api.pipeline import RetrievalPhase
from api.reranking import RerankedChunk
from api.retrieval import RetrievedChunk
from api.settings import AppSettings

# What happened to the passage (extensible: a client shows an unknown one as is).
RETURNED = "returned"
NOT_RETRIEVED = "not_retrieved"
NOT_SCORED = "not_scored"
BELOW_MIN_SCORE = "below_min_score"
NEAR_DUPLICATE = "near_duplicate"
PAST_TOP_K = "past_top_k"


@dataclass(frozen=True)
class Explanation:
    target: str
    outcome: str
    detail: str
    chunk_id: str | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_rank: int | None = None
    rerank_rank: int | None = None
    rerank_score: float | None = None
    returned_rank: int | None = None


def _first(
    chunks: Sequence[RetrievedChunk | RerankedChunk],
    chunk_id: str | None,
    filename: str | None,
) -> tuple[int, RetrievedChunk | RerankedChunk] | None:
    for index, chunk in enumerate(chunks, start=1):
        if (chunk_id is not None and chunk.chunk_id == chunk_id) or (
            chunk_id is None and chunk.filename == filename
        ):
            return index, chunk
    return None


def explain(
    phase: RetrievalPhase,
    settings: AppSettings,
    *,
    chunk_id: str | None = None,
    filename: str | None = None,
) -> Explanation:
    """What became of ``chunk_id`` (or the best passage of ``filename``) in ``phase``."""

    target = chunk_id if chunk_id is not None else f"any passage of {filename}"
    fused = _first(phase.fused_candidates, chunk_id, filename)
    if fused is None:
        return Explanation(
            target,
            NOT_RETRIEVED,
            "Neither the dense nor the BM25 search returned it among their top "
            f"{settings.dense_retrieval_limit}/{settings.sparse_retrieval_limit}: the "
            "question shares too little with it (in words or meaning). Rephrase it with "
            "the passage's own terms.",
            chunk_id=chunk_id,
        )
    fused_rank, candidate = fused
    ranks = phase.leg_ranks.get(candidate.point_id, [None, None])
    found = Explanation(
        target,
        "",
        "",
        chunk_id=candidate.chunk_id,
        dense_rank=ranks[0],
        sparse_rank=ranks[1],
        fused_rank=fused_rank,
    )
    scored = _first(phase.scored_chunks, candidate.chunk_id, None)
    if scored is None:
        return replace(
            found,
            outcome=NOT_SCORED,
            detail=f"Fusion ranked it {fused_rank}, past the {settings.rerank_candidates} "
            "candidates the cross-encoder scores (RERANK_CANDIDATES).",
        )
    rerank_rank, scored_chunk = scored
    assert isinstance(scored_chunk, RerankedChunk)
    score = scored_chunk.rerank_score
    found = replace(found, rerank_rank=rerank_rank, rerank_score=score)
    returned = _first(phase.context_chunks, candidate.chunk_id, None)
    if returned is not None:
        return replace(
            found,
            outcome=RETURNED,
            detail=f"Returned at rank {returned[0]}.",
            returned_rank=returned[0],
        )
    floor = float(settings.rerank_min_score)
    if score < floor and not phase.below_min_score:
        return replace(
            found,
            outcome=BELOW_MIN_SCORE,
            detail=f"The cross-encoder scored it {score:.3f}, under the bar of {floor} "
            "(RERANK_MIN_SCORE).",
        )
    if candidate.point_id in phase.diversity_dropped_ids:
        return replace(
            found,
            outcome=NEAR_DUPLICATE,
            detail="Dropped as a near-duplicate of a passage ranked above it "
            "(CONTEXT_DIVERSITY_MAX_SIMILARITY).",
        )
    return replace(
        found,
        outcome=PAST_TOP_K,
        detail=f"Scored {score:.3f} (rank {rerank_rank}), past the {settings.rerank_top_k} "
        "results kept (RERANK_TOP_K).",
    )
