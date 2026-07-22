"""Post-rerank diversity/dedup selection.

The overlap between adjacent chunks (and neighbor expansion, which pulls in a chunk's
ordinal-neighbors) means the reranker can keep several near-identical chunks, wasting
context slots on redundant text. ``select_diverse`` runs *after* scoring as a pure
selection filter — it never changes rerank scores or their order (which stay pinned per
CLAUDE.md) — dropping a survivor that duplicates a higher-ranked kept chunk and
refilling the freed slot from lower-ranked survivors so context still gets up to
``rerank_top_k`` distinct chunks.
"""

from __future__ import annotations

import re
from collections.abc import Set as AbstractSet

from api.reranking import RerankedChunk, RerankOutcome
from api.settings import AppSettings

_WORD_RE = re.compile(r"\w+")


def word_shingles(text: str, n: int = 3) -> set[str]:
    """Return the set of n-word shingles (lowercased) for near-duplicate comparison.

    Text with fewer than ``n`` words yields a single whole-text shingle so short
    chunks still compare meaningfully.
    """

    words = _WORD_RE.findall(text.lower())
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: AbstractSet[str], b: AbstractSet[str]) -> float:
    """Jaccard similarity of two shingle sets; 0.0 when both are empty."""

    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def select_diverse(
    outcome: RerankOutcome, settings: AppSettings
) -> tuple[RerankOutcome, set[str]]:
    """Filter ``outcome.kept`` for diversity, returning the new outcome + dropped ids.

    Iterates the min-score-cleared, rerank-ordered candidates (``outcome.scored``),
    greedily keeping ones that are neither an ordinal-neighbor of an already-kept
    same-document chunk nor textually near-duplicate of any kept chunk, up to
    ``rerank_top_k``. Returns the dropped point ids for tracing (``near_duplicate``).
    """

    if not settings.context_diversity_enabled:
        return outcome, set()

    min_score = float(settings.rerank_min_score)
    radius = max(int(settings.context_neighbor_radius), 1)
    threshold = float(settings.context_diversity_max_similarity)
    top_k = int(settings.rerank_top_k)

    eligible = [chunk for chunk in outcome.scored if chunk.rerank_score >= min_score]
    kept: list[RerankedChunk] = []
    kept_shingles: list[set[str]] = []
    dropped: set[str] = set()

    for candidate in eligible:
        if len(kept) >= top_k:
            break
        near_ordinal = any(
            keeper.filename == candidate.filename
            and keeper.chunk_ordinal is not None
            and candidate.chunk_ordinal is not None
            and abs(keeper.chunk_ordinal - candidate.chunk_ordinal) <= radius
            for keeper in kept
        )
        shingles = word_shingles(candidate.text)
        near_text = any(jaccard(shingles, existing) >= threshold for existing in kept_shingles)
        if near_ordinal or near_text:
            dropped.add(candidate.point_id)
            continue
        kept.append(candidate)
        kept_shingles.append(shingles)

    return RerankOutcome(scored=outcome.scored, kept=kept), dropped
