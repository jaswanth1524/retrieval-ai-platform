from __future__ import annotations

from typing import Any

from api.diversity import jaccard, select_diverse, word_shingles
from api.reranking import RerankedChunk, RerankOutcome
from api.settings import AppSettings


def make_settings(**overrides: Any) -> AppSettings:
    defaults: dict[str, Any] = {
        "rerank_top_k": 3,
        "rerank_min_score": 0.15,
        "context_neighbor_radius": 1,
        "context_diversity_max_similarity": 0.6,
    }
    defaults.update(overrides)
    return AppSettings(_env_file=None, **defaults)  # type: ignore[call-arg]


def chunk(
    point_id: str,
    *,
    filename: str = "a.md",
    ordinal: int | None = None,
    text: str = "unique text here",
    score: float = 0.9,
) -> RerankedChunk:
    return RerankedChunk(
        point_id=point_id,
        filename=filename,
        page=1,
        section="Body",
        chunk_id=point_id,
        text=text,
        retrieval_score=score,
        rerank_score=score,
        chunk_ordinal=ordinal,
    )


def outcome_of(*chunks: RerankedChunk) -> RerankOutcome:
    return RerankOutcome(scored=list(chunks), kept=list(chunks))


def test_word_shingles_and_jaccard() -> None:
    same = word_shingles("the quick brown fox")
    assert jaccard(same, same) == 1.0
    disjoint = jaccard(word_shingles("alpha beta gamma"), word_shingles("totally other words"))
    assert disjoint == 0.0


def test_disabled_diversity_is_identity() -> None:
    settings = make_settings(context_diversity_enabled=False)
    out = outcome_of(
        chunk("c1", ordinal=1),
        chunk("c2", ordinal=2, text="different words entirely"),
    )
    result, dropped = select_diverse(out, settings)
    assert result.kept == out.kept
    assert dropped == set()


def test_drops_ordinal_neighbor_and_refills() -> None:
    settings = make_settings(rerank_top_k=2)
    out = outcome_of(
        chunk("c1", ordinal=5, text="first distinct passage about alpha"),
        chunk("c2", ordinal=6, text="second passage overlapping alpha ideas"),  # neighbor of c1
        chunk("c3", ordinal=40, text="totally separate topic on zeta matters"),
    )
    result, dropped = select_diverse(out, settings)

    kept_ids = [c.point_id for c in result.kept]
    assert "c2" in dropped  # adjacent ordinal to kept c1
    assert kept_ids == ["c1", "c3"]  # freed slot refilled from the distant chunk


def test_drops_textual_near_duplicate_across_files() -> None:
    settings = make_settings(rerank_top_k=2, context_diversity_max_similarity=0.5)
    shared = "the deployment guide explains docker compose setup steps in detail"
    out = outcome_of(
        chunk("c1", filename="a.md", ordinal=1, text=shared),
        chunk("c2", filename="b.md", ordinal=9, text=shared),  # different file, same text
        chunk("c3", filename="c.md", ordinal=3, text="unrelated content about billing exports"),
    )
    result, dropped = select_diverse(out, settings)

    assert "c2" in dropped
    assert [c.point_id for c in result.kept] == ["c1", "c3"]


def test_never_exceeds_top_k() -> None:
    settings = make_settings(rerank_top_k=2)
    out = outcome_of(
        chunk("c1", ordinal=1, text="alpha content one"),
        chunk("c2", ordinal=10, text="beta content two"),
        chunk("c3", ordinal=20, text="gamma content three"),
    )
    result, _ = select_diverse(out, settings)
    assert len(result.kept) == 2


def test_zero_radius_keeps_ordinal_neighbors_with_distinct_text() -> None:
    """context_neighbor_radius=0 disables neighbor expansion entirely (see the
    setting's docstring) — with no expansion, ordinal-adjacent chunks aren't
    near-identical, so the ordinal check must not fire at radius 0. Shingle-based
    textual dedup still applies independently."""

    settings = make_settings(context_neighbor_radius=0, rerank_top_k=2)
    out = outcome_of(
        chunk("c1", ordinal=5, text="first distinct passage about alpha"),
        chunk("c2", ordinal=6, text="second passage covers a totally different topic"),
    )
    result, dropped = select_diverse(out, settings)

    assert dropped == set()
    assert [c.point_id for c in result.kept] == ["c1", "c2"]


def test_zero_radius_still_drops_textual_near_duplicates() -> None:
    settings = make_settings(context_neighbor_radius=0, rerank_top_k=2)
    shared = "the deployment guide explains docker compose setup steps in detail"
    # Deliberately far-apart ordinals: with adjacent ones the pre-change code's
    # clamped radius would drop c2 via the ordinal check, so the test would pass
    # without ever exercising the shingle path it exists to cover.
    out = outcome_of(
        chunk("c1", ordinal=5, text=shared),
        chunk("c2", ordinal=40, text=shared),
    )
    result, dropped = select_diverse(out, settings)

    assert "c2" in dropped
    assert [c.point_id for c in result.kept] == ["c1"]


def test_ignores_below_min_score_candidates() -> None:
    settings = make_settings(rerank_min_score=0.5)
    out = RerankOutcome(
        scored=[
            chunk("c1", ordinal=1, score=0.9, text="kept passage"),
            chunk("c2", ordinal=2, score=0.1, text="too low to matter"),
        ],
        kept=[chunk("c1", ordinal=1, score=0.9, text="kept passage")],
    )
    result, dropped = select_diverse(out, settings)
    assert [c.point_id for c in result.kept] == ["c1"]
    assert "c2" not in dropped  # never eligible, so not a diversity drop
