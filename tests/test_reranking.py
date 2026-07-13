from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import pytest

from api.reranking import (
    LocalCrossEncoderReranker,
    RerankingError,
    rerank_candidates,
    rerank_candidates_detailed,
)
from api.retrieval import RetrievalError, RetrievedChunk
from api.settings import AppSettings


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class FakeReranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.seen_query: str | None = None
        self.seen_documents: list[str] = []

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.seen_query = query
        self.seen_documents = list(documents)
        return self.scores


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.seen_query: str | None = None
        self.seen_documents: list[str] = []
        self.seen_batch_size: int | None = None

    def rerank(
        self,
        query: str,
        documents: Iterable[str],
        batch_size: int = 64,
        **kwargs: Any,
    ) -> Iterable[float]:
        self.seen_query = query
        self.seen_documents = list(documents)
        self.seen_batch_size = batch_size
        return self.scores


def make_settings(**overrides: Any) -> AppSettings:
    defaults: dict[str, Any] = {
        "fused_top_n": 3,
        "rerank_top_k": 2,
        "reranker_batch_size": 7,
    }
    defaults.update(overrides)
    return AppSettings(_env_file=None, **defaults)  # type: ignore[call-arg]


def make_candidate(chunk_id: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        point_id=f"point-{chunk_id}",
        filename="guide.md",
        page=1,
        section="Setup",
        chunk_id=chunk_id,
        text=text,
        score=score,
    )


def test_rerank_candidates_sorts_by_cross_encoder_score_and_truncates() -> None:
    settings = make_settings(rerank_top_k=2)
    candidates = [
        make_candidate("c1", "first", 0.9),
        make_candidate("c2", "second", 0.8),
        make_candidate("c3", "third", 0.7),
    ]
    reranker = FakeReranker([0.2, 0.95, 0.5])

    results = rerank_candidates("  setup docs  ", candidates, reranker, settings)

    assert reranker.seen_query == "setup docs"
    assert reranker.seen_documents == ["first", "second", "third"]
    assert [result.chunk_id for result in results] == ["c2", "c3"]
    assert results[0].retrieval_score == 0.8
    assert results[0].rerank_score == pytest.approx(_sigmoid(0.95))


def test_rerank_candidates_scores_only_rerank_candidates_setting() -> None:
    """``rerank_candidates`` (clamped to fused_top_n) trims how many of the fused
    candidates are actually sent through the cross-encoder — independent of
    fused_top_n itself, which stays the fusion spec's own top-N."""

    settings = make_settings(fused_top_n=5, rerank_top_k=5, rerank_candidates=2)
    candidates = [
        make_candidate("c1", "first", 0.9),
        make_candidate("c2", "second", 0.8),
        make_candidate("c3", "third", 0.7),
    ]
    reranker = FakeReranker([0.1, 0.2])

    results = rerank_candidates("query", candidates, reranker, settings)

    assert reranker.seen_documents == ["first", "second"]
    assert [result.chunk_id for result in results] == ["c2", "c1"]


def test_rerank_candidates_rerank_candidates_setting_never_exceeds_fused_top_n() -> None:
    settings = make_settings(fused_top_n=2, rerank_top_k=5, rerank_candidates=50)
    candidates = [
        make_candidate("c1", "first", 0.9),
        make_candidate("c2", "second", 0.8),
        make_candidate("c3", "third", 0.7),
    ]
    reranker = FakeReranker([0.1, 0.2])

    results = rerank_candidates("query", candidates, reranker, settings)

    assert reranker.seen_documents == ["first", "second"]
    assert [result.chunk_id for result in results] == ["c2", "c1"]


def test_rerank_candidates_scores_only_fused_top_n() -> None:
    settings = make_settings(fused_top_n=2, rerank_top_k=5)
    candidates = [
        make_candidate("c1", "first", 0.9),
        make_candidate("c2", "second", 0.8),
        make_candidate("c3", "third", 0.7),
    ]
    reranker = FakeReranker([0.1, 0.2])

    results = rerank_candidates("query", candidates, reranker, settings)

    assert reranker.seen_documents == ["first", "second"]
    assert [result.chunk_id for result in results] == ["c2", "c1"]


def test_rerank_candidates_uses_retrieval_score_as_tiebreaker() -> None:
    settings = make_settings(rerank_top_k=2)
    candidates = [
        make_candidate("low", "low retrieval", 0.1),
        make_candidate("high", "high retrieval", 0.9),
    ]
    reranker = FakeReranker([0.5, 0.5])

    results = rerank_candidates("query", candidates, reranker, settings)

    assert [result.chunk_id for result in results] == ["high", "low"]


def test_rerank_candidates_rejects_empty_query() -> None:
    with pytest.raises(RetrievalError, match="Query text"):
        rerank_candidates(" ", [], FakeReranker([]), make_settings())


def test_rerank_candidates_returns_empty_without_calling_reranker() -> None:
    reranker = FakeReranker([1.0])

    results = rerank_candidates("query", [], reranker, make_settings())

    assert results == []
    assert reranker.seen_query is None


def test_rerank_candidates_rejects_score_count_mismatch() -> None:
    candidates = [make_candidate("c1", "first", 0.9), make_candidate("c2", "second", 0.8)]

    with pytest.raises(RerankingError, match="score count mismatch"):
        rerank_candidates("query", candidates, FakeReranker([0.1]), make_settings())


def test_rerank_candidates_filters_below_min_score() -> None:
    settings = make_settings(rerank_top_k=5, rerank_min_score=0.30)
    candidates = [
        make_candidate("relevant", "on topic", 0.9),
        make_candidate("irrelevant", "off topic", 0.1),
    ]
    # sigmoid(2.0) ~= 0.88 (passes 0.30); sigmoid(-5.0) ~= 0.0067 (fails 0.30).
    reranker = FakeReranker([2.0, -5.0])

    results = rerank_candidates("query", candidates, reranker, settings)

    assert [result.chunk_id for result in results] == ["relevant"]
    assert len(results) < settings.rerank_top_k


def test_rerank_candidates_returns_empty_when_all_below_min_score() -> None:
    settings = make_settings(rerank_top_k=5, rerank_min_score=0.30)
    candidates = [
        make_candidate("c1", "first", 0.9),
        make_candidate("c2", "second", 0.8),
    ]
    reranker = FakeReranker([-6.0, -7.0])

    results = rerank_candidates("query", candidates, reranker, settings)

    assert results == []


def test_rerank_candidates_detailed_scored_includes_dropped_below_min_score() -> None:
    """`.kept` matches `rerank_candidates`'s return; `.scored` also carries the
    candidate that was filtered out, for trace/debug visibility into why."""

    settings = make_settings(rerank_top_k=5, rerank_min_score=0.30)
    candidates = [
        make_candidate("relevant", "on topic", 0.9),
        make_candidate("irrelevant", "off topic", 0.1),
    ]
    reranker = FakeReranker([2.0, -5.0])

    outcome = rerank_candidates_detailed("query", candidates, reranker, settings)

    assert [c.chunk_id for c in outcome.kept] == ["relevant"]
    assert {c.chunk_id for c in outcome.scored} == {"relevant", "irrelevant"}
    assert outcome.kept == rerank_candidates("query", candidates, reranker, settings)


def test_rerank_candidates_detailed_scored_excludes_beyond_rerank_candidates_cap() -> None:
    """A candidate outside the `rerank_candidates` cutoff was never sent to the
    cross-encoder at all, so it must not appear in `.scored` either."""

    settings = make_settings(fused_top_n=5, rerank_top_k=5, rerank_candidates=2)
    candidates = [
        make_candidate("c1", "first", 0.9),
        make_candidate("c2", "second", 0.8),
        make_candidate("c3", "third", 0.7),
    ]
    reranker = FakeReranker([0.1, 0.2])

    outcome = rerank_candidates_detailed("query", candidates, reranker, settings)

    assert {c.chunk_id for c in outcome.scored} == {"c1", "c2"}


def test_rerank_candidates_detailed_empty_candidates_returns_empty_outcome() -> None:
    outcome = rerank_candidates_detailed("query", [], FakeReranker([1.0]), make_settings())

    assert outcome.scored == []
    assert outcome.kept == []


def test_local_cross_encoder_reranker_delegates_to_model() -> None:
    settings = make_settings(reranker_batch_size=11)
    model = FakeCrossEncoder([0.7, 0.4])
    reranker = LocalCrossEncoderReranker(settings, model=model)

    scores = reranker.score("query", ["doc one", "doc two"])

    assert scores == [0.7, 0.4]
    assert model.seen_query == "query"
    assert model.seen_documents == ["doc one", "doc two"]
    assert model.seen_batch_size == 11


class RaisingCrossEncoder:
    def rerank(
        self, query: str, documents: Iterable[str], batch_size: int = 64, **kwargs: Any
    ) -> Iterable[float]:
        raise RuntimeError("model download failed")


def test_local_cross_encoder_reranker_wraps_model_failure_as_reranking_error() -> None:
    """The model is lazy_load=True, so a bad model name or failed download only
    surfaces on this first call, not at construction — this call must translate it
    into RerankingError (-> 502) rather than an unwrapped exception (-> 500)."""

    reranker = LocalCrossEncoderReranker(make_settings(), model=RaisingCrossEncoder())

    with pytest.raises(RerankingError, match="model download failed"):
        reranker.score("query", ["doc"])

