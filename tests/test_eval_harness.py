from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import models

from api.embeddings import EmbeddedText
from api.settings import AppSettings
from eval.harness import (
    HarnessDataError,
    build_parser,
    load_retrieval_dataset,
    rank_for_question,
    run_retrieval_eval,
)


def make_settings(**overrides: Any) -> AppSettings:
    defaults: dict[str, Any] = {
        "qdrant_url": ":memory:",
        "embedding_model_tag": "test-embedding:v1",
        "rerank_top_k": 3,
        "rerank_candidates": 10,
        "rerank_min_score": 0.0,
    }
    defaults.update(overrides)
    return AppSettings(_env_file=None, **defaults)  # type: ignore[call-arg]


class FakeEmbeddingProvider:
    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]:
        sparse = models.SparseVector(indices=[1], values=[1.0])
        return [EmbeddedText(dense=[1.0, 0.0, 0.0], sparse=sparse) for _ in texts]


class FakeRepository:
    """Returns fixed scored points; ranking order is the reranker's job."""

    def __init__(self, points: list[models.ScoredPoint]) -> None:
        self._points = points

    def ensure_ready(self, settings: AppSettings) -> None:  # noqa: D401 - protocol stub
        return None

    def hybrid_search(
        self,
        settings: AppSettings,
        query_embedding: EmbeddedText,
        filenames: Sequence[str] | None = None,
    ) -> list[models.ScoredPoint]:
        return self._points


class FakeReranker:
    """Scores documents by their order in a preset chunk_id priority list."""

    def __init__(self, priority: list[str]) -> None:
        self._priority = priority

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        # Higher score for earlier-priority text; documents are the chunk texts.
        return [
            float(len(self._priority) - self._priority.index(doc)) if doc in self._priority else 0.0
            for doc in documents
        ]


def _point(chunk_id: str, filename: str) -> models.ScoredPoint:
    return models.ScoredPoint(
        id=str(uuid5(NAMESPACE_URL, chunk_id)),
        version=0,
        score=1.0,
        payload={
            "filename": filename,
            "page": 1,
            "section": "Body",
            "chunk_id": chunk_id,
            "text": chunk_id,  # text == chunk_id so the fake reranker can key on it
            "chunk_ordinal": 1,
        },
    )


def test_rank_for_question_returns_reranked_order() -> None:
    points = [_point("c1", "a.md"), _point("c2", "b.md"), _point("c3", "a.md")]
    result = rank_for_question(
        "q",
        repository=FakeRepository(points),
        embedding_provider=FakeEmbeddingProvider(),
        reranker=FakeReranker(["c3", "c1", "c2"]),
        settings=make_settings(),
    )
    assert result.chunk_ids == ["c3", "c1", "c2"]
    assert result.filenames == ["a.md", "a.md", "b.md"]


def test_run_retrieval_eval_scores_against_labels() -> None:
    from eval.harness import Collaborators, RetrievalExample

    points = [_point("c1", "a.md"), _point("c2", "b.md")]

    class FakeCollaborators(Collaborators):
        def __init__(self) -> None:  # bypass heavy real construction
            self.settings = make_settings()
            self.repository = FakeRepository(points)
            self.embedding_provider = FakeEmbeddingProvider()
            self.reranker = FakeReranker(["c1", "c2"])
            self.generator = None  # type: ignore[assignment]

    examples = [
        RetrievalExample(
            question="q",
            reference=None,
            relevant_filenames=frozenset({"a.md"}),
            relevant_chunk_ids=frozenset(),
        )
    ]
    results = run_retrieval_eval(examples, FakeCollaborators(), ks=(1,))
    assert results["question_count"] == 1
    assert results["aggregate"]["filename"]["hit@1"] == 1.0
    assert results["aggregate"]["chunk"] == {}  # no chunk labels supplied


def test_load_retrieval_dataset_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text(
        '{"question": "q1", "relevant_filenames": ["a.md"]}\n'
        '{"question": "q2", "reference": "r", "relevant_chunk_ids": ["c1"]}\n',
        encoding="utf-8",
    )
    examples = load_retrieval_dataset(path)
    assert len(examples) == 2
    assert examples[0].relevant_filenames == frozenset({"a.md"})
    assert examples[1].reference == "r"
    assert examples[1].relevant_chunk_ids == frozenset({"c1"})


def test_load_retrieval_dataset_rejects_missing_question(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"relevant_filenames": ["a.md"]}\n', encoding="utf-8")
    with pytest.raises(HarnessDataError, match="question"):
        load_retrieval_dataset(path)


def test_load_retrieval_dataset_rejects_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(HarnessDataError, match="at least one"):
        load_retrieval_dataset(path)


def test_parser_requires_subcommand() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_retrieval_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["retrieval", "data.jsonl"])
    assert args.command == "retrieval"
    assert args.max_regression == 0.05
    assert args.ks is None
