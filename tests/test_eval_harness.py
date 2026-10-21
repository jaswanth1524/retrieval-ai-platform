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
from tests.factories import make_test_settings


def make_settings(**overrides: Any) -> AppSettings:
    defaults: dict[str, Any] = {
        "qdrant_url": ":memory:",
        "embedding_model_tag": "test-embedding:v1",
        "qdrant_dense_vector_size": 3,  # the fake embedder's size
        "rerank_top_k": 3,
        "rerank_candidates": 10,
        "rerank_min_score": 0.0,
    }
    defaults.update(overrides)
    return make_test_settings(**defaults)


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
        tags: Sequence[str] | None = None,
    ) -> list[models.ScoredPoint]:
        return self._points

    def search_leg(
        self, settings: AppSettings, query_embedding: EmbeddedText, leg: str
    ) -> list[models.ScoredPoint]:
        # Dense sees the points in order, sparse reversed: the arms must differ.
        return self._points if leg == "dense" else list(reversed(self._points))

    def fetch_neighbors(self, settings: AppSettings, filename: str, ordinals: Any) -> list[Any]:
        return []


class FakeReranker:
    """Scores documents by their order in a preset chunk_id priority list."""

    def __init__(self, priority: list[str]) -> None:
        self._priority = priority

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        # Higher score for earlier-priority text; documents are the chunk texts behind
        # their "filename › section" prefix.
        texts = [doc.split("\n", 1)[-1] for doc in documents]
        return [
            float(len(self._priority) - self._priority.index(text))
            if text in self._priority
            else 0.0
            for text in texts
        ]


def _point(chunk_id: str, filename: str, ordinal: int | None = None) -> models.ScoredPoint:
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
            # Distinct by default, so the diversity filter sees no neighbours.
            "chunk_ordinal": ordinal if ordinal is not None else 10 * int(chunk_id[1:]),
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


def test_rank_for_question_applies_the_diversity_filter_like_the_api() -> None:
    """The harness used to score rerank output directly, so a regression in the
    near-duplicate filter the API applies next was invisible to it."""

    points = [_point("c1", "a.md", ordinal=4), _point("c2", "a.md", ordinal=5)]
    result = rank_for_question(
        "q",
        repository=FakeRepository(points),
        embedding_provider=FakeEmbeddingProvider(),
        reranker=FakeReranker(["c1", "c2"]),
        settings=make_settings(),
    )
    # c2 is c1's ordinal neighbour in the same file: the API keeps one of them.
    assert result.chunk_ids == ["c1"]


def test_each_arm_scores_its_own_stage() -> None:
    points = [_point("c1", "a.md"), _point("c2", "b.md"), _point("c3", "a.md")]

    def run(arm: str) -> list[str]:
        return rank_for_question(
            "q",
            repository=FakeRepository(points),
            embedding_provider=FakeEmbeddingProvider(),
            reranker=FakeReranker(["c3", "c1", "c2"]),
            settings=make_settings(),
            arm=arm,
        ).chunk_ids

    assert run("reranked") == ["c3", "c1", "c2"]
    assert run("fused") == ["c1", "c2", "c3"]
    assert run("dense") == ["c1", "c2", "c3"]
    assert run("sparse") == ["c3", "c2", "c1"]
    with pytest.raises(ValueError, match="Unknown arm"):
        run("bogus")


def test_baseline_checks_refuse_a_run_they_cannot_compare() -> None:
    from eval.metrics import baseline_shape_problems

    baseline = {
        "k_values": [1, 3],
        "question_count": 5,
        "fingerprint": {"arm": "reranked"},
        "aggregate": {"filename": {"hit@1": 0.8}, "chunk": {}},
    }
    same = {**baseline, "aggregate": {"filename": {"hit@1": 0.7}, "chunk": {}}}

    assert baseline_shape_problems(same, baseline) == []
    assert baseline_shape_problems({**same, "k_values": [5]}, baseline)
    assert baseline_shape_problems({**same, "question_count": 4}, baseline)
    assert baseline_shape_problems({**same, "fingerprint": {"arm": "dense"}}, baseline)
    # A setting that ranks results, changed since the baseline: refused, not just noted.
    measured = {**baseline, "fingerprint": {"arm": "reranked", "reranker_onnx_file": "auto"}}
    int8 = {**same, "fingerprint": {"arm": "reranked", "reranker_onnx_file": "auto"}}
    fp32 = {**same, "fingerprint": {"arm": "reranked", "reranker_onnx_file": ""}}
    assert baseline_shape_problems(int8, measured) == []
    assert "reranker_onnx_file differs" in baseline_shape_problems(fp32, measured)[0]
    # A key the older baseline never recorded is only a note.
    assert baseline_shape_problems(int8, baseline) == []
    # So is the machine: recorded to explain a drift, never a refusal.
    on_amd64 = {**baseline, "fingerprint": {"arm": "reranked", "platform_machine": "x86_64"}}
    on_arm64 = {**same, "fingerprint": {"arm": "reranked", "platform_machine": "arm64"}}
    assert baseline_shape_problems(on_arm64, on_amd64) == []
    unlabeled = {**same, "aggregate": {"filename": {}, "chunk": {}}}
    assert "no metric" in baseline_shape_problems(unlabeled, baseline)[0]


def test_validate_checks_a_dataset_without_a_database(tmp_path: Path) -> None:
    from eval.harness import main

    labeled = tmp_path / "labeled.jsonl"
    labeled.write_text('{"question": "q", "relevant_filenames": ["a.md"]}\n')
    unlabeled = tmp_path / "unlabeled.jsonl"
    unlabeled.write_text('{"question": "q"}\n')

    assert main(["validate", str(labeled)]) == 0
    assert main(["validate", str(unlabeled)]) == 2
    assert main(["validate", "eval/datasets/retrieval_eval.sample.jsonl"]) == 0


class PassageRepository(FakeRepository):
    """Adds the payload reads passage resolution uses."""

    def chunks_for_filename(self, settings: AppSettings, filename: str) -> list[dict[str, Any]]:
        return [
            dict(point.payload or {})
            for point in self._points
            if (point.payload or {}).get("filename") == filename
        ]

    def filename_metadata(self, settings: AppSettings) -> dict[str, Any]:
        return {(p.payload or {})["filename"]: None for p in self._points}


def test_passages_resolve_to_the_chunks_containing_them_and_are_scored() -> None:
    from eval.harness import Collaborators, RetrievalExample

    points = [_point("c1", "a.md"), _point("c2", "b.md")]

    class FakeCollaborators(Collaborators):
        def __init__(self) -> None:
            self.settings = make_settings()
            self.repository = PassageRepository(points)
            self.embedding_provider = FakeEmbeddingProvider()
            self.reranker = FakeReranker(["c2", "c1"])
            self.generator = None  # type: ignore[assignment]

    examples = [
        RetrievalExample(
            question="q",
            reference=None,
            relevant_filenames=frozenset({"a.md"}),
            relevant_chunk_ids=frozenset(),
            relevant_passages=frozenset({"  C1 "}),
        )
    ]
    results = run_retrieval_eval(examples, FakeCollaborators(), ks=(1, 2))

    # c1 is the labeled chunk, ranked second behind c2.
    assert results["aggregate"]["chunk"]["hit@1"] == 0.0
    assert results["aggregate"]["chunk"]["hit@2"] == 1.0


def test_a_passage_in_no_chunk_fails_the_run() -> None:
    from eval.harness import resolve_passages

    example = load_retrieval_dataset_from([{"question": "q", "relevant_passages": ["nowhere"]}])
    with pytest.raises(HarnessDataError, match="in no indexed chunk"):
        resolve_passages(example, PassageRepository([_point("c1", "a.md")]), make_settings())


def test_a_passage_only_row_searches_every_file_even_when_other_rows_name_files() -> None:
    from eval.harness import resolve_passages

    examples = load_retrieval_dataset_from(
        [
            {"question": "q", "relevant_filenames": ["a.md"]},
            {"question": "q", "relevant_passages": ["c2"]},
            {"question": "named", "relevant_filenames": ["a.md"], "relevant_passages": ["c1"]},
        ]
    )
    repository = PassageRepository([_point("c1", "a.md"), _point("c2", "b.md")])

    # Row 2 names no file, so b.md is searched although only a.md is named elsewhere;
    # results are keyed by position, so the duplicate question text doesn't collide.
    assert resolve_passages(examples, repository, make_settings()) == {
        1: frozenset({"c2"}),
        2: frozenset({"c1"}),
    }


def load_retrieval_dataset_from(rows: list[dict[str, Any]]) -> list[Any]:
    from eval.harness import _example_from_mapping

    return [_example_from_mapping(row, index) for index, row in enumerate(rows, 1)]


def test_ingest_skips_dotfiles_and_unsupported_types(tmp_path: Path) -> None:
    from eval.harness import ingestible_files

    for name in ("b.md", "a.TXT", ".DS_Store", "notes.xyz", ".hidden.md"):
        (tmp_path / name).write_text("x")
    (tmp_path / "sub").mkdir()

    files, skipped = ingestible_files(tmp_path)

    assert [path.name for path in files] == ["a.TXT", "b.md"]
    assert [path.name for path in skipped] == [".DS_Store", ".hidden.md", "notes.xyz"]


def test_the_fingerprint_records_every_setting_that_moves_the_ranking() -> None:
    from eval.harness import settings_fingerprint
    from tests.factories import make_test_settings

    fingerprint = settings_fingerprint(make_test_settings(), "reranked")

    for key in (
        "rrf_k",
        "dense_retrieval_limit",
        "sparse_retrieval_limit",
        "query_expansion_count",
        "reranker_batch_size",
        "platform_machine",
        "rerank_relative_floor",
    ):
        assert key in fingerprint


def test_unanswerable_questions_are_scored_on_keeping_nothing(tmp_path: Path) -> None:
    from eval.harness import Collaborators, RetrievalExample, settings_fingerprint

    points = [_point("c1", "a.md")]

    class FakeCollaborators(Collaborators):
        def __init__(self, min_score: float) -> None:
            self.settings = make_settings(rerank_min_score=min_score)
            self.repository = FakeRepository(points)
            self.embedding_provider = FakeEmbeddingProvider()
            # c1 scores 0 raw (0.5 once normalized): below a 0.9 floor.
            self.reranker = FakeReranker(["other"])
            self.generator = None  # type: ignore[assignment]

    examples = [
        RetrievalExample(
            question="off topic",
            reference=None,
            relevant_filenames=frozenset(),
            relevant_chunk_ids=frozenset(),
            expect_no_context=True,
        )
    ]
    kept_nothing = run_retrieval_eval(examples, FakeCollaborators(0.9), ks=(1,))
    assert kept_nothing["aggregate"]["unanswerable"] == {"no_context_rate": 1.0}
    assert kept_nothing["per_question"][0]["no_context"] is True
    kept_something = run_retrieval_eval(examples, FakeCollaborators(0.0), ks=(1,))
    assert kept_something["aggregate"]["unanswerable"] == {"no_context_rate": 0.0}
    # The multilingual knobs are part of what a baseline was measured with.
    fingerprint = settings_fingerprint(make_settings(), "reranked")
    assert fingerprint["sparse_language"] == "english"
    assert fingerprint["dense_passage_prefix"] == ""

    path = tmp_path / "d.jsonl"
    path.write_text('{"question": "q", "expect_no_context": true}\n', encoding="utf-8")
    assert load_retrieval_dataset(path)[0].labeled
    path.write_text(
        '{"question": "q", "expect_no_context": true, "relevant_filenames": ["a.md"]}\n',
        encoding="utf-8",
    )
    with pytest.raises(HarnessDataError, match="expect_no_context"):
        load_retrieval_dataset(path)
    path.write_text('{"question": "q", "expect_no_context": "yes"}\n', encoding="utf-8")
    with pytest.raises(HarnessDataError, match="true or false"):
        load_retrieval_dataset(path)


def test_a_follow_up_is_retrieved_with_its_rewritten_query() -> None:
    from eval.harness import RetrievalExample

    points = [_point("c1", "a.md"), _point("c2", "b.md")]
    seen: list[str] = []

    class Rewriter:
        def complete(self, messages: Any, settings: Any) -> str:
            return "standalone question about c2"

        def stream(self, messages: Any, settings: Any) -> Any:
            yield ""

    class RecordingReranker(FakeReranker):
        def score(self, query: str, documents: Sequence[str]) -> list[float]:
            seen.append(query)
            return super().score(query, documents)

    rank_for_question(
        "And that one?",
        repository=FakeRepository(points),
        embedding_provider=FakeEmbeddingProvider(),
        reranker=RecordingReranker(["c2", "c1"]),
        settings=make_settings(),
        generator=Rewriter(),
        history=({"role": "user", "content": "Tell me about c2"},),
    )
    assert seen == ["standalone question about c2"]
    example = RetrievalExample(
        question="q", reference=None, relevant_filenames=frozenset(), relevant_chunk_ids=frozenset()
    )
    assert example.history == ()


def test_history_in_a_dataset_is_validated(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text(
        '{"question": "q", "history": [{"role": "user", "content": "earlier"}], '
        '"relevant_filenames": ["a.md"]}\n',
        encoding="utf-8",
    )
    assert load_retrieval_dataset(path)[0].history == ({"role": "user", "content": "earlier"},)
    path.write_text('{"question": "q", "history": [{"role": "bot"}]}\n', encoding="utf-8")
    with pytest.raises(HarnessDataError, match="history"):
        load_retrieval_dataset(path)


def test_answer_metrics_measure_grounding() -> None:
    from eval.harness import AnswerRow, RetrievalExample, answer_metrics

    def example(files: set[str], no_context: bool = False) -> RetrievalExample:
        return RetrievalExample(
            question="q",
            reference=None,
            relevant_filenames=frozenset(files),
            relevant_chunk_ids=frozenset(),
            expect_no_context=no_context,
        )

    def row(cited: list[str], status: str) -> AnswerRow:
        return AnswerRow("q", "a", ["c"], None, cited_filenames=cited, answer_status=status)

    metrics = answer_metrics(
        [example({"a.md"}), example({"b.md"}), example(set(), no_context=True)],
        [row(["a.md"], "answered"), row(["c.md"], "uncited"), row([], "no_context")],
    )
    assert metrics == {
        "cited_relevant_rate": 0.5,
        "answered_rate": 0.5,
        "refused_unanswerable_rate": 1.0,
        "uncited_rate": 1 / 3,
    }
    assert row(["a.md"], "answered").to_dict()["cited_filenames"] == ["a.md"]
