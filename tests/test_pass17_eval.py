"""Pass 17: eval fixes."""

from __future__ import annotations

from eval.compare import Run
from eval.harness import settings_fingerprint
from tests.factories import make_test_settings


def test_a_repeated_question_is_compared_turn_by_turn() -> None:
    run = Run("answer", [{"question": "And then?", "a": 1}, {"question": "And then?", "a": 2}])
    keyed = run.questions()
    assert keyed["And then?"]["a"] == 1
    assert keyed["And then? (#2)"]["a"] == 2


def test_a_follow_up_dataset_records_the_model_that_rewrote_its_questions() -> None:
    settings = make_test_settings()
    assert "llm_model" not in settings_fingerprint(settings, "reranked")
    printed = settings_fingerprint(settings, "reranked", follow_ups=True)
    assert printed["llm_provider"] == settings.llm_provider
    assert printed["condense_max_tokens"] == settings.condense_max_tokens


def test_a_sweep_replays_the_reranked_arm_exactly_and_scores_each_combination() -> None:
    from eval.harness import Collaborators, RetrievalExample, rank_for_question, run_sweep
    from eval.sweep import parse_grid
    from tests.test_eval_harness import (
        FakeEmbeddingProvider,
        FakeRepository,
        FakeReranker,
        _point,
        make_settings,
    )

    points = [_point("c1", "a.md"), _point("c2", "b.md"), _point("c3", "a.md")]
    reranker = FakeReranker(["c3", "c1", "c2"])  # sigmoid scores ~0.95, 0.88, 0.73
    settings = make_settings(rerank_min_score=0.0, rerank_top_k=3)

    class FakeCollaborators(Collaborators):
        def __init__(self) -> None:
            self.settings = settings
            self.repository = FakeRepository(points)
            self.embedding_provider = FakeEmbeddingProvider()
            self.reranker = reranker
            self.generator = None  # type: ignore[assignment]

    example = RetrievalExample(
        question="q",
        reference=None,
        relevant_filenames=frozenset({"b.md"}),
        relevant_chunk_ids=frozenset({"c2"}),
    )
    result = run_sweep(
        [example],
        FakeCollaborators(),
        parse_grid(["rerank_min_score=0.8", "rerank_top_k=1"]),
        (1, 3),
        "chunk.mrr",
    )
    rows = {
        (r["settings"]["rerank_min_score"], r["settings"]["rerank_top_k"]): r
        for r in result["rows"]
    }
    assert len(rows) == 4
    # The current settings replay to exactly what the reranked arm returns.
    actual = rank_for_question(
        "q",
        repository=FakeRepository(points),
        embedding_provider=FakeEmbeddingProvider(),
        reranker=reranker,
        settings=settings,
    )
    assert actual.chunk_ids == ["c3", "c1", "c2"]
    assert rows[(0.0, 3)]["current"] and rows[(0.0, 3)]["aggregate"]["chunk"]["mrr"] == 1 / 3
    # Past the bar or the top-K cut, c2 is gone.
    assert rows[(0.8, 3)]["aggregate"]["chunk"]["mrr"] == 0.0
    assert rows[(0.0, 1)]["aggregate"]["chunk"]["mrr"] == 0.0
    assert result["rows"][0]["settings"] == {"rerank_min_score": 0.0, "rerank_top_k": 3}


def test_a_grid_naming_a_setting_the_sweep_cant_vary_is_refused() -> None:
    import pytest

    from eval.sweep import SweepGridError, parse_grid

    with pytest.raises(SweepGridError, match="rrf_k"):
        parse_grid(["rrf_k=30,60"])
    with pytest.raises(SweepGridError):
        parse_grid(["rerank_top_k=four"])


def test_a_generated_question_is_kept_only_with_a_phrase_from_its_passage() -> None:
    from eval.generate import Passage, generate_rows, parse_question

    passage = "Employees must book flights at least fourteen days ahead through the agency."
    assert parse_question(
        'Sure: {"question": "How early do I book flights", "answer_phrase": "fourteen days ahead"}',
        passage,
    ) == ("How early do I book flights?", "fourteen days ahead")
    assert parse_question('{"question": "Q?", "answer_phrase": "ten days"}', passage) is None
    assert parse_question("no json here", passage) is None
    assert parse_question('{"question": ""}', passage) is None

    class Replies:
        def __init__(self, replies: list[str]) -> None:
            self.replies = replies

        def complete(self, messages: object, settings: object) -> str:
            return self.replies.pop(0)

        def stream(self, messages: object, settings: object) -> object:
            raise AssertionError

    rows, skipped = generate_rows(
        [Passage("travel.md", "c1", passage), Passage("travel.md", "c2", passage)],
        Replies(
            [
                '{"question": "How early do I book flights?", "answer_phrase": "fourteen days"}',
                '{"question": "Who pays?", "answer_phrase": "the employer"}',
            ]
        ),  # type: ignore[arg-type]
        make_test_settings(),
    )
    assert skipped == 1
    assert rows == [
        {
            "question": "How early do I book flights?",
            "relevant_filenames": ["travel.md"],
            "relevant_passages": ["fourteen days"],
            "answer_contains": ["fourteen days"],
        }
    ]


def test_passages_are_sampled_across_documents() -> None:
    from eval.generate import sample_passages

    long = " ".join(["word"] * 20)

    class Repo:
        def filename_metadata(self, settings: object) -> dict[str, object]:
            return {"a.md": None, "b.md": None}

        def chunks_for_filename(self, settings: object, filename: str) -> list[dict[str, str]]:
            return [{"chunk_id": f"{filename}-{i}", "text": long} for i in range(5)] + [
                {"chunk_id": "short", "text": "too short"}
            ]

    picked = sample_passages(Repo(), make_test_settings(), 4, seed=1)
    assert len(picked) == 4
    assert {p.filename for p in picked} == {"a.md", "b.md"}
    assert all(p.chunk_id != "short" for p in picked)


def test_the_query_log_drafts_labelled_rows_and_a_review_list() -> None:
    from eval.generate import rows_from_queries

    labelled, review = rows_from_queries(
        [
            {"question": "How long  is leave?", "status": "insufficient_context", "created_at": 1},
            {
                "question": "how long is leave?",
                "status": "ok",
                "filenames": ["leave.md"],
                "created_at": 2,
            },
            {"question": "Is there a gym?", "status": "insufficient_context", "created_at": 3},
            {"question": "Broke", "status": "error", "created_at": 4},
            {"question": "Summarize x", "status": "ok", "kind": "summary", "created_at": 5},
        ]
    )
    assert labelled == [{"question": "how long is leave?", "relevant_filenames": ["leave.md"]}]
    assert [r["question"] for r in review] == ["Is there a gym?"]
    assert review[0]["expect_no_context"] is True


def test_the_score_report_can_take_its_sides_from_ratings() -> None:
    from dataclasses import replace

    from api.relevance import score_report
    from tests.test_relevance import _summary

    # The model refused everything under 0.4; people rated the 0.3s helpful anyway.
    traces = [
        *(replace(_summary("answered", 0.8), trace_id=f"a{i}") for i in range(20)),
        *(replace(_summary("insufficient_context", 0.3), trace_id=f"b{i}") for i in range(20)),
        *(replace(_summary("insufficient_context", 0.1), trace_id=f"c{i}") for i in range(20)),
    ]
    by_model = score_report(traces, 0.4)
    assert by_model.labels == "model" and by_model.unanswered_count == 40

    ratings = {f"a{i}": "up" for i in range(20)} | {f"b{i}": "up" for i in range(20)}
    ratings |= {f"c{i}": "down" for i in range(20)}
    by_people = score_report(traces, 0.4, ratings=ratings)
    assert by_people.labels == "feedback"
    assert (by_people.answered_count, by_people.unanswered_count) == (40, 20)
    assert by_people.suggested_min_score is not None and by_people.suggested_min_score < 0.3
    # Unrated traces are left out, not counted for either side.
    assert score_report(traces, 0.4, ratings={"a1": "up"}).left_out == 59


def test_a_metric_change_carries_a_paired_bootstrap_interval() -> None:
    from eval.compare import diff_runs, format_diff, paired_interval

    def run(hits: list[float]) -> Run:
        return Run(
            "retrieval",
            {
                "aggregate": {"filename": {"hit@1": sum(hits) / len(hits)}},
                "per_question": [
                    {"question": f"q{i}", "filename": {"hit@1": hit}} for i, hit in enumerate(hits)
                ],
            },
        )

    # One question of twenty flipped: an interval that spans 0.
    one_flip = diff_runs(run([1.0] * 10 + [0.0] * 10), run([1.0] * 11 + [0.0] * 9))
    low, high = one_flip.intervals["filename hit@1"]
    assert low <= 0 <= high
    assert "may be noise" in format_diff(one_flip)
    # Every question got better: clear of 0.
    all_up = diff_runs(run([0.0] * 20), run([1.0] * 20))
    assert all_up.intervals["filename hit@1"][0] > 0
    assert paired_interval([(0.0, 1.0)]) is None


def test_a_dataset_row_can_be_asked_within_a_scope() -> None:
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from eval.harness import Collaborators, load_retrieval_dataset, run_retrieval_eval
    from tests.test_eval_harness import (
        FakeEmbeddingProvider,
        FakeRepository,
        FakeReranker,
        _point,
        make_settings,
    )

    seen: list[object] = []

    class Recording(FakeRepository):
        def hybrid_search(self, settings, query_embedding, filenames=None, tags=None, **_):  # type: ignore[no-untyped-def]
            seen.append((filenames, tags))
            return super().hybrid_search(settings, query_embedding, filenames, tags)

    class FakeCollaborators(Collaborators):
        def __init__(self) -> None:
            self.settings = make_settings()
            self.repository = Recording([_point("c1", "a.md")])
            self.embedding_provider = FakeEmbeddingProvider()
            self.reranker = FakeReranker(["c1"])
            self.generator = None  # type: ignore[assignment]

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "d.jsonl"
        path.write_text(
            '{"question": "q", "relevant_filenames": ["a.md"], "filenames": ["a.md"]}\n',
            encoding="utf-8",
        )
        examples = load_retrieval_dataset(path)
    run_retrieval_eval(examples, FakeCollaborators(), ks=(1,))
    assert seen and seen[0][0] == ["a.md"]


def test_answers_are_scored_on_whether_they_cite_the_right_passages() -> None:
    from eval.answer_eval import AnswerRow, answer_metrics
    from eval.harness import RetrievalExample

    example = RetrievalExample(
        question="q",
        reference=None,
        relevant_filenames=frozenset({"a.md"}),
        relevant_chunk_ids=frozenset({"c1"}),
    )
    row = AnswerRow("q", "a", [], None, cited_filenames=["a.md"], cited_chunk_ids=["c1", "c2"])
    metrics = answer_metrics([example], [row])
    assert metrics["cited_chunk_precision"] == 0.5
    # A passage label resolved to its chunk counts too.
    passage_only = RetrievalExample("q", None, frozenset({"a.md"}), frozenset())
    resolved = answer_metrics([passage_only], [row], {0: frozenset({"c2"})})
    assert resolved["cited_chunk_precision"] == 0.5
    assert "cited_chunk_precision" not in answer_metrics([passage_only], [row])
