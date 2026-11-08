"""Pass 15 answer and tuning fixes: the re-check's rules, the score report's sample and
filters, trace diffs, degraded answers kept out of the cache, summary parts that fail,
query-log documents, follow-up language, and candidates' source numbers."""

from __future__ import annotations

import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import Any

import pytest

from api.answer_cache import AnswerCache
from api.citations import retry_uncited_answer
from api.llm import (
    ChatMessage,
    GenerationError,
    begin_usage,
    note_degraded,
    was_degraded,
)
from api.prompting import generate_followups
from api.query_log import LoggedQuery, logged_query
from api.relevance import _best_bar, score_report, trace_diff
from api.reranking import RerankedChunk
from api.retrieval import RetrievedChunk
from api.routes.traces import _recheck_rows
from api.settings import AppSettings
from api.summarize import SummaryRequest
from api.tracing import (
    QueryTrace,
    TraceCandidate,
    TraceConfig,
    TraceSummary,
    build_trace_candidates,
)
from tests.factories import make_test_settings
from tests.test_summarize import FakeRepository, ScriptedGenerator, _payload, _summarizer

# --- re-check -----------------------------------------------------------------------


def _row(question: str, **fields: Any) -> LoggedQuery:
    values: dict[str, Any] = {
        "created_at": time.time(),
        "workspace": "",
        "question": question,
        "status": "insufficient_context",
        "top_score": None,
        "mode": "stream",
        "trace_id": None,
    }
    values.update(fields)
    return LoggedQuery(**values)


def _scored(filename: str, score: float) -> RerankedChunk:
    return RerankedChunk(
        point_id=filename,
        filename=filename,
        page=1,
        section="",
        chunk_id=f"{filename}-1",
        text="text",
        retrieval_score=0.5,
        rerank_score=score,
    )


class _Phase:
    def __init__(self, top: RerankedChunk | None, below: bool = False) -> None:
        self.scored_chunks = [top] if top is not None else []
        self.context_chunks = [top] if top is not None else []
        self.below_min_score = below


class _Pipeline:
    def __init__(self, results: dict[str, Any]) -> None:
        self.results = results
        self.searched: list[str] = []

    def retrieve(self, query: str, **kwargs: object) -> Any:
        self.searched.append(query)
        result = self.results[query]
        if isinstance(result, Exception):
            raise result
        return result


def test_a_refusal_over_the_same_passage_is_not_covered_now() -> None:
    pipeline = _Pipeline(
        {
            "same?": _Phase(_scored("a.md", 0.81)),
            "better?": _Phase(_scored("a.md", 0.9)),
            "elsewhere?": _Phase(_scored("new.md", 0.5)),
            "never?": _Phase(_scored("a.md", 0.3)),
            "still weak?": _Phase(_scored("a.md", 0.05), below=True),
        }
    )
    rows = [
        _row("same?", top_score=0.8, top_filename="a.md"),
        _row("better?", top_score=0.8, top_filename="a.md"),
        _row("elsewhere?", top_score=0.8, top_filename="a.md"),
        _row("never?"),
        _row("still weak?", top_score=0.01),
    ]

    items, stopped = _recheck_rows(rows, pipeline, 0)  # type: ignore[arg-type]

    assert not stopped
    assert {item.question: item.covered_now for item in items} == {
        "same?": False,
        "better?": True,
        "elsewhere?": True,
        "never?": True,
        "still weak?": False,
    }


def test_the_recheck_searches_the_rewrite_skips_summaries_and_isolates_errors() -> None:
    pipeline = _Pipeline(
        {
            "What about part-timers' leave?": _Phase(_scored("leave.md", 0.7)),
            "broken?": RuntimeError("qdrant down"),
        }
    )
    rows = [
        _row('Summarize the document "a.pdf".', kind="summary"),
        _row("what about part-timers?", retrieval_query="What about part-timers' leave?"),
        _row("broken?"),
    ]

    items, _ = _recheck_rows(rows, pipeline, 0)  # type: ignore[arg-type]

    assert pipeline.searched == ["What about part-timers' leave?", "broken?"]
    assert items[0].question == "what about part-timers?" and items[0].covered_now
    assert items[1].error is not None and not items[1].covered_now


def test_the_recheck_stops_when_its_time_is_up() -> None:
    class Slow(_Pipeline):
        def retrieve(self, query: str, **kwargs: object) -> Any:
            time.sleep(0.05)
            return super().retrieve(query)

    pipeline = Slow({f"q{n}?": _Phase(None) for n in range(20)})
    items, stopped = _recheck_rows(
        [_row(f"q{n}?") for n in range(20)],
        pipeline,  # type: ignore[arg-type]
        0.12,
    )

    assert stopped and 1 <= len(items) < 20


# --- score report -------------------------------------------------------------------


def _summary(score: float, status: str, **fields: Any) -> TraceSummary:
    values: dict[str, Any] = {
        "trace_id": str(random.random()),
        "created_at": 0.0,
        "question": "q",
        "mode": "stream",
        "status": "ok",
        "llm_provider": "ollama",
        "total_ms": 1.0,
        "candidate_count": 1,
        "kept_count": 1,
        "answer_status": status,
        "top_rerank_score": score,
    }
    values.update(fields)
    return TraceSummary(**values)


def test_the_best_bar_matches_a_count_over_every_trace() -> None:
    rng = random.Random(7)
    for _ in range(50):
        answered = [round(rng.random(), 2) for _ in range(rng.randint(0, 30))]
        unanswered = [round(rng.random() * 0.6, 2) for _ in range(rng.randint(0, 30))]
        bar, wrong = _best_bar(answered, unanswered)
        assert wrong == sum(s < bar for s in answered) + sum(s >= bar for s in unanswered)
        scores = sorted({*answered, *unanswered})
        bars = [0.0, *((a + b) / 2 for a, b in zip(scores, scores[1:], strict=False)), 1.0]
        assert wrong == min(
            sum(s < b for s in answered) + sum(s >= b for s in unanswered) for b in bars
        )


def test_the_report_leaves_out_other_rerankers_picked_passages_and_summaries() -> None:
    summaries = [
        _summary(0.9, "answered", reranker_model="jina"),
        _summary(0.9, "answered", reranker_model="minilm"),
        _summary(0.01, "answered", reranker_model="jina", gated=False),
        _summary(0.5, "answered", kind="summary"),
        _summary(0.02, "insufficient_context", reranker_model=None),
    ]

    report = score_report(summaries, 0.15, reranker_model="jina")

    assert report.answered_count == 1 and report.unanswered_count == 1
    assert report.left_out == 3


# --- trace diff ---------------------------------------------------------------------


def _candidate(point_id: str, score: float, selected: bool = True) -> TraceCandidate:
    return TraceCandidate(
        point_id=point_id,
        filename="a.md",
        page=1,
        section="",
        chunk_id=point_id,
        retrieval_score=0.5,
        rerank_score=score,
        kept=True,
        drop_reason=None,
        selected_for_context=selected,
        neighbor_expanded=False,
    )


def _trace(candidates: list[TraceCandidate], config: TraceConfig | None = None) -> QueryTrace:
    return QueryTrace(
        trace_id="t",
        created_at=0.0,
        question="q",
        mode="stream",
        status="ok",
        config=config,
        candidates=candidates,
        prompt_messages=None,
        answer="a",
        cited_source_numbers=[],
        timings=None,
        error=None,
    )


def _config(**fields: Any) -> TraceConfig:
    values: dict[str, Any] = {
        "llm_provider": "ollama",
        "rerank_top_k": 5,
        "max_context_chunks": 5,
        "llm_temperature": 0.1,
        "rerank_min_score": 0.15,
        "fused_top_n": 20,
        "filenames": None,
    }
    values.update(fields)
    return TraceConfig(**values)


def test_a_score_that_moved_is_a_change_and_an_unrecorded_setting_is_not() -> None:
    a = _trace([_candidate("p1", 0.9), _candidate("p2", 0.5)], _config())
    b = _trace(
        [_candidate("p1", 0.6), _candidate("p2", 0.505)],
        _config(rerank_candidates=50, reranker_onnx_file="model_int8.onnx"),
    )

    diff = trace_diff(a, b)

    assert [c.point_id for c in diff.candidates] == ["p1"]
    assert diff.settings == []
    c = _trace([_candidate("p1", 0.9)], _config(rerank_candidates=20))
    assert [s.name for s in trace_diff(b, c).settings] == ["rerank_candidates"]


# --- degraded answers ---------------------------------------------------------------


def test_the_degraded_mark_is_per_answer() -> None:
    begin_usage()
    assert not was_degraded()
    note_degraded()
    assert was_degraded()
    begin_usage()
    assert not was_degraded()


class _Failing:
    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        raise GenerationError("provider down")

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings) -> Iterator[str]:
        raise GenerationError("provider down")


def test_a_failed_citation_retry_marks_the_answer_degraded() -> None:
    begin_usage()
    prompt: list[ChatMessage] = [
        {"role": "system", "content": "Answer from the sources."},
        {"role": "user", "content": "Sources:\n[1] a\n\nQuestion: q"},
    ]
    assert retry_uncited_answer(prompt, "An answer.", _Failing(), make_test_settings()) is None
    assert was_degraded()


class _UncitedThenFailing:
    """The answer cites nothing; the citation retry's call fails."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        self.calls += 1
        if self.calls > 1:
            raise GenerationError("provider down")
        return "Alpha beta is what the guide covers."

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings) -> Iterator[str]:
        yield self.complete(messages, settings)


@pytest.mark.parametrize("stream", [False, True])
def test_an_answer_whose_citation_retry_failed_is_not_cached(stream: bool) -> None:
    from qdrant_client import QdrantClient

    from api.ingest_service import IngestService
    from api.pipeline import RagPipeline
    from api.repository import VectorRepository
    from tests.test_pipeline import (
        FakeReranker,
        ListTraceStore,
        StaticEmbeddingProvider,
        make_embedding,
        make_settings,
    )

    settings = make_settings(citation_retry_mode="llm")
    repository = VectorRepository(QdrantClient(":memory:"))
    embeddings = StaticEmbeddingProvider([make_embedding(1.0)])
    IngestService(repository, embeddings, settings).ingest("guide.txt", b"Intro\nalpha beta")
    cache = AnswerCache(16, 600)
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=embeddings,
        reranker=FakeReranker(),
        generator=_UncitedThenFailing(),
        settings=settings,
        trace_store=ListTraceStore(),
        answer_cache=cache,
    )

    if stream:
        events = list(pipeline.answer_stream("What is alpha?"))
        assert events[-1]["type"] == "done"
    else:
        assert pipeline.answer("What is alpha?").answer
    assert len(cache) == 0


# --- summary parts ------------------------------------------------------------------


class _FlakyNotes(ScriptedGenerator):
    """Notes calls: the first ``fail`` fail (part 1 twice: skipped), then succeed."""

    def __init__(self, fail: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fail = fail

    def _reply(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        if "Excerpts from" in messages[-1]["content"] and self.fail > 0:
            self.fail -= 1
            raise GenerationError("provider hiccup")
        return super()._reply(messages, settings)


def _long_document() -> FakeRepository:
    text = "word " * 150
    return FakeRepository({"a.txt": [_payload("a.txt", n, text) for n in range(1, 9)]})


def test_a_part_that_fails_twice_is_left_out_and_reported() -> None:
    generator = _FlakyNotes(2, notes=["Part two [1].", "Part three [1].", "Part four [1]."])
    settings = make_test_settings(
        llm_provider="ollama", summary_batch_chars=1000, summary_max_batches=4
    )
    cache = AnswerCache(10, 3600)

    result = _summarizer(_long_document(), generator, settings, answer_cache=cache).summarize(
        SummaryRequest(mode="summarize", filenames=("a.txt",))
    )

    assert result.parts_skipped == 1
    assert result.grounded.answer
    assert len(cache._entries) == 0  # noqa: SLF001 - a partial summary isn't kept


def test_a_part_that_fails_once_is_tried_again() -> None:
    generator = _FlakyNotes(1)
    settings = make_test_settings(
        llm_provider="ollama", summary_batch_chars=1000, summary_max_batches=4
    )

    result = _summarizer(_long_document(), generator, settings).summarize(
        SummaryRequest(mode="summarize", filenames=("a.txt",))
    )

    assert result.parts_skipped == 0


def test_every_part_failing_is_still_an_error() -> None:
    generator = _FlakyNotes(1000)
    settings = make_test_settings(
        llm_provider="ollama", summary_batch_chars=1000, summary_max_batches=4
    )

    with pytest.raises(GenerationError):
        _summarizer(_long_document(), generator, settings).summarize(
            SummaryRequest(mode="summarize", filenames=("a.txt",))
        )


def test_notes_cut_off_at_the_limit_are_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.summarize as summarize_module

    monkeypatch.setattr(summarize_module, "stopped_at_token_limit", lambda: True)
    settings = make_test_settings(
        llm_provider="ollama", summary_batch_chars=1000, summary_max_batches=4
    )

    result = _summarizer(_long_document(), ScriptedGenerator(), settings).summarize(
        SummaryRequest(mode="summarize", filenames=("a.txt",))
    )

    assert result.notes_truncated >= 2


# --- query log ----------------------------------------------------------------------


def test_the_log_counts_the_documents_an_answer_cited() -> None:
    cited = replace(_candidate("p1", 0.9), filename="cited.md", source_number=1, cited=True)
    uncited = replace(_candidate("p2", 0.8), filename="other.md", source_number=2)
    trace = replace(
        _trace([cited, uncited], _config()),
        condensed_question="What about part-timers' leave?",
        question="what about part-timers?",
    )

    row = logged_query(trace)

    assert row.filenames == ("cited.md",)
    assert row.kind == "question" and row.top_filename == "cited.md"
    assert row.retrieval_query == "What about part-timers' leave?"
    # Nothing cited (or traced before citations were recorded): the prompt's documents.
    assert logged_query(_trace([_candidate("p1", 0.9)], _config())).filenames == ("a.md",)


def test_a_summary_is_logged_as_one() -> None:
    trace = replace(_trace([], _config(filenames=["a.pdf"])), kind="summary")
    row = logged_query(trace)
    assert row.kind == "summary" and row.filenames == ("a.pdf",)


# --- source numbers -----------------------------------------------------------------


def _retrieved(point_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        point_id=point_id,
        filename="a.md",
        page=1,
        section="",
        chunk_id=point_id,
        text="t",
        score=0.5,
    )


def test_candidates_carry_their_prompt_number_and_whether_they_were_cited() -> None:
    chunks = [_scored(name, score) for name, score in (("p1", 0.9), ("p2", 0.8), ("p3", 0.7))]
    candidates = build_trace_candidates(
        [_retrieved("p1"), _retrieved("p2"), _retrieved("p3")],
        chunks,
        chunks,
        chunks,
        min_score=0.1,
        context_budget_dropped_ids={"p2"},
        cited_source_numbers={2},
    )

    numbers = {c.point_id: (c.source_number, c.cited) for c in candidates}
    assert numbers == {"p1": (1, False), "p2": (None, False), "p3": (2, True)}


# --- follow-ups ---------------------------------------------------------------------


def test_follow_ups_are_asked_for_in_the_answer_language() -> None:
    seen: list[list[ChatMessage]] = []

    class Recording:
        def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
            seen.append(list(messages))
            return "Wie viele Tage?"

        def stream(self, messages: Sequence[ChatMessage], settings: AppSettings) -> Iterator[str]:
            yield ""

    settings = make_test_settings(followup_count=2, answer_language="German")
    out = generate_followups("q", "a", [_scored("a.md", 0.9)], Recording(), settings)

    assert out == ["Wie viele Tage?"]
    assert seen[0][0]["content"].endswith("Write the questions in German.")


# --- CLI ----------------------------------------------------------------------------


def test_the_cli_says_when_summary_parts_were_left_out_or_cut_off() -> None:
    import io

    from cli.__main__ import _print_sources
    from cli.client import StreamResult

    out = io.StringIO()
    done = {"sources": [], "parts_skipped": 1, "notes_truncated": 2}
    _print_sources(StreamResult(done=done, followups=[]), out)

    assert "1 part(s) left out" in out.getvalue()
    assert "notes on 2 part(s) stopped" in out.getvalue()


# --- summaries of part of a document ------------------------------------------------


def test_a_summary_reads_only_its_page_range_and_is_cached_apart() -> None:
    from api.repository import PassageFilter

    repository = FakeRepository(
        {"a.txt": [_payload("a.txt", n, f"Page {n} text.", page=n) for n in range(1, 7)]}
    )
    generator = ScriptedGenerator(answer="Pages two to three [1].")
    cache = AnswerCache(10, 3600)
    summarizer = _summarizer(repository, generator, answer_cache=cache)
    narrowed = SummaryRequest(
        mode="summarize", filenames=("a.txt",), narrow=PassageFilter(page_min=2, page_max=3)
    )

    result = summarizer.summarize(narrowed)

    prompt = generator.calls[-1][-1]["content"]
    assert "Page 2 text." in prompt and "Page 3 text." in prompt
    assert "Page 1 text." not in prompt and "Page 4 text." not in prompt
    assert result.chunks_total == 2
    from api.summarize import task_description

    assert task_description(narrowed).endswith("Only pages 2-3.")
    whole = summarizer.summarize(SummaryRequest(mode="summarize", filenames=("a.txt",)))
    assert whole.grounded.cached is False and whole.chunks_total == 6


def test_a_page_range_is_for_a_summary_of_one_document() -> None:
    from pydantic import ValidationError

    from api.schemas import SummaryRequest as SummaryBody

    SummaryBody(mode="summarize", filenames=["a.txt"], page_min=2)
    with pytest.raises(ValidationError, match="narrows a summary of one document"):
        SummaryBody(mode="compare", filenames=["a.txt", "b.txt"], page_min=2)
