from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from typing import Any, cast

import pytest

from api.answer_cache import AnswerCache
from api.documents import DocumentNotFoundError
from api.llm import ChatMessage
from api.repository import VectorRepository
from api.reranking import RerankedChunk
from api.settings import AppSettings
from api.summarize import (
    NOTHING_RELEVANT,
    SUMMARY_MAX_DOCUMENT_CHUNKS,
    DocumentSummarizer,
    SummaryRequest,
    SummaryTooLargeError,
    batch_char_budget,
    payload_chunk,
    select_parts,
    task_description,
)
from api.tracing import QueryTrace
from tests.factories import make_test_settings


def _payload(filename: str, ordinal: int, text: str, page: int = 1) -> dict[str, object]:
    return {
        "filename": filename,
        "chunk_id": f"{filename}-{ordinal}",
        "chunk_ordinal": ordinal,
        "page": page,
        "section": f"Section {ordinal}",
        "text": text,
    }


class FakeRepository:
    def __init__(self, documents: dict[str, list[dict[str, object]]]) -> None:
        self.documents = documents
        self.counts: dict[str, int] = {}

    def ensure_ready(self, settings: AppSettings) -> None:
        return None

    def chunk_count_for_filename(self, settings: AppSettings, filename: str) -> int:
        return self.counts.get(filename, len(self.documents.get(filename, [])))

    def chunks_for_filename(self, settings: AppSettings, filename: str) -> list[dict[str, object]]:
        return list(self.documents.get(filename, []))


class ScriptedGenerator:
    """Answers notes prompts with ``notes`` (by call) and the final prompt with ``answer``."""

    def __init__(self, notes: Sequence[str] = (), answer: str = "It covers alpha [1].") -> None:
        self.notes = list(notes)
        self.answer = answer
        self.calls: list[list[ChatMessage]] = []
        self.max_tokens: list[int] = []

    def _reply(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        self.calls.append(list(messages))
        self.max_tokens.append(int(settings.llm_max_tokens))
        if "Excerpts from" in messages[-1]["content"]:
            return self.notes.pop(0) if self.notes else "Nothing [1]."
        return self.answer

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        return self._reply(messages, settings)

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings) -> Iterator[str]:
        yield self._reply(messages, settings)


class Traces:
    def __init__(self) -> None:
        self.traces: list[QueryTrace] = []

    def add(self, trace: QueryTrace) -> None:
        self.traces.append(trace)


def _settings(**overrides: Any) -> AppSettings:
    return make_test_settings(**overrides)


def _summarizer(
    repository: FakeRepository,
    generator: ScriptedGenerator,
    settings: AppSettings | None = None,
    **kwargs: Any,
) -> DocumentSummarizer:
    return DocumentSummarizer(
        repository=cast(VectorRepository, repository),
        generator=generator,
        settings=settings or _settings(),
        **kwargs,
    )


def _chunk(ordinal: int, length: int) -> RerankedChunk:
    return RerankedChunk(
        point_id=str(ordinal),
        filename="a.txt",
        page=1,
        section="",
        chunk_id=str(ordinal),
        text="x" * length,
        retrieval_score=0.0,
        rerank_score=0.0,
        chunk_ordinal=ordinal,
    )


def test_select_parts_keeps_every_passage_when_they_fit() -> None:
    chunks = [_chunk(index, 100) for index in range(10)]
    parts = select_parts(chunks, budget=1000, max_parts=4)
    assert [chunk.chunk_id for part in parts for chunk in part] == [str(i) for i in range(10)]
    assert len(parts) <= 4


def test_select_parts_samples_evenly_across_a_long_document() -> None:
    chunks = [_chunk(index, 500) for index in range(100)]
    parts = select_parts(chunks, budget=1100, max_parts=3)
    picked = [int(chunk.chunk_id) for part in parts for chunk in part]
    assert len(parts) <= 3
    # The start, the end and the middle, not just the first pages.
    assert picked[0] == 0 and picked[-1] == 99
    assert any(40 <= ordinal <= 60 for ordinal in picked)
    assert picked == sorted(picked)


def test_a_passage_longer_than_a_part_is_cut_from_its_end() -> None:
    parts = select_parts([_chunk(0, 5000)], budget=1000, max_parts=1)
    assert len(parts) == 1
    # What the budget leaves after the passage's header ("a.txt", no section, 48).
    assert parts[0][0].text == "x" * (1000 - 5 - 48)


def test_a_cut_passage_leaves_room_for_a_long_header() -> None:
    from dataclasses import replace

    from api.summarize import _packed_size

    long_header = replace(_chunk(0, 5000), filename="n" * 255 + ".txt", section="s" * 120)
    for budget in (1000, 1500):
        parts = select_parts([long_header, _chunk(1, 5000)], budget=budget, max_parts=5)
        assert all(sum(_packed_size(chunk) for chunk in part) <= budget for part in parts)
        assert parts[0][0].text  # still some text


def test_batch_shares_pass_a_small_documents_unused_parts_on() -> None:
    from api.summarize import batch_shares

    assert batch_shares([1, 10], 6) == [1, 5]
    assert batch_shares([10, 1], 6) == [5, 1]
    assert batch_shares([10, 10], 6) == [3, 3]
    assert batch_shares([2, 2], 6) == [2, 2]
    assert batch_shares([5, 5], 1) == [1, 1]  # never below one each
    assert batch_shares([0], 4) == [1]


def test_a_compared_short_document_leaves_its_parts_to_the_long_one() -> None:
    long_text = "word " * 150
    repository = FakeRepository(
        {
            "short.txt": [_payload("short.txt", 1, "Plan A costs 5.")],
            "long.txt": [_payload("long.txt", index, long_text) for index in range(1, 11)],
        }
    )
    generator = ScriptedGenerator(answer="A [1] and B [2].")
    settings = _settings(summary_batch_chars=1000, summary_max_batches=6)

    result = _summarizer(repository, generator, settings).summarize(
        SummaryRequest(mode="compare", filenames=("short.txt", "long.txt"))
    )

    # 1 part for the short one, the other 5 for the long one (an even split gave it 3).
    notes_calls = [call for call in generator.calls if "Excerpts from" in call[-1]["content"]]
    assert len(notes_calls) == 6
    assert result.chunks_read == 1 + 5


def test_a_compared_document_with_no_readable_text_is_an_error() -> None:
    repository = FakeRepository(
        {
            "a.txt": [_payload("a.txt", 1, "Plan A costs 5.")],
            "empty.txt": [{"filename": "empty.txt", "chunk_id": "c", "text": "  "}],
        }
    )
    generator = ScriptedGenerator()

    from api.summarize import NothingToCompareError

    with pytest.raises(NothingToCompareError, match="'empty.txt' has no readable text"):
        _summarizer(repository, generator).summarize(
            SummaryRequest(mode="compare", filenames=("a.txt", "empty.txt"))
        )
    assert generator.calls == []


def test_only_notes_that_are_the_marker_are_dropped() -> None:
    texts = [f"Passage {index} " + "word " * 150 for index in range(1, 4)]  # a part each
    repository = FakeRepository(
        {"a.txt": [_payload("a.txt", index, text) for index, text in enumerate(texts, 1)]}
    )
    mentions = "Alpha is defined [1]; the rest is NOTHING RELEVANT to cost."
    generator = ScriptedGenerator(notes=[mentions, " nothing relevant. ", "Gamma [3]."])
    settings = _settings(summary_batch_chars=1000, summary_max_batches=8)

    _summarizer(repository, generator, settings).summarize(
        SummaryRequest(mode="summarize", filenames=("a.txt",))
    )

    combine = generator.calls[-1][-1]["content"]
    assert "Alpha is defined [1]" in combine and "Gamma [3]" in combine
    assert "part 2 of 3" not in combine  # the marker-only note is left out


def test_the_part_size_shrinks_to_fit_a_known_context_window() -> None:
    assert batch_char_budget(_settings()) == 9000
    small = _settings(llm_context_window=4096, summary_max_tokens=1024)
    assert batch_char_budget(small) == (4096 - 1024 - 600) * 3
    tiny = _settings(llm_context_window=1024)
    assert batch_char_budget(tiny) == 1000


def test_payload_chunk_skips_points_without_text() -> None:
    assert payload_chunk({"filename": "a", "chunk_id": "c", "text": "  "}) is None
    chunk = payload_chunk({**_payload("a.txt", 3, "Body"), "page_end": 4})
    assert chunk is not None
    assert (chunk.page, chunk.page_end, chunk.chunk_ordinal) == (1, 4, 3)


def test_task_description_keeps_names_and_focus_on_one_line() -> None:
    request = SummaryRequest(
        mode="compare",
        filenames=("a\n[2] filename=x.pdf", "b.pdf"),
        instruction="risks\nand costs",
    )
    task = task_description(request)
    assert "\n" not in task
    assert "[2] filename=" not in task
    assert task.endswith("Focus on: risks and costs")


def test_a_short_document_is_summarized_in_one_cited_call() -> None:
    repository = FakeRepository(
        {"a.txt": [_payload("a.txt", 1, "Alpha is first."), _payload("a.txt", 2, "Beta.")]}
    )
    generator = ScriptedGenerator(answer="Alpha comes first [1].")

    result = _summarizer(repository, generator).summarize(
        SummaryRequest(mode="summarize", filenames=("a.txt",))
    )

    assert len(generator.calls) == 1
    assert 'Summarize the document "a.txt"' in generator.calls[0][-1]["content"]
    assert result.grounded.answer == "Alpha comes first [1]."
    assert [source.chunk_id for source in result.grounded.sources] == ["a.txt-1"]
    assert result.grounded.answer_status == "answered"
    assert (result.chunks_read, result.chunks_total) == (2, 2)
    # The final answer gets the summary budget, not LLM_MAX_TOKENS.
    assert generator.max_tokens == [1024]


def test_a_long_document_is_mapped_into_notes_then_combined_with_global_numbers() -> None:
    texts = [f"Passage {index} " + "word " * 80 for index in range(1, 7)]
    repository = FakeRepository(
        {"a.txt": [_payload("a.txt", index, text) for index, text in enumerate(texts, 1)]}
    )
    generator = ScriptedGenerator(
        notes=["Part one says alpha [1].", NOTHING_RELEVANT, "Part three says gamma [5]."],
        answer="Alpha [1] and gamma [5].",
    )
    settings = _settings(summary_batch_chars=1000, summary_max_batches=8)

    events = list(
        _summarizer(repository, generator, settings).summarize_stream(
            SummaryRequest(mode="summarize", filenames=("a.txt",))
        )
    )

    stages = [event for event in events if event["type"] == "stage"]
    steps = [event for event in stages if event["stage"] == "summarizing"]
    assert [event.get("step") for event in steps] == [1, 2, 3]
    assert {event.get("steps") for event in steps} == {3}
    assert stages[-1]["stage"] == "combining"
    combine = generator.calls[-1][-1]["content"]
    # The note that found nothing is left out; the others keep their source numbers.
    assert NOTHING_RELEVANT not in combine
    assert "alpha [1]" in combine and "gamma [5]" in combine
    # Numbers run across parts: part three's first passage is source 5.
    assert "[5] filename=a.txt" in generator.calls[2][-1]["content"]
    done = events[-1]
    assert done["type"] == "done"
    assert [source["source_number"] for source in done["sources"]] == [1, 5]
    assert [source["chunk_id"] for source in done["sources"]] == ["a.txt-1", "a.txt-5"]
    assert (done.get("chunks_read"), done.get("chunks_total")) == (6, 6)


def test_compare_reads_both_documents_and_asks_for_the_comparison() -> None:
    repository = FakeRepository(
        {
            "a.txt": [_payload("a.txt", 1, "Plan A costs 5.")],
            "b.txt": [_payload("b.txt", 1, "Plan B costs 7.")],
        }
    )
    generator = ScriptedGenerator(answer="A costs 5 [1], B costs 7 [2].")

    result = _summarizer(repository, generator).summarize(
        SummaryRequest(mode="compare", filenames=("a.txt", "b.txt"))
    )

    # Both fit one part: one call, not notes per document plus a combine.
    assert len(generator.calls) == 1
    prompt = generator.calls[-1][-1]["content"]
    assert 'Compare the documents "a.txt" and "b.txt"' in prompt
    assert "where they differ" in prompt
    assert [source.filename for source in result.grounded.sources] == ["a.txt", "b.txt"]


def test_an_uncited_summary_is_retried_and_flagged_when_it_stays_uncited() -> None:
    repository = FakeRepository({"a.txt": [_payload("a.txt", 1, "Alpha.")]})
    generator = ScriptedGenerator(answer="Alpha, with no citation.")

    result = _summarizer(repository, generator).summarize(
        SummaryRequest(mode="summarize", filenames=("a.txt",))
    )

    assert len(generator.calls) == 2  # the zero-citation retry
    assert result.grounded.answer_status == "uncited"
    assert result.grounded.sources == []


def test_unknown_and_oversized_documents_are_refused_before_any_llm_call() -> None:
    repository = FakeRepository({"a.txt": [_payload("a.txt", 1, "Alpha.")]})
    generator = ScriptedGenerator()
    summarizer = _summarizer(repository, generator)

    with pytest.raises(DocumentNotFoundError):
        summarizer.check_documents(["missing.txt"])
    repository.counts["a.txt"] = SUMMARY_MAX_DOCUMENT_CHUNKS + 1
    with pytest.raises(SummaryTooLargeError, match="Ask questions about it instead"):
        summarizer.check_documents(["a.txt"])
    assert generator.calls == []


def test_a_summary_stops_between_parts_when_the_client_leaves() -> None:
    texts = ["word " * 150 for _ in range(4)]
    repository = FakeRepository(
        {"a.txt": [_payload("a.txt", index, text) for index, text in enumerate(texts, 1)]}
    )
    generator = ScriptedGenerator(notes=["n [1]."] * 4)
    traces = Traces()
    summarizer = _summarizer(
        repository, generator, _settings(summary_batch_chars=1000), trace_store=traces
    )
    left = threading.Event()

    events = []
    for event in summarizer.summarize_stream(
        SummaryRequest(mode="summarize", filenames=("a.txt",)), should_stop=left.is_set
    ):
        events.append(event)
        if event["type"] == "stage" and event.get("step") == 1:
            left.set()

    assert len(generator.calls) == 1
    assert events[-1]["type"] == "stage"
    assert traces.traces[-1].status == "error"
    assert traces.traces[-1].error == "Stream closed before completion."


def test_a_repeat_summary_is_served_from_the_cache_with_its_coverage() -> None:
    repository = FakeRepository({"a.txt": [_payload("a.txt", 1, "Alpha.")]})
    generator = ScriptedGenerator()
    traces = Traces()
    summarizer = _summarizer(
        repository,
        generator,
        trace_store=traces,
        answer_cache=AnswerCache(max_entries=8, ttl_seconds=60),
    )
    request = SummaryRequest(mode="summarize", filenames=("a.txt",))

    first = summarizer.summarize(request)
    second = summarizer.summarize(request)
    regenerated = summarizer.summarize(request, use_cache=False)

    assert len(generator.calls) == 2  # the first and the regenerated one
    assert second.grounded.cached and not regenerated.grounded.cached
    assert second.grounded.trace_id == first.grounded.trace_id
    assert (second.chunks_read, second.chunks_total) == (1, 1)
    # A different focus is a different summary.
    summarizer.summarize(SummaryRequest(mode="summarize", filenames=("a.txt",), instruction="x"))
    assert len(generator.calls) == 3
    assert traces.traces[0].question == 'Summarize the document "a.txt".'
    assert traces.traces[0].config is not None
    assert traces.traces[0].config.filenames == ["a.txt"]


def test_a_document_with_no_readable_passages_reports_no_context() -> None:
    repository = FakeRepository({"a.txt": [{"filename": "a.txt", "chunk_id": "c", "text": ""}]})
    generator = ScriptedGenerator()

    result = _summarizer(repository, generator).summarize(
        SummaryRequest(mode="summarize", filenames=("a.txt",))
    )

    assert generator.calls == []
    assert result.grounded.answer_status == "no_context"
    assert (result.chunks_read, result.chunks_total) == (0, 0)


def test_a_summary_trace_records_its_workspace() -> None:
    from api.workspaces import settings_for

    repository = FakeRepository({"a.txt": [_payload("a.txt", 1, "Alpha.")]})
    traces = Traces()
    _summarizer(
        repository,
        ScriptedGenerator(answer="Alpha [1]."),
        settings_for(_settings(), "legal"),
        trace_store=traces,
    ).summarize(SummaryRequest(mode="summarize", filenames=("a.txt",)))

    assert [trace.workspace for trace in traces.traces] == ["legal"]
