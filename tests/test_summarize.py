from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence
from typing import Any, cast

import pytest

import api.llm as llm_module
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


def test_a_summary_stream_left_as_done_arrives_is_still_cached() -> None:
    repository = FakeRepository({"a.txt": [_payload("a.txt", 1, "Alpha.")]})
    generator = ScriptedGenerator()
    summarizer = _summarizer(
        repository, generator, answer_cache=AnswerCache(max_entries=8, ttl_seconds=60)
    )
    request = SummaryRequest(mode="summarize", filenames=("a.txt",))

    stream = summarizer.summarize_stream(request)
    for event in stream:
        if event["type"] == "done":
            break
    # The client went away with the `done` frame in hand: the generator is closed at
    # that yield, and the cache entry was written before it went out.
    stream.close()

    replay = list(summarizer.summarize_stream(request))
    assert len(generator.calls) == 1
    assert replay[-1]["type"] == "done" and replay[-1].get("cached") is True


def test_cjk_passages_are_packed_by_tokens_not_characters() -> None:
    from api.reranking import RerankedChunk
    from api.summarize import _pack

    def chunk(n: int, text: str) -> RerankedChunk:
        return RerankedChunk(
            point_id=f"p{n}",
            filename="a.txt",
            page=1,
            section="S",
            chunk_id=f"c{n}",
            text=text,
            retrieval_score=0.0,
            rerank_score=0.0,
        )

    latin = [chunk(n, "x" * 300) for n in range(4)]
    cjk = [chunk(n, "休" * 300) for n in range(4)]
    # 300 CJK characters weigh three times 300 Latin ones: more, smaller parts.
    assert len(_pack(cjk, 1000)) > len(_pack(latin, 1000))
    # A single CJK passage over the budget is cut to fit it.
    (only,) = _pack([chunk(9, "休" * 2000)], 1000)
    assert len(only[0].text) < 1000 // 3 + 1


def test_a_brief_covers_several_documents_by_theme() -> None:
    repository = FakeRepository(
        {name: [_payload(name, 1, f"{name} text.")] for name in ("a.txt", "b.txt", "c.txt")}
    )
    generator = ScriptedGenerator()
    request = SummaryRequest(mode="brief", filenames=("a.txt", "b.txt", "c.txt"))

    result = _summarizer(repository, generator).summarize(request)

    prompt = generator.calls[-1][-1]["content"]
    assert 'covering the documents "a.txt", "b.txt" and "c.txt"' in prompt
    assert "by theme" in prompt
    assert all(f"{name} text." in prompt for name in ("a.txt", "b.txt", "c.txt"))
    assert result.chunks_read == 3


def test_a_brief_past_the_document_limit_is_refused() -> None:
    names = tuple(f"d{n}.txt" for n in range(6))
    repository = FakeRepository({name: [_payload(name, 1, "t")] for name in names})
    summarizer = _summarizer(repository, ScriptedGenerator(), _settings(summary_max_documents=5))
    with pytest.raises(SummaryTooLargeError, match="at most 5"):
        summarizer.check_documents(names)


class SlowNotes(ScriptedGenerator):
    """Notes that finish out of order, each reporting token usage like LiteLLM does."""

    def __init__(self) -> None:
        super().__init__(answer="All of it [1].")
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        content = messages[-1]["content"]
        if "Excerpts from" not in content:
            return super().complete(messages, settings)
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        part = int(content.split("(part ")[1].split(" of")[0])
        time.sleep(0.02 * max(0, 8 - part))  # earlier parts finish later
        llm_module._count_usage({"usage": {"prompt_tokens": 10, "completion_tokens": 1}}, settings)
        with self.lock:
            self.active -= 1
        return f"Notes on part {part} [{part}]."


def _long_document(parts: int) -> FakeRepository:
    return FakeRepository(
        {"long.txt": [_payload("long.txt", n, "w" * 1200) for n in range(1, parts * 3 + 1)]}
    )


def test_hosted_parts_run_side_by_side_and_keep_their_order_and_usage() -> None:
    generator = SlowNotes()
    settings = _settings(
        llm_provider="openai",
        openai_api_key="k",
        summary_batch_chars=3000,
        summary_parallel_parts=4,
    )
    request = SummaryRequest(mode="summarize", filenames=("long.txt",))

    events = list(_summarizer(_long_document(4), generator, settings).summarize_stream(request))

    assert generator.peak > 1
    combine = generator.calls[-1][-1]["content"]
    positions = [combine.index(f"Notes on part {n}") for n in range(1, 4)]
    assert positions == sorted(positions)  # in document order, whatever finished first
    steps = [e["step"] for e in events if e["type"] == "stage" and e["stage"] == "summarizing"]
    assert steps == list(range(1, len(steps) + 1)) and len(steps) >= 3
    done = next(e for e in events if e["type"] == "done")
    assert done["usage"]["prompt_tokens"] >= 40  # every part's tokens counted


def test_ollama_parts_run_one_at_a_time() -> None:
    generator = SlowNotes()
    settings = _settings(llm_provider="ollama", summary_batch_chars=3000, summary_parallel_parts=4)
    request = SummaryRequest(mode="summarize", filenames=("long.txt",))
    list(_summarizer(_long_document(3), generator, settings).summarize_stream(request))
    assert generator.peak == 1


class NotesRecorder(SlowNotes):
    def __init__(self) -> None:
        super().__init__()
        self.notes_prompts: list[str] = []

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        if "Excerpts from" in messages[-1]["content"]:
            self.notes_prompts.append(messages[-1]["content"])
        return super().complete(messages, settings)


def test_notes_for_several_documents_say_the_others_are_read_separately() -> None:
    # A part shows one document while a compare/brief task names all of them; told
    # nothing more, gpt-4o-mini answered NOTHING RELEVANT for every part.
    marker = "the others are read separately"
    settings = _settings(llm_provider="ollama", summary_batch_chars=3000)
    repository = FakeRepository(
        {name: [_payload(name, n, "w" * 1200) for n in range(1, 7)] for name in ("a.txt", "b.txt")}
    )
    generator = NotesRecorder()
    request = SummaryRequest(mode="compare", filenames=("a.txt", "b.txt"))
    list(_summarizer(repository, generator, settings).summarize_stream(request))
    assert len(generator.notes_prompts) >= 2
    assert all(marker in prompt for prompt in generator.notes_prompts)

    generator = NotesRecorder()
    request = SummaryRequest(mode="summarize", filenames=("long.txt",))
    list(_summarizer(_long_document(3), generator, settings).summarize_stream(request))
    assert generator.notes_prompts
    assert not any(marker in prompt for prompt in generator.notes_prompts)


@pytest.mark.parametrize("mode", ["backfill", "auto"])
def test_an_uncited_local_summary_is_backfilled_without_a_retry_or_its_stage(mode: str) -> None:
    repository = FakeRepository(
        {"a.txt": [_payload("a.txt", 1, "The office opens at nine on weekdays.")]}
    )
    generator = ScriptedGenerator(answer="The office opens at nine on weekdays.")
    settings = _settings(llm_provider="ollama", citation_retry_mode=mode)

    events = list(
        _summarizer(repository, generator, settings).summarize_stream(
            SummaryRequest(mode="summarize", filenames=("a.txt",))
        )
    )

    assert len(generator.calls) == 1
    assert "verifying_citations" not in [e.get("stage") for e in events if e["type"] == "stage"]
    done = events[-1]
    assert done["answer"] == "The office opens at nine on weekdays [1]."
    assert done["answer_status"] == "answered"


def test_a_summary_waits_for_the_local_model_between_parts_and_says_so() -> None:
    from api.llm import GenerationSlot

    texts = ["word " * 150 for _ in range(3)]
    repository = FakeRepository(
        {"a.txt": [_payload("a.txt", index, text) for index, text in enumerate(texts, 1)]}
    )
    settings = _settings(llm_provider="ollama", summary_batch_chars=1000)
    generator = ScriptedGenerator(notes=["n [1]."] * 3)
    holder = GenerationSlot(settings)
    assert holder.acquire()
    events: list[Any] = []
    waiting = threading.Event()

    def run() -> None:
        for event in _summarizer(repository, generator, settings).summarize_stream(
            SummaryRequest(mode="summarize", filenames=("a.txt",))
        ):
            events.append(event)
            if event.get("stage") == "waiting_for_model":
                waiting.set()

    thread = threading.Thread(target=run)
    thread.start()
    assert waiting.wait(10)
    assert generator.calls == []
    holder.release()
    thread.join(10)
    assert events[-1]["type"] == "done"
    after = GenerationSlot(settings)
    assert after.try_acquire()
    after.release()


def test_a_summary_whose_reader_left_while_waiting_calls_no_model() -> None:
    from api.llm import GenerationSlot

    repository = FakeRepository({"a.txt": [_payload("a.txt", 1, "Alpha.")]})
    settings = _settings(llm_provider="ollama")
    generator = ScriptedGenerator()
    holder = GenerationSlot(settings)
    assert holder.acquire()
    gone = threading.Event()
    events: list[Any] = []
    thread = threading.Thread(
        target=lambda: events.extend(
            _summarizer(repository, generator, settings).summarize_stream(
                SummaryRequest(mode="summarize", filenames=("a.txt",)), should_stop=gone.is_set
            )
        )
    )
    thread.start()
    time.sleep(0.5)
    gone.set()
    thread.join(10)
    assert not thread.is_alive()
    assert generator.calls == []
    assert events[-1].get("stage") == "waiting_for_model"
    holder.release()


def test_combined_notes_are_cut_to_their_weighted_share_of_the_budget() -> None:
    from api.prompting import weighted_length
    from api.summarize import build_combine_messages

    notes = [("a.txt", n, 20, "要点" * 300 + f" [{n}]") for n in range(1, 21)]
    prompt = build_combine_messages("Summarize", "summarize", notes, 2000)[-1]["content"]
    bodies = [
        block.split(" of 20:\n", 1)[1] for block in prompt.split("\n\n") if " of 20:" in block
    ]
    assert len(bodies) == 20
    # 100 weighted characters each (plus the " …" marking the cut), never 400 raw ones.
    assert all(weighted_length(body.removesuffix(" …")) <= 100 for body in bodies)


def test_a_backfilled_summary_says_so_in_its_answer_trace_and_cache() -> None:
    repository = FakeRepository(
        {"a.txt": [_payload("a.txt", 1, "The office opens at nine on weekdays.")]}
    )
    generator = ScriptedGenerator(answer="The office opens at nine on weekdays.")
    traces = Traces()
    summarizer = _summarizer(
        repository,
        generator,
        _settings(llm_provider="ollama", citation_retry_mode="backfill"),
        trace_store=traces,
        answer_cache=AnswerCache(max_entries=8, ttl_seconds=600),
    )
    request = SummaryRequest(mode="summarize", filenames=("a.txt",))

    first = summarizer.summarize(request)
    assert first.grounded.citation_backfilled is True
    assert traces.traces[-1].citation_backfilled is True
    assert traces.traces[-1].citation_retry_used is False
    again = list(summarizer.summarize_stream(request))
    assert again[-1]["cached"] is True and again[-1]["citation_backfilled"] is True


def test_an_identical_summary_waits_for_the_running_one() -> None:
    import threading

    gate = threading.Event()
    entered = threading.Event()

    class GatedGenerator(ScriptedGenerator):
        def _reply(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
            entered.set()
            assert gate.wait(10)
            return super()._reply(messages, settings)

    repository = FakeRepository({"a.txt": [_payload("a.txt", 1, "Alpha.")]})
    generator = GatedGenerator()
    summarizer = _summarizer(
        repository, generator, answer_cache=AnswerCache(max_entries=8, ttl_seconds=60)
    )
    request = SummaryRequest(mode="summarize", filenames=("a.txt",))
    results: list[Any] = []
    leader = threading.Thread(target=lambda: results.append(summarizer.summarize(request)))
    leader.start()
    assert entered.wait(5)
    follower_events = summarizer.summarize_stream(request)
    assert next(follower_events) == {"type": "stage", "stage": "joining_identical_question"}
    gate.set()
    rest = list(follower_events)
    leader.join(10)

    done = next(event for event in rest if event["type"] == "done")
    assert done["cached"] is True and done["chunks_read"] == 1
    assert len(generator.calls) == 1
