"""Pass 18: answer-path fixes found in review."""

from __future__ import annotations

from collections.abc import Sequence

from api.generation import ChatMessage
from api.settings import AppSettings
from tests.factories import make_test_settings
from tests.test_pass17_answers import _Counting, _done, _pipeline, _retrieved, _scored
from tests.test_pipeline import FakeGenerator


def test_a_bracketed_number_that_is_no_source_stays_in_the_answer_and_cites_nothing() -> None:
    from api.citations import cited_numbers, needs_citation_retry

    # One source: "[12]" is the document's own reference, quoted.
    quoted = _pipeline(
        FakeGenerator("Alpha beta, as shown in [12] and matrix[7]."),
        followups_enabled=False,
        citation_retry_mode="off",
    )
    answer = quoted.answer("alpha")
    assert answer.answer == "Alpha beta, as shown in [12] and matrix[7]."
    assert answer.answer_status == "uncited" and answer.sources == []
    assert _done(quoted, "alpha beta")["answer"] == answer.answer

    # Such an answer is still one to add citations to; "[0]" names no source either.
    assert needs_citation_retry("Thirty days [7].", 4)
    assert not needs_citation_retry("Thirty days [2] [9].", 4)
    assert cited_numbers("Zero [0] and a range [0-2].", 4) == {1, 2}


def test_a_dead_marker_is_backfilled_and_a_zero_marker_supports_nothing() -> None:
    from api.citations import unsupported_claims
    from api.reranking import RerankedChunk

    filled = _pipeline(
        FakeGenerator("Alpha beta is the intro [9]."),
        followups_enabled=False,
        citation_retry_mode="backfill",
    ).answer("alpha")
    assert filled.citation_backfilled and [s.source_number for s in filled.sources] == [1]

    chunks = [
        RerankedChunk("p1", "a.md", 1, "S", "c1", "Alpha beta is the intro.", 0.5, 0.9),
        RerankedChunk("p2", "a.md", 1, "S", "c2", "Gamma delta epsilon zeta.", 0.5, 0.8),
    ]
    # "[0]" used to be read as the last source (chunks[-1]) and counted as supported.
    assert unsupported_claims("Gamma delta epsilon zeta apply [0].", chunks) == 0
    assert unsupported_claims("Gamma delta epsilon zeta apply [1].", chunks) == 1


def test_regenerate_replaces_the_reply_kept_for_its_prompt() -> None:
    from api.corpus import bump_corpus_generation

    class _Changing(_Counting):
        def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
            self.calls += 1
            return f"Alpha is documented, take {self.calls} [1]."

    generator = _Changing("")
    pipeline = _pipeline(generator, followups_enabled=False, prompt_cache_size=8)
    first = pipeline.answer("alpha")
    regenerated = pipeline.answer("alpha", use_cache=False)
    assert regenerated.answer != first.answer and not regenerated.prompt_cached

    # The answer cache misses (something unrelated changed): the prompt cache now holds
    # the regenerated reply, not the one the reader turned down.
    bump_corpus_generation()
    again = pipeline.answer("alpha")
    assert again.prompt_cached and again.answer == regenerated.answer


def test_explain_judges_a_file_by_the_passage_that_got_furthest() -> None:
    from api.explain import explain
    from api.pipeline import RetrievalPhase

    settings = make_test_settings(rerank_min_score=0.3, rerank_top_k=2, rerank_candidates=3)
    # b.md's first fused passage scores under the bar; its second is returned.
    fused = [_retrieved("b1", "b.md"), _retrieved("a1"), _retrieved("b2", "b.md")]
    scored = [_scored("a1", 0.9), _scored("b2", 0.7, "b.md"), _scored("b1", 0.1, "b.md")]
    phase = RetrievalPhase(
        context_chunks=scored[:2],
        fused_candidates=fused,
        scored_chunks=scored,
        embed_ms=0,
        search_ms=0,
        rerank_ms=0,
    )
    found = explain(phase, settings, filename="b.md")
    assert (found.outcome, found.chunk_id, found.returned_rank) == ("returned", "b2", 2)

    missing = explain(phase, settings, chunk_id="gone", narrowed=True)
    assert missing.outcome == "not_retrieved" and "limited" in missing.detail
    assert "limited" not in explain(phase, settings, chunk_id="gone").detail


def test_clip_text_survives_a_head_with_no_word_in_it() -> None:
    from api.prompting import clip_text

    assert clip_text("   " * 20 + "word", 10) == "…"
    assert clip_text("alpha beta gamma", 12) == "alpha beta…"
    assert clip_text("abcdefghijklmnop", 5) == "abcde…"


def test_a_summary_whose_prompt_was_cut_says_so_and_is_not_cached() -> None:
    import api.llm as llm
    from api.answer_cache import AnswerCache
    from api.summarize import SummaryRequest
    from tests.test_summarize import FakeRepository, ScriptedGenerator, _payload, _summarizer

    class _CutPrompt(ScriptedGenerator):
        def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
            reply = super().complete(messages, settings)
            llm.carry_prompt_cut(True)  # what the usage of a cut prompt notes
            return reply

    repository = FakeRepository({"a.txt": [_payload("a.txt", 1, "Alpha is first.")]})
    generator = _CutPrompt(answer="Alpha comes first [1].")
    summarizer = _summarizer(
        repository, generator, answer_cache=AnswerCache(max_entries=8, ttl_seconds=60)
    )
    request = SummaryRequest(mode="summarize", filenames=("a.txt",))

    first = summarizer.summarize(request)
    again = summarizer.summarize(request)

    assert first.grounded.prompt_cut and again.grounded.prompt_cut
    assert not again.grounded.cached and len(generator.calls) == 2


def test_a_sync_summary_stops_between_parts_when_its_client_has_gone() -> None:
    import threading

    import pytest

    from api.pipeline import QuestionAbandoned
    from api.summarize import SummaryRequest
    from tests.test_summarize import (
        FakeRepository,
        ScriptedGenerator,
        _payload,
        _settings,
        _summarizer,
    )

    left = threading.Event()

    class _LeavesAfterOne(ScriptedGenerator):
        def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
            left.set()
            return super().complete(messages, settings)

    texts = ["word " * 150 for _ in range(4)]
    repository = FakeRepository(
        {"a.txt": [_payload("a.txt", index, text) for index, text in enumerate(texts, 1)]}
    )
    generator = _LeavesAfterOne(notes=["n [1]."] * 4)
    summarizer = _summarizer(repository, generator, _settings(summary_batch_chars=1000))

    with pytest.raises(QuestionAbandoned):
        summarizer.summarize(
            SummaryRequest(mode="summarize", filenames=("a.txt",)), should_stop=left.is_set
        )
    assert len(generator.calls) == 1


def test_only_a_questions_answer_can_be_continued_and_a_continuation_is_not_logged() -> None:
    import dataclasses

    import pytest

    from api.pipeline import ContinueRefused
    from api.query_log import QueryLogSink

    class _Truncating(FakeGenerator):
        def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
            from api.llm import reset_finish_reason

            reply = super().complete(messages, settings)
            reset_finish_reason("length")
            return reply

    class _Traces:
        def __init__(self) -> None:
            self.traces: list = []  # type: ignore[type-arg]

        def add(self, trace) -> None:  # type: ignore[no-untyped-def]
            self.traces.append(trace)

    pipeline = _pipeline(_Truncating("Alpha is documented [1] and"), followups_enabled=False)
    traces = _Traces()
    pipeline._trace_store = traces  # type: ignore[assignment]
    answer = pipeline.answer("alpha")
    trace = traces.traces[-1]
    assert answer.truncated and trace.kind == "question"
    pipeline.continuable(trace.trace_id, trace, pipeline._settings)

    summary = dataclasses.replace(trace, kind="summary")
    with pytest.raises(ContinueRefused) as refused:
        pipeline.continuable(summary.trace_id, summary, pipeline._settings)
    assert refused.value.status_code == 409

    class _Store:
        def __init__(self) -> None:
            self.rows: list = []  # type: ignore[type-arg]

        def add(self, row) -> None:  # type: ignore[no-untyped-def]
            self.rows.append(row)

    store = _Store()
    sink = QueryLogSink(store)  # type: ignore[arg-type]
    sink.add(trace)
    sink.add(dataclasses.replace(trace, trace_id="t2", continues=trace.trace_id))
    assert len(store.rows) == 1


def test_answer_rows_say_when_an_answer_was_cut_and_what_it_cost() -> None:
    from eval.answer_eval import AnswerRow, answer_metrics
    from eval.harness import RetrievalExample

    rows = [
        AnswerRow("q1", "a", [""], None, truncated=True, usage=(100, 20), cost_usd=0.001),
        AnswerRow("q2", "a", [""], None, prompt_cut=True, top_rerank_score=0.42),
    ]
    examples = [RetrievalExample(q, None, frozenset(), frozenset()) for q in ("q1", "q2")]
    metrics = answer_metrics(examples, rows)
    assert metrics["truncated_rate"] == 0.5 and metrics["prompt_cut_rate"] == 0.5
    back = [AnswerRow.from_dict(row.to_dict()) for row in rows]
    assert back[0].usage == (100, 20) and back[0].cost_usd == 0.001 and back[0].truncated
    assert back[1].prompt_cut and back[1].top_rerank_score == 0.42


def test_an_answer_run_file_carries_its_settings_and_scores(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    from eval.compare import load_run

    rows = [{"question": "q", "answer": "a", "contexts": [""], "answer_status": "answered"}]
    new = tmp_path / "new.json"
    new.write_text(json.dumps({"fingerprint": {}, "metrics": {}, "examples": rows}))
    old = tmp_path / "old.json"
    old.write_text(json.dumps(rows))
    assert load_run(new).kind == "answer" and load_run(old).kind == "answer"
    assert load_run(new).data == load_run(old).data


def test_auto_continue_finishes_a_cut_off_answer_in_the_same_stream() -> None:
    import json as _json
    from collections.abc import Sequence as _Seq

    from fastapi.testclient import TestClient

    from api.dependencies import clear_dependency_caches, get_generator
    from api.generation import ChatMessage as _Msg
    from api.llm import reset_finish_reason
    from api.settings import AppSettings as _Settings
    from tests.routes_http.support import hermetic_app, upload_and_wait
    from tests.routes_http.support import make_settings as route_settings

    class _CutTwice:
        def __init__(self, parts: list[str]) -> None:
            self.parts = parts
            self.calls: list[list[_Msg]] = []

        def complete(self, messages: _Seq[_Msg], settings: _Settings) -> str:
            return "unused"

        def stream(self, messages: _Seq[_Msg], settings: _Settings):  # type: ignore[no-untyped-def]
            self.calls.append(list(messages))
            part = self.parts[len(self.calls) - 1]
            yield part
            reset_finish_reason("length" if len(self.calls) < len(self.parts) else "stop")

    def ask(generator: _CutTwice, **settings: object) -> tuple[list[dict[str, object]], _CutTwice]:
        clear_dependency_caches()
        app = hermetic_app(
            route_settings(citation_retry_enabled=False, **settings),
            __import__("qdrant_client").QdrantClient(":memory:"),
        )
        app.dependency_overrides[get_generator] = lambda: generator
        with TestClient(app) as client:
            upload_and_wait(client, "guide.txt", b"Intro\nalpha beta gamma")
            text = client.post("/questions/stream", json={"question": "alpha?"}).text
        clear_dependency_caches()
        events = [_json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]
        return events, generator

    parts = ["Alpha is documented [1] and", " beta follows [1] and", " gamma ends it [1]."]
    events, generator = ask(_CutTwice(parts), auto_continue_max=2)
    done = next(e for e in events if e["type"] == "done")
    assert done["answer"] == "Alpha is documented [1] and beta follows [1] and gamma ends it [1]."
    assert done["truncated"] is False
    assert [e.get("stage") for e in events if e["type"] == "stage"].count("continuing") == 2
    streamed = "".join(str(e["text"]) for e in events if e["type"] == "delta")
    assert streamed == done["answer"]
    # Ollama continues a trailing assistant message: the answer so far, as sent.
    assert generator.calls[2][-1]["content"] == "Alpha is documented [1] and beta follows [1] and"

    # Off (the default), or past its limit: the reader is left to press Continue.
    events, _ = ask(_CutTwice(parts))
    assert next(e for e in events if e["type"] == "done")["truncated"] is True
    events, _ = ask(_CutTwice(parts), auto_continue_max=1)
    done = next(e for e in events if e["type"] == "done")
    assert done["truncated"] is True and done["answer"].endswith("beta follows [1] and")

    # Asked in words (hosted OpenAI): a model that starts over has the repeat dropped.
    restart = [
        "Alpha is documented here in full [1] and",
        "Alpha is documented here in full [1] and beta [1].",
    ]
    events, generator = ask(
        _CutTwice(restart), auto_continue_max=1, llm_provider="openai", openai_api_key="k"
    )
    done = next((e for e in events if e["type"] == "done"), events[-1])
    assert done["answer"] == "Alpha is documented here in full [1] and beta [1].", events


def test_a_padded_table_read_whole_is_not_taken_for_a_cut_prompt() -> None:
    # Measured on Ollama (qwen2.5, uncut): a table of "------------" cells reported 3921
    # tokens against a character estimate of 9226, space-padded cells 1021 against 5626;
    # at num_ctx 2048 both reported 1026 — a real cut, still flagged.
    from api.llm import LiteLLMGenerator, begin_usage, was_prompt_cut
    from tests.factories import make_test_settings

    def client(reported: int):  # type: ignore[no-untyped-def]
        def call(**kwargs: object) -> object:
            return {
                "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": reported, "completion_tokens": 1},
            }

        return call

    settings = make_test_settings(
        llm_provider="openai_compatible",
        openai_compatible_base_url="http://127.0.0.1:9/v1",
        openai_compatible_model="m",
    )
    dashes = "\n".join("| " + " | ".join(["-" * 12] * 6) + " |" for _ in range(300))
    spaces = "\n".join("|" + " " * 40 + "|" + " " * 40 + "|" for _ in range(200))
    for body, uncut in ((dashes, 3921), (spaces, 1021)):
        messages = [{"role": "user", "content": body}]
        begin_usage()
        LiteLLMGenerator(settings, completion_client=client(uncut)).complete(messages, settings)
        assert not was_prompt_cut()
    begin_usage()
    messages = [{"role": "user", "content": dashes}]
    LiteLLMGenerator(settings, completion_client=client(1026)).complete(messages, settings)
    assert was_prompt_cut()


def test_a_continuation_that_repeats_the_last_words_is_joined_once() -> None:
    # gpt-5-mini, continuing by instruction (AUTO_CONTINUE_MAX, live): "… It also" went
    # on "It also states …", and "… expense policies (" went on "policies (no …".
    from api.pipeline import drop_restart, join_continuation

    def joined(before: str, after: str) -> str:
        return join_continuation(before, drop_restart(before, after))

    assert (
        joined("employees must follow it. It also", "It also states that it is reviewed.")
        == "employees must follow it. It also states that it is reviewed."
    )
    assert (
        joined("about expense policies (", "policies (no expense rules are given).")
        == "about expense policies (no expense rules are given)."
    )
    # A list item written again from its marker, a lone marker, one longer word.
    item = "- No numerical rules or per-diem amounts are present in the excerpt"
    assert joined(f"Travel\n{item}", f"{item} [1].\n\nLeave") == f"Travel\n{item} [1].\n\nLeave"
    assert (
        joined("rules [3].\n- ", "- Review cadence: yearly [1].")
        == "rules [3].\n- Review cadence: yearly [1]."
    )
    assert joined("repaired within", "within two days [4].") == "repaired within two days [4]."
    # One repeated short word is left alone: "the" may well begin the next phrase.
    assert joined("the end of the", "the policy says") == "the end of the the policy says"
    assert joined("Leave is ten", "days per year.") == "Leave is ten days per year."
