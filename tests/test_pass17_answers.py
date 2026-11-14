"""Pass 17: answer-path fixes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient

from api.answer_cache import AnswerCache
from api.generation import ChatMessage
from api.llm import GenerationError
from api.pipeline import IngestService, RagPipeline
from api.repository import VectorRepository
from api.settings import AppSettings
from tests.factories import make_test_settings
from tests.test_pipeline import (
    FakeGenerator,
    FakeReranker,
    StaticEmbeddingProvider,
    make_embedding,
    make_settings,
)


class _FollowupsFail(FakeGenerator):
    """Answers; the follow-up call fails."""

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        if "follow-up" in messages[0]["content"].lower():
            raise GenerationError("provider down")
        return super().complete(messages, settings)


def _pipeline(generator: FakeGenerator, **overrides: object) -> RagPipeline:
    settings = make_settings(**{"followups_enabled": True, **overrides})
    repository = VectorRepository(QdrantClient(":memory:"))
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )
    return RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=generator,
        settings=settings,
        answer_cache=AnswerCache(max_entries=8, ttl_seconds=60),
    )


def _done(pipeline: RagPipeline, question: str) -> dict[str, object]:
    events = list(pipeline.answer_stream(question))
    return dict(next(event for event in events if event["type"] == "done"))


def test_a_streamed_answer_whose_followups_failed_is_not_kept_like_the_sync_one() -> None:
    streamed = _pipeline(_FollowupsFail("Alpha is documented [1]."))
    assert not _done(streamed, "alpha").get("cached")
    assert not _done(streamed, "alpha").get("cached")

    synced = _pipeline(_FollowupsFail("Alpha is documented [1]."))
    assert not synced.answer("alpha").cached
    assert not synced.answer("alpha").cached


def test_discard_leaves_a_newer_answer_stored_under_the_same_key() -> None:
    from api.answer_cache import CachedAnswer

    cache = AnswerCache(max_entries=4, ttl_seconds=60)
    first = CachedAnswer.__new__(CachedAnswer)
    newer = CachedAnswer.__new__(CachedAnswer)
    key = ("k",)
    cache.put(key, first)
    cache.put(key, newer)
    assert cache.discard(key, first) is False
    assert cache.get(key) is newer
    assert cache.discard(key, newer) is True
    assert cache.get(key) is None


def test_a_sync_answer_stops_before_generation_when_the_client_has_gone() -> None:
    from api.pipeline import QuestionAbandoned

    calls: list[str] = []

    class _Counting(FakeGenerator):
        def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
            calls.append("complete")
            return super().complete(messages, settings)

    pipeline = _pipeline(_Counting("Alpha is documented [1]."))
    import pytest

    with pytest.raises(QuestionAbandoned):
        pipeline.answer("alpha", should_stop=lambda: True)
    assert calls == []
    assert pipeline.answer("alpha").answer == "Alpha is documented [1]."


def test_a_blank_question_is_a_400_before_a_slot_on_both_routes() -> None:
    from fastapi.testclient import TestClient

    from api.dependencies import get_question_slots
    from tests.routes_http.support import hermetic_app
    from tests.routes_http.support import make_settings as route_settings

    class _NoSlots:
        def try_acquire(self, kind: str = "question") -> None:
            raise AssertionError("a slot was taken for a blank question")

    app = hermetic_app(route_settings(), QdrantClient(":memory:"))
    app.dependency_overrides[get_question_slots] = lambda: _NoSlots()
    with TestClient(app) as client:
        for path in ("/questions", "/questions/stream"):
            response = client.post(path, json={"question": "  \n "})
            assert response.status_code == 400
            assert "Query text is required" in response.json()["detail"]


def test_stray_markers_go_and_real_or_quoted_ones_stay() -> None:
    from api.citations import strip_stray_markers

    assert strip_stray_markers("Thirty days [7].", 4) == "Thirty days."
    assert strip_stray_markers("Thirty days [2] [9].", 4) == "Thirty days [2]."
    assert strip_stray_markers("Thirty days [3-9].", 4) == "Thirty days [3-9]."
    assert strip_stray_markers("Per the 2019 rules [2019].", 4) == "Per the 2019 rules [2019]."
    assert strip_stray_markers("No sources [1].", 0) == "No sources."
    assert strip_stray_markers("Alpha is documented [1].", 4) == "Alpha is documented [1]."


def test_an_openai_compatible_stream_asks_for_usage_and_drops_it_once_refused() -> None:
    from api.llm import LiteLLMGenerator

    calls: list[dict[str, object]] = []

    def client(**kwargs: object) -> object:
        calls.append(kwargs)
        if "stream_options" in kwargs:
            raise ValueError("unknown field: stream_options")
        return iter(())

    settings = make_test_settings(
        llm_provider="openai_compatible",
        openai_compatible_base_url="http://127.0.0.1:9/v1",
        openai_compatible_model="m",
    )
    generator = LiteLLMGenerator(settings, completion_client=client)
    list(generator.stream([{"role": "user", "content": "q"}], settings))
    list(generator.stream([{"role": "user", "content": "q"}], settings))
    assert ["stream_options" in call for call in calls] == [True, False, False]


def test_a_long_standalone_cjk_question_skips_the_condense_call() -> None:
    from api.prompting import needs_condense

    assert not needs_condense("员工每年可以享受多少天的带薪年假")
    assert not needs_condense("年次休暇は何日取得できますか")
    assert needs_condense("它是多少")
    assert needs_condense("それはいつ")
    assert needs_condense("多少")
    assert needs_condense("What about it?")


def test_insufficiency_needs_the_sources_named_for_its_everyday_words() -> None:
    from api.citations import answer_status, states_insufficiency

    assert not states_insufficiency("Employees with insufficient leave can take unpaid leave.")
    assert not states_insufficiency("The form does not contain a signature field.")
    assert states_insufficiency("The provided documents do not contain enough information.")
    assert states_insufficiency("The context does not contain the refund rules.")
    assert states_insufficiency("There is insufficient information in the sources.")
    assert states_insufficiency("I cannot answer that from the documents.")
    assert states_insufficiency("Los documentos no contienen esa información.")
    assert answer_status("The form does not contain a signature field.", 0, 3) == "uncited"


class _CountsCondense(FakeGenerator):
    """Rewrites follow-ups (counted) and answers everything else."""

    def __init__(self, answer: str) -> None:
        super().__init__(answer)
        self.condense_calls = 0

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        if "standalone search query" in messages[0]["content"]:
            self.condense_calls += 1
            return "What is alpha beta?"
        return super().complete(messages, settings)


def test_a_follow_up_asked_again_reuses_its_rewrite_unless_regenerated() -> None:
    generator = _CountsCondense("Alpha is documented [1].")
    pipeline = _pipeline(generator, followups_enabled=False)
    history: list[ChatMessage] = [
        {"role": "user", "content": "Tell me about alpha beta"},
        {"role": "assistant", "content": "Alpha beta is described in the guide."},
    ]
    pipeline.answer("And what about it?", history=history, use_cache=False)
    assert generator.condense_calls == 1
    # Asked again: the rewrite is remembered — no call, no "condensing" stage.
    events = list(pipeline.answer_stream("And what about it?", history=history))
    assert generator.condense_calls == 1
    assert not any(e["type"] == "stage" and e.get("stage") == "condensing" for e in events)
    # Regenerate asks the model again.
    pipeline.answer("And what about it?", history=history, use_cache=False)
    assert generator.condense_calls == 2


def test_citing_sentences_no_source_supports_are_counted_in_both_modes() -> None:
    unsupported = _pipeline(
        FakeGenerator("Gamma delta epsilon zeta apply here [1]."), followups_enabled=False
    )
    assert unsupported.answer("alpha").unsupported_claims == 1
    assert _done(unsupported, "alpha beta")["unsupported_claims"] == 1

    supported = _pipeline(FakeGenerator("Alpha beta is the intro [1]."), followups_enabled=False)
    assert supported.answer("alpha").unsupported_claims == 0
    assert _done(supported, "alpha beta").get("unsupported_claims", 0) == 0


class _Counting(FakeGenerator):
    def __init__(self, answer: str) -> None:
        super().__init__(answer)
        self.calls = 0

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        self.calls += 1
        return super().complete(messages, settings)

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings):  # type: ignore[no-untyped-def]
        self.calls += 1
        yield from super().stream(messages, settings)


def test_the_prompt_cache_saves_the_call_when_retrieval_finds_the_same_passages() -> None:
    from api.corpus import bump_corpus_generation

    generator = _Counting("Alpha is documented [1].")
    pipeline = _pipeline(generator, followups_enabled=False, prompt_cache_size=8)
    first = pipeline.answer("alpha")
    assert generator.calls == 1 and not first.prompt_cached

    # Something unrelated changed: the answer cache can't tell, so it misses; the same
    # passages make the same prompt, whose reply is reused.
    bump_corpus_generation()
    again = pipeline.answer("alpha")
    assert not again.cached and again.prompt_cached and generator.calls == 1
    assert again.answer == first.answer and [s.chunk_id for s in again.sources] == [
        s.chunk_id for s in first.sources
    ]

    # The stream reuses it too, with no wait for the model and no "reading" stage.
    bump_corpus_generation()
    events = list(pipeline.answer_stream("alpha"))
    done = next(e for e in events if e["type"] == "done")
    assert done["prompt_cached"] and generator.calls == 1
    assert not any(e["type"] == "stage" and e.get("stage") == "reading" for e in events)

    # Regenerate asks the model.
    assert not pipeline.answer("alpha", use_cache=False).prompt_cached
    assert generator.calls == 2

    # A new relevant document changes the passages, so the prompt: asked the model.
    IngestService(
        pipeline._repository, StaticEmbeddingProvider([make_embedding(1.0)]), pipeline._settings
    ).ingest("more.txt", b"Intro\nalpha gamma")
    assert not pipeline.answer("alpha").prompt_cached
    assert generator.calls == 3


def test_a_cut_off_reply_is_not_kept_by_the_prompt_cache() -> None:
    from api.corpus import bump_corpus_generation
    from api.llm import reset_finish_reason

    class _Cut(_Counting):
        def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
            answer = super().complete(messages, settings)
            reset_finish_reason("length")
            return answer

    generator = _Cut("Alpha is documented [1] and")
    pipeline = _pipeline(generator, followups_enabled=False, prompt_cache_size=8)
    pipeline.answer("alpha")
    bump_corpus_generation()
    assert not pipeline.answer("alpha").prompt_cached
    assert generator.calls == 2


def test_a_prompt_the_provider_read_far_less_of_is_flagged_and_not_cached() -> None:
    from api.llm import LiteLLMGenerator, begin_usage, was_prompt_cut

    long_text = " ".join(f"Fact {i}: item {i} costs {i * 7} euros." for i in range(400))
    messages: list[ChatMessage] = [
        {"role": "system", "content": "Answer from the facts."},
        {"role": "user", "content": long_text},
    ]

    def client(reported: int):  # type: ignore[no-untyped-def]
        def call(**kwargs: object) -> object:
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": reported, "completion_tokens": 1},
            }

        return call

    settings = make_test_settings(
        llm_provider="openai_compatible",
        openai_compatible_base_url="http://127.0.0.1:9/v1",
        openai_compatible_model="m",
    )
    begin_usage()
    LiteLLMGenerator(settings, completion_client=client(200)).complete(messages, settings)
    assert was_prompt_cut()
    begin_usage()
    LiteLLMGenerator(settings, completion_client=client(6000)).complete(messages, settings)
    assert not was_prompt_cut()


def _sse_events(text: str) -> list[dict[str, object]]:
    import json

    return [
        json.loads(line[len("data: ") :]) for line in text.splitlines() if line.startswith("data: ")
    ]


def test_a_cut_off_answer_is_continued_with_citations_over_the_whole_text() -> None:
    from fastapi.testclient import TestClient

    from api.dependencies import clear_dependency_caches, get_generator
    from api.llm import reset_finish_reason
    from tests.routes_http.support import hermetic_app, upload_and_wait
    from tests.routes_http.support import make_settings as route_settings

    class _CutOnce:
        def __init__(self) -> None:
            self.calls: list[list[ChatMessage]] = []

        def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
            return "unused"

        def stream(self, messages: Sequence[ChatMessage], settings: AppSettings):  # type: ignore[no-untyped-def]
            self.calls.append(list(messages))
            if len(self.calls) == 1:
                yield "Alpha is documented [1] and"
                reset_finish_reason("length")
            else:
                yield " beta follows it [1]."  # a model continuing writes its space
                reset_finish_reason("stop")

    generator = _CutOnce()
    clear_dependency_caches()
    app = hermetic_app(route_settings(citation_retry_enabled=False), QdrantClient(":memory:"))
    app.dependency_overrides[get_generator] = lambda: generator
    with TestClient(app) as client:
        assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
        first = _sse_events(client.post("/questions/stream", json={"question": "alpha?"}).text)
        done = next(e for e in first if e["type"] == "done")
        assert done["truncated"] is True

        refused = client.post("/questions/continue", json={"trace_id": "nope"})
        assert refused.status_code == 404
        rest = _sse_events(
            client.post("/questions/continue", json={"trace_id": done["trace_id"]}).text
        )
        finished = next(e for e in rest if e["type"] == "done")
        assert finished["answer"] == "Alpha is documented [1] and beta follows it [1]."
        assert finished["truncated"] is False
        assert [s["filename"] for s in finished["sources"]] == ["guide.txt"]
        # The model (Ollama: it continues a trailing assistant message) saw its prompt
        # and its answer so far, to carry on from.
        sent = generator.calls[1]
        assert sent[-1] == {"role": "assistant", "content": "Alpha is documented [1] and"}
        assert sent[:-1] == generator.calls[0]
        # The continued answer isn't cut off: continuing it again is refused.
        again = client.post("/questions/continue", json={"trace_id": finished["trace_id"]})
        assert again.status_code == 409
    clear_dependency_caches()


def test_continuing_rebuilds_from_the_original_prompt_and_drops_a_restart() -> None:
    from api.pipeline import (
        CONTINUE_INSTRUCTION,
        base_prompt,
        continuation_messages,
        drop_restart,
    )

    base: list[ChatMessage] = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
    ]
    prefill = continuation_messages(base, "Alpha is ", make_test_settings(llm_provider="ollama"))
    assert prefill[-1] == {"role": "assistant", "content": "Alpha is"}
    asked = continuation_messages(
        base, "Alpha is", make_test_settings(llm_provider="openai", openai_api_key="k")
    )
    assert asked[-1] == {"role": "user", "content": CONTINUE_INSTRUCTION}
    # Continuing a continued answer starts again from the original prompt.
    assert base_prompt(prefill) == base and base_prompt(asked) == base

    before = "The travel policy rules are: book flights fourteen days ahead [1]. Economy"
    restarted = before + " class is the default [1]."
    assert drop_restart(before, restarted) == " class is the default [1]."
    assert drop_restart(before, " class is the default.") == " class is the default."


def test_a_prefilled_continuation_is_joined_as_written() -> None:
    from api.pipeline import join_continuation

    assert join_continuation("cited as [", "2].", prefilled=True) == "cited as [2]."
    assert join_continuation("business class ", " is allowed", prefilled=True) == (
        "business class is allowed"
    )
    assert join_continuation("business class", "is allowed") == "business class is allowed"


def _retrieved(chunk_id: str, filename: str = "a.md") -> Any:
    from api.retrieval import RetrievedChunk

    return RetrievedChunk(f"p-{chunk_id}", filename, 1, "S", chunk_id, chunk_id, 0.5)


def _scored(chunk_id: str, score: float, filename: str = "a.md") -> Any:
    from api.reranking import RerankedChunk

    return RerankedChunk(f"p-{chunk_id}", filename, 1, "S", chunk_id, chunk_id, 0.5, score)


def test_a_search_explains_where_an_expected_passage_was_lost() -> None:
    from api.explain import explain
    from api.pipeline import RetrievalPhase

    settings = make_test_settings(rerank_min_score=0.3, rerank_top_k=1, rerank_candidates=3)
    fused = [_retrieved(c) for c in ("c1", "c2", "c3", "c4", "c5")]
    scored = [_scored("c1", 0.9), _scored("c2", 0.6), _scored("c3", 0.1)]
    phase = RetrievalPhase(
        context_chunks=[scored[0]],
        fused_candidates=fused,
        scored_chunks=scored,
        embed_ms=0,
        search_ms=0,
        rerank_ms=0,
        leg_ranks={"p-c4": [7, None]},
    )
    outcomes = {
        c: explain(phase, settings, chunk_id=c).outcome for c in ("c1", "c2", "c3", "c4", "gone")
    }
    assert outcomes == {
        "c1": "returned",
        "c2": "past_top_k",
        "c3": "below_min_score",
        "c4": "not_scored",
        "gone": "not_retrieved",
    }
    assert explain(phase, settings, chunk_id="c4").dense_rank == 7
    dropped = RetrievalPhase(
        context_chunks=[scored[0]],
        fused_candidates=fused,
        scored_chunks=scored,
        embed_ms=0,
        search_ms=0,
        rerank_ms=0,
        diversity_dropped_ids=frozenset({"p-c2"}),
    )
    loose = make_test_settings(rerank_min_score=0.3, rerank_top_k=5, rerank_candidates=3)
    assert explain(dropped, loose, chunk_id="c2").outcome == "near_duplicate"
    # A filename names its best-fused passage.
    assert explain(phase, settings, filename="a.md").chunk_id == "c1"


def test_search_explain_reports_leg_ranks_and_the_explanation_over_http() -> None:
    from fastapi.testclient import TestClient

    from api.dependencies import clear_dependency_caches
    from tests.routes_http.support import hermetic_app, upload_and_wait
    from tests.routes_http.support import make_settings as route_settings

    clear_dependency_caches()
    app = hermetic_app(route_settings(), QdrantClient(":memory:"))
    with TestClient(app) as client:
        assert upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")["state"] == "done"
        plain = client.post("/search", json={"query": "alpha"}).json()
        assert plain["explanation"] is None and plain["results"][0]["dense_rank"] is None
        asked = client.post(
            "/search",
            json={"query": "alpha", "explain": True, "expect_filename": "guide.txt"},
        ).json()
        assert asked["explanation"]["outcome"] == "returned"
        assert asked["results"][0]["dense_rank"] == 1
    clear_dependency_caches()
