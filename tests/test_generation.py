from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from api.generation import (
    CITATION_EXCERPT_MAX_CHARS,
    CITATION_RETRY_REMINDER,
    ChatMessage,
    GenerationConfigError,
    GenerationError,
    LiteLLMGenerator,
    build_condense_messages,
    build_grounded_messages,
    completion_model_and_kwargs,
    condense_question,
    effective_max_tokens,
    extract_completion_text,
    finalize_citations,
    generate_grounded_answer,
    generate_query_variants,
    needs_citation_retry,
    retry_uncited_answer,
    source_citations,
)
from api.reranking import RerankedChunk
from api.retrieval import RetrievalError
from api.settings import AppSettings


class FakeGenerator:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.messages: list[ChatMessage] = []

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        self.messages = list(messages)
        self.settings = settings
        return self.answer


class ScriptedGenerator:
    """Returns queued answers in order; records the messages of each call."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls: list[list[ChatMessage]] = []
        self.settings_seen: list[AppSettings] = []

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        self.calls.append(list(messages))
        self.settings_seen.append(settings)
        index = min(len(self.calls) - 1, len(self.answers) - 1)
        return self.answers[index]


class RaisingGenerator:
    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        raise GenerationError("boom")


class FakeCompletionClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.kwargs: dict[str, Any] | None = None

    def __call__(self, **kwargs: Any) -> object:
        self.kwargs = kwargs
        return self.response


@dataclass
class Message:
    content: str


@dataclass
class Choice:
    message: Message


@dataclass
class Response:
    choices: list[Choice]


def make_settings(**overrides: Any) -> AppSettings:
    defaults: dict[str, Any] = {
        "max_context_chunks": 2,
        "llm_provider": "ollama",
        "llm_model": "llama3.1:8b",
        "ollama_base_url": "http://ollama:11434",
        "llm_temperature": 0,
        "llm_max_tokens": 512,
    }
    defaults.update(overrides)
    return AppSettings(_env_file=None, **defaults)  # type: ignore[call-arg]


def make_chunk(chunk_id: str, text: str, page: int = 1) -> RerankedChunk:
    return RerankedChunk(
        point_id=f"point-{chunk_id}",
        filename="guide.md",
        page=page,
        section="Setup",
        chunk_id=chunk_id,
        text=text,
        retrieval_score=0.2,
        rerank_score=0.9,
    )


def test_build_grounded_messages_requires_context_only_and_citations() -> None:
    messages = build_grounded_messages(
        "How do I run it?",
        [make_chunk("c1", "Run docker compose up.", page=3)],
    )

    assert messages[0]["role"] == "system"
    assert "Answer only from the provided context" in messages[0]["content"]
    assert "If the context is insufficient" in messages[0]["content"]
    user_prompt = messages[1]["content"]
    assert "How do I run it?" in user_prompt
    assert "[1] filename=guide.md; page=3; section=Setup; chunk_id=c1" in user_prompt
    assert "Run docker compose up." in user_prompt


def test_build_grounded_messages_inserts_history_between_system_and_user() -> None:
    history: list[ChatMessage] = [
        {"role": "user", "content": "What is DocRAG?"},
        {"role": "assistant", "content": "A document Q&A system [1]."},
    ]
    messages = build_grounded_messages(
        "What about the second one?",
        [make_chunk("c1", "Run docker compose up.", page=3)],
        history,
    )

    assert messages[0]["role"] == "system"
    assert "continuity only" in messages[0]["content"]
    assert "Answer only from the provided context" in messages[0]["content"]
    assert messages[1] == history[0]
    assert messages[2] == history[1]
    assert messages[3]["role"] == "user"
    assert "What about the second one?" in messages[3]["content"]


def test_build_grounded_messages_no_history_shape_is_unchanged() -> None:
    messages = build_grounded_messages(
        "How do I run it?",
        [make_chunk("c1", "Run docker compose up.", page=3)],
    )

    assert len(messages) == 2
    assert "continuity only" not in messages[0]["content"]


def test_build_condense_messages_renders_transcript_and_follow_up() -> None:
    history: list[ChatMessage] = [
        {"role": "user", "content": "What is DocRAG?"},
        {"role": "assistant", "content": "A document Q&A system."},
    ]
    messages = build_condense_messages("What about the second one?", history)

    assert messages[0]["role"] == "system"
    user_content = messages[1]["content"]
    assert "User: What is DocRAG?" in user_content
    assert "Assistant: A document Q&A system." in user_content
    assert "Follow-up question: What about the second one?" in user_content


def test_condense_question_forwards_temperature_zero_and_condense_max_tokens() -> None:
    settings = make_settings(llm_temperature=0.9, llm_max_tokens=512, condense_max_tokens=64)
    generator = FakeGenerator("What is the second document about?")
    history: list[ChatMessage] = [{"role": "user", "content": "Tell me about doc A and doc B."}]

    result = condense_question("What about the second one?", history, generator, settings)

    assert result == "What is the second document about?"
    assert generator.settings.llm_temperature == 0.0
    assert generator.settings.llm_max_tokens == 64


def test_condense_question_strips_surrounding_quotes_and_whitespace() -> None:
    settings = make_settings()
    generator = FakeGenerator('  "What is the second document about?"  ')
    history: list[ChatMessage] = [{"role": "user", "content": "Tell me about doc A and doc B."}]

    result = condense_question("What about the second one?", history, generator, settings)

    assert result == "What is the second document about?"


def test_condense_question_returns_empty_string_for_empty_response() -> None:
    settings = make_settings()
    generator = FakeGenerator("   ")
    history: list[ChatMessage] = [{"role": "user", "content": "Tell me about doc A."}]

    result = condense_question("What about it?", history, generator, settings)

    assert result == ""


def test_generate_grounded_answer_limits_context_and_returns_source_metadata() -> None:
    settings = make_settings(max_context_chunks=2)
    chunks = [
        make_chunk("c1", "first"),
        make_chunk("c2", "second"),
        make_chunk("c3", "third"),
    ]
    generator = FakeGenerator("Use docker compose up [1].")

    answer = generate_grounded_answer("  How to run?  ", chunks, generator, settings)

    assert answer.answer == "Use docker compose up [1]."
    # Context is capped at 2 chunks (c1, c2); only [1] is cited, so only c1 is returned.
    assert [source.chunk_id for source in answer.sources] == ["c1"]
    assert answer.sources[0].source_number == 1
    assert answer.sources[0].text == "first"
    assert "third" not in generator.messages[1]["content"]


def test_source_citations_truncates_long_excerpts() -> None:
    long_text = "word " * 100  # far exceeds CITATION_EXCERPT_MAX_CHARS
    chunk = make_chunk("c1", long_text)

    citations = source_citations([chunk])

    assert citations[0].text.endswith("...")
    assert len(citations[0].text) <= CITATION_EXCERPT_MAX_CHARS + 3


def test_generate_grounded_answer_returns_only_cited_sources() -> None:
    settings = make_settings(max_context_chunks=3)
    chunks = [make_chunk("c1", "first"), make_chunk("c2", "second"), make_chunk("c3", "third")]
    generator = FakeGenerator("The third document explains it [3].")

    answer = generate_grounded_answer("How?", chunks, generator, settings)

    assert [source.chunk_id for source in answer.sources] == ["c3"]
    assert answer.sources[0].source_number == 3


def test_generate_grounded_answer_returns_no_sources_when_none_cited() -> None:
    settings = make_settings(max_context_chunks=2)
    chunks = [make_chunk("c1", "first"), make_chunk("c2", "second")]
    generator = FakeGenerator("An answer with no bracketed citations at all.")

    answer = generate_grounded_answer("How?", chunks, generator, settings)

    assert answer.sources == []


def test_generate_grounded_answer_ignores_out_of_range_citations() -> None:
    settings = make_settings(max_context_chunks=2)
    chunks = [make_chunk("c1", "first"), make_chunk("c2", "second")]
    generator = FakeGenerator("Grounded in [2], with a bogus marker [9].")

    answer = generate_grounded_answer("How?", chunks, generator, settings)

    assert [source.chunk_id for source in answer.sources] == ["c2"]


def test_generate_grounded_answer_returns_insufficient_when_no_context() -> None:
    generator = FakeGenerator("should not be called")

    answer = generate_grounded_answer("Question?", [], generator, make_settings())

    assert "not have enough information" in answer.answer
    assert answer.sources == []
    assert generator.messages == []


def test_generate_grounded_answer_rejects_empty_query() -> None:
    with pytest.raises(RetrievalError, match="Query text"):
        generate_grounded_answer(" ", [], FakeGenerator("unused"), make_settings())


def test_generate_grounded_answer_rejects_empty_model_response() -> None:
    with pytest.raises(GenerationError, match="empty answer"):
        generate_grounded_answer(
            "Question?",
            [make_chunk("c1", "text")],
            FakeGenerator(" "),
            make_settings(),
        )


def test_litellm_generator_uses_ollama_configuration() -> None:
    completion_client = FakeCompletionClient({"choices": [{"message": {"content": "Answer [1]."}}]})
    settings = make_settings()
    generator = LiteLLMGenerator(settings, completion_client=completion_client)

    answer = generator.complete([{"role": "user", "content": "Hi"}], settings)

    assert answer == "Answer [1]."
    assert completion_client.kwargs is not None
    assert completion_client.kwargs["model"] == "ollama/llama3.1:8b"
    assert completion_client.kwargs["base_url"] == "http://ollama:11434"
    assert completion_client.kwargs["temperature"] == 0
    assert completion_client.kwargs["max_tokens"] == 512
    assert completion_client.kwargs["timeout"] == 60.0
    assert completion_client.kwargs["drop_params"] is True


def test_litellm_generator_routes_to_settings_provider_not_construction_provider() -> None:
    """A per-request settings override picks the provider, not the construction settings.

    Guards the per-request switch: one cached generator built with the ollama base must
    still route to OpenAI when handed OpenAI settings at call time.
    """

    completion_client = FakeCompletionClient({"choices": [{"message": {"content": "Answer [1]."}}]})
    base_settings = make_settings()  # llm_provider="ollama"
    generator = LiteLLMGenerator(base_settings, completion_client=completion_client)

    override = base_settings.model_copy(
        update={
            "llm_provider": "openai",
            "openai_model": "gpt-test",
            "openai_api_key": "test-key",
        }
    )
    generator.complete([{"role": "user", "content": "Hi"}], override)

    assert completion_client.kwargs is not None
    assert completion_client.kwargs["model"] == "gpt-test"
    assert completion_client.kwargs["api_key"] == "test-key"
    assert "base_url" not in completion_client.kwargs


def test_litellm_generator_openai_override_without_key_raises() -> None:
    completion_client = FakeCompletionClient({"choices": [{"message": {"content": "x"}}]})
    base_settings = make_settings()
    generator = LiteLLMGenerator(base_settings, completion_client=completion_client)

    override = base_settings.model_copy(update={"llm_provider": "openai", "openai_api_key": None})
    with pytest.raises(GenerationConfigError, match="OPENAI_API_KEY"):
        generator.complete([{"role": "user", "content": "Hi"}], override)


def test_completion_model_and_kwargs_supports_openai_with_key() -> None:
    model, kwargs = completion_model_and_kwargs(
        make_settings(
            llm_provider="openai",
            openai_model="gpt-test",
            openai_api_key="test-key",
        )
    )

    assert model == "gpt-test"
    assert kwargs == {"api_key": "test-key"}


def test_completion_model_and_kwargs_rejects_openai_without_key() -> None:
    with pytest.raises(GenerationConfigError, match="OPENAI_API_KEY"):
        completion_model_and_kwargs(make_settings(llm_provider="openai", openai_api_key=None))


def test_completion_model_and_kwargs_rejects_unknown_provider() -> None:
    with pytest.raises(GenerationConfigError, match="Unsupported"):
        completion_model_and_kwargs(make_settings(llm_provider="anthropic"))


def test_completion_model_and_kwargs_adds_reasoning_effort_for_reasoning_model() -> None:
    model, kwargs = completion_model_and_kwargs(
        make_settings(
            llm_provider="openai",
            openai_model="gpt-5-mini",
            openai_api_key="test-key",
            openai_reasoning_effort="low",
        )
    )

    assert model == "gpt-5-mini"
    assert kwargs == {"api_key": "test-key", "reasoning_effort": "low"}


def test_completion_model_and_kwargs_omits_reasoning_effort_for_plain_model() -> None:
    _, kwargs = completion_model_and_kwargs(
        make_settings(
            llm_provider="openai",
            openai_model="gpt-4o-mini",
            openai_api_key="test-key",
            openai_reasoning_effort="low",
        )
    )

    assert "reasoning_effort" not in kwargs


def test_completion_model_and_kwargs_omits_reasoning_effort_when_disabled() -> None:
    _, kwargs = completion_model_and_kwargs(
        make_settings(
            llm_provider="openai",
            openai_model="gpt-5-mini",
            openai_api_key="test-key",
            openai_reasoning_effort="",
        )
    )

    assert "reasoning_effort" not in kwargs


def test_effective_max_tokens_adds_headroom_only_for_reasoning_models() -> None:
    settings = make_settings(llm_max_tokens=512, reasoning_token_headroom=3072)

    assert effective_max_tokens(settings, "gpt-5-mini") == 512 + 3072
    assert effective_max_tokens(settings, "gpt-4o-mini") == 512
    assert effective_max_tokens(settings, "ollama/llama3.1:8b") == 512


def test_reasoning_model_gets_expanded_max_tokens_in_completion_call() -> None:
    completion_client = FakeCompletionClient(
        {"choices": [{"message": {"content": "Answer [1]."}}]}
    )
    settings = make_settings(
        llm_provider="openai",
        openai_model="gpt-5-mini",
        openai_api_key="test-key",
        llm_max_tokens=512,
        reasoning_token_headroom=3072,
    )
    generator = LiteLLMGenerator(settings, completion_client=completion_client)

    generator.complete([{"role": "user", "content": "Hi"}], settings)

    assert completion_client.kwargs is not None
    assert completion_client.kwargs["max_tokens"] == 512 + 3072
    assert completion_client.kwargs["reasoning_effort"] == "low"


def test_extract_completion_text_supports_object_responses() -> None:
    response = Response(choices=[Choice(Message("Object answer"))])

    assert extract_completion_text(response) == "Object answer"


def test_extract_completion_text_rejects_malformed_response() -> None:
    with pytest.raises(GenerationError, match="choices"):
        extract_completion_text({})


class RaisingCompletionClient:
    def __call__(self, **kwargs: Any) -> object:
        raise RuntimeError("connection refused")


def test_litellm_generator_wraps_provider_exceptions_as_generation_error() -> None:
    """The completion boundary must translate provider-library exceptions (auth,
    connection, rate-limit, ...) into GenerationError so main.py maps them to a clean
    502 instead of an opaque 500 (proven live: a bad OpenAI key previously surfaced as
    a raw 500 after ~40s of LiteLLM retries).
    """

    generator = LiteLLMGenerator(make_settings(), completion_client=RaisingCompletionClient())

    with pytest.raises(GenerationError, match="connection refused"):
        generator.complete([{"role": "user", "content": "Hi"}], make_settings())


class RaisingAPIConnectionClient:
    def __init__(self, *, model: str, llm_provider: str) -> None:
        self._model = model
        self._llm_provider = llm_provider

    def __call__(self, **kwargs: Any) -> object:
        from litellm.exceptions import APIConnectionError

        raise APIConnectionError(
            message="Connection refused",
            model=self._model,
            llm_provider=self._llm_provider,
        )


def test_litellm_generator_raises_actionable_message_for_unreachable_ollama() -> None:
    """A connection failure to Ollama must surface a message pointing the user at
    `ollama serve` and the configured OLLAMA_BASE_URL — the exact regression proven
    live this session (Ollama not running -> opaque 502)."""

    settings = make_settings(llm_provider="ollama", ollama_base_url="http://localhost:11434")
    completion_client = RaisingAPIConnectionClient(
        model="ollama/llama3.1:8b", llm_provider="ollama"
    )
    generator = LiteLLMGenerator(settings, completion_client=completion_client)

    with pytest.raises(GenerationError, match="ollama serve"):
        generator.complete([{"role": "user", "content": "Hi"}], settings)


def test_litellm_generator_openai_connection_failure_uses_generic_message() -> None:
    """The Ollama-specific branch must not fire for other providers."""

    settings = make_settings(llm_provider="openai", openai_api_key="sk-test")
    completion_client = RaisingAPIConnectionClient(model="gpt-4o-mini", llm_provider="openai")
    generator = LiteLLMGenerator(settings, completion_client=completion_client)

    with pytest.raises(GenerationError) as excinfo:
        generator.complete([{"role": "user", "content": "Hi"}], settings)
    assert "ollama serve" not in str(excinfo.value)
    assert "Generation provider request failed" in str(excinfo.value)


def test_litellm_generator_passes_num_retries_from_settings() -> None:
    completion_client = FakeCompletionClient({"choices": [{"message": {"content": "ok"}}]})
    settings = make_settings(llm_num_retries=2)
    generator = LiteLLMGenerator(settings, completion_client=completion_client)

    generator.complete([{"role": "user", "content": "Hi"}], settings)

    assert completion_client.kwargs is not None
    assert completion_client.kwargs["num_retries"] == 2


def test_completion_model_and_kwargs_passes_ollama_keep_alive() -> None:
    model, kwargs = completion_model_and_kwargs(
        make_settings(llm_provider="ollama", ollama_keep_alive="45m")
    )

    assert model == "ollama/llama3.1:8b"
    assert kwargs["keep_alive"] == "45m"


class FakeStreamCompletionClient:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.kwargs: dict[str, Any] | None = None

    def __call__(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return iter(self.chunks)


def test_litellm_generator_stream_yields_deltas_and_sets_stream_true() -> None:
    chunks = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {}}]},
    ]
    completion_client = FakeStreamCompletionClient(chunks)
    settings = make_settings()
    generator = LiteLLMGenerator(settings, completion_client=completion_client)

    deltas = list(generator.stream([{"role": "user", "content": "Hi"}], settings))

    assert deltas == ["Hel", "lo"]
    assert completion_client.kwargs is not None
    assert completion_client.kwargs["stream"] is True


def test_litellm_generator_stream_wraps_connection_failure_for_ollama() -> None:
    class RaisingStreamClient:
        def __call__(self, **kwargs: Any) -> Any:
            from litellm.exceptions import APIConnectionError

            raise APIConnectionError(
                message="Connection refused", model="ollama/llama3.1:8b", llm_provider="ollama"
            )

    settings = make_settings(llm_provider="ollama", ollama_base_url="http://localhost:11434")
    generator = LiteLLMGenerator(settings, completion_client=RaisingStreamClient())

    with pytest.raises(GenerationError, match="ollama serve"):
        list(generator.stream([{"role": "user", "content": "Hi"}], settings))


def test_generate_query_variants_parses_and_dedupes() -> None:
    generator = ScriptedGenerator(
        ["1. how to deploy docrag\n2. docrag deployment steps\n- how to deploy docrag"]
    )
    variants = generate_query_variants("deploy docrag", 3, generator, make_settings())
    # Numbering/bullets stripped, exact duplicate collapsed.
    assert variants == ["how to deploy docrag", "docrag deployment steps"]


def test_generate_query_variants_preserves_leading_digit_in_content() -> None:
    generator = ScriptedGenerator(["1. 3D printing basics\n- 5G network design"])
    variants = generate_query_variants("printing", 3, generator, make_settings())
    # List markers stripped, but "3D" / "5G" leading digits are kept.
    assert variants == ["3D printing basics", "5G network design"]


def test_generate_query_variants_drops_echo_of_original() -> None:
    generator = ScriptedGenerator(["deploy docrag\nalternative deploy phrasing"])
    variants = generate_query_variants("deploy docrag", 3, generator, make_settings())
    assert variants == ["alternative deploy phrasing"]


def test_generate_query_variants_returns_empty_on_error() -> None:
    assert generate_query_variants("q", 3, RaisingGenerator(), make_settings()) == []


def test_generate_query_variants_respects_zero_count() -> None:
    generator = ScriptedGenerator(["a\nb\nc"])
    assert generate_query_variants("q", 0, generator, make_settings()) == []


def test_generate_query_variants_forwards_expansion_settings() -> None:
    generator = ScriptedGenerator(["variant one\nvariant two"])
    settings = make_settings(query_expansion_max_tokens=64)
    generate_query_variants("q", 2, generator, settings)
    # The expansion call runs at temperature 0.3 with the expansion token cap.
    assert generator.settings_seen[0].llm_temperature == 0.3
    assert generator.settings_seen[0].llm_max_tokens == 64


def test_needs_citation_retry_logic() -> None:
    assert needs_citation_retry("The answer is here.", 2) is True
    assert needs_citation_retry("The answer is [1].", 2) is False
    assert needs_citation_retry("The answer is here.", 0) is False
    assert needs_citation_retry("The documents do not contain enough information.", 2) is False


def test_retry_uncited_answer_returns_cited_retry_and_the_messages_it_sent() -> None:
    generator = ScriptedGenerator(["Now with citation [1]."])
    prompt = build_grounded_messages("q", [make_chunk("c1", "context text")])

    result = retry_uncited_answer(prompt, "no citation here", generator, make_settings())

    assert result is not None
    answer, sent = result
    assert answer == "Now with citation [1]."
    # The retry continues the original conversation rather than restating it: the
    # grounded prompt, then the uncited answer, then the reminder.
    assert sent[: len(prompt)] == prompt
    assert sent[-2] == {"role": "assistant", "content": "no citation here"}
    assert sent[-1] == {"role": "user", "content": CITATION_RETRY_REMINDER}
    assert sent == generator.calls[-1]


def test_retry_uncited_answer_returns_none_when_still_uncited() -> None:
    generator = ScriptedGenerator(["still no citation"])
    prompt = build_grounded_messages("q", [make_chunk("c1", "context text")])

    assert retry_uncited_answer(prompt, "no citation", generator, make_settings()) is None


def test_finalize_citations_reports_the_retry_prompt_as_what_produced_the_answer() -> None:
    """The outcome's prompt must be the retry continuation, not the first prompt.

    This is what the debug trace records. Reporting the original prompt alongside a
    retried answer would show a prompt the model was never asked.
    """

    generator = ScriptedGenerator(["Cited now [1]."])
    chunks = [make_chunk("c1", "context text")]
    prompt = build_grounded_messages("q", chunks)

    outcome = finalize_citations(
        prompt, "no citation", source_citations(chunks), generator, make_settings()
    )

    assert outcome.retry_used is True
    assert outcome.answer == "Cited now [1]."
    assert outcome.prompt_messages[-1] == {"role": "user", "content": CITATION_RETRY_REMINDER}


def test_finalize_citations_reports_the_original_prompt_when_no_retry_happens() -> None:
    generator = ScriptedGenerator(["unused"])
    chunks = [make_chunk("c1", "context text")]
    prompt = build_grounded_messages("q", chunks)

    outcome = finalize_citations(
        prompt, "Already cited [1].", source_citations(chunks), generator, make_settings()
    )

    assert outcome.retry_used is False
    assert outcome.prompt_messages == prompt


def test_generate_grounded_answer_carries_the_retry_prompt_out() -> None:
    generator = ScriptedGenerator(["The setup is simple.", "The setup is simple [1]."])

    result = generate_grounded_answer(
        "How do I set up?",
        [make_chunk("c1", "Run docker compose up.")],
        generator,
        make_settings(citation_retry_enabled=True),
    )

    assert result.citation_retry_used is True
    assert result.prompt_messages[-1] == {"role": "user", "content": CITATION_RETRY_REMINDER}
    # And it is the message list the generator was actually called with, not a rebuild.
    assert result.prompt_messages == generator.calls[-1]


def test_generate_grounded_answer_retries_uncited_answer() -> None:
    generator = ScriptedGenerator(["The setup is simple.", "The setup is simple [1]."])
    result = generate_grounded_answer(
        "How do I set up?",
        [make_chunk("c1", "Run docker compose up.")],
        generator,
        make_settings(citation_retry_enabled=True),
    )
    assert result.answer == "The setup is simple [1]."
    assert result.citation_retry_used is True
    assert [source.source_number for source in result.sources] == [1]


def test_generate_grounded_answer_skips_retry_when_disabled() -> None:
    generator = ScriptedGenerator(["The setup is simple."])
    result = generate_grounded_answer(
        "How do I set up?",
        [make_chunk("c1", "Run docker compose up.")],
        generator,
        make_settings(citation_retry_enabled=False),
    )
    assert result.answer == "The setup is simple."
    assert result.citation_retry_used is False
    assert len(generator.calls) == 1  # no retry call


def test_generate_grounded_answer_skips_retry_on_insufficiency() -> None:
    generator = ScriptedGenerator(["The provided documents do not contain enough information."])
    result = generate_grounded_answer(
        "How do I set up?",
        [make_chunk("c1", "Unrelated content.")],
        generator,
        make_settings(citation_retry_enabled=True),
    )
    assert result.citation_retry_used is False
    assert len(generator.calls) == 1
