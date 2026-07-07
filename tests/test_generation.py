from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from api.generation import (
    CITATION_EXCERPT_MAX_CHARS,
    ChatMessage,
    GenerationConfigError,
    GenerationError,
    LiteLLMGenerator,
    build_grounded_messages,
    completion_model_and_kwargs,
    extract_completion_text,
    generate_grounded_answer,
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
    return AppSettings(**defaults)


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


def test_generate_grounded_answer_keeps_all_sources_when_none_cited() -> None:
    settings = make_settings(max_context_chunks=2)
    chunks = [make_chunk("c1", "first"), make_chunk("c2", "second")]
    generator = FakeGenerator("An answer with no bracketed citations at all.")

    answer = generate_grounded_answer("How?", chunks, generator, settings)

    assert [source.chunk_id for source in answer.sources] == ["c1", "c2"]


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


def test_litellm_generator_passes_num_retries_from_settings() -> None:
    completion_client = FakeCompletionClient({"choices": [{"message": {"content": "ok"}}]})
    settings = make_settings(llm_num_retries=2)
    generator = LiteLLMGenerator(settings, completion_client=completion_client)

    generator.complete([{"role": "user", "content": "Hi"}], settings)

    assert completion_client.kwargs is not None
    assert completion_client.kwargs["num_retries"] == 2
