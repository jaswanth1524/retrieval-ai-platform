"""Grounded answer generation through LiteLLM."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict

import litellm
from litellm.exceptions import APIConnectionError as LiteLLMAPIConnectionError

from api.reranking import RerankedChunk
from api.retrieval import RetrievalError
from api.settings import AppSettings

INSUFFICIENT_CONTEXT_ANSWER = (
    "I do not have enough information in the provided documents to answer that question."
)

_CITATION_RE = re.compile(r"\[(\d+)\]")

CITATION_EXCERPT_MAX_CHARS = 200


class GenerationError(RuntimeError):
    """Raised when grounded generation fails."""


class GenerationConfigError(GenerationError):
    """Raised when the selected generation provider is not configured."""


class ChatMessage(TypedDict):
    """LiteLLM-compatible chat message."""

    role: Literal["system", "user", "assistant"]
    content: str


class CompletionClient(Protocol):
    """Callable surface used for LiteLLM completion."""

    def __call__(self, **kwargs: Any) -> object: ...


@dataclass(frozen=True)
class SourceCitation:
    """Source metadata returned with each generated answer."""

    source_number: int
    filename: str
    page: int
    section: str
    chunk_id: str
    text: str


@dataclass(frozen=True)
class GroundedAnswer:
    """Generated answer and the sources made available to the model."""

    answer: str
    sources: list[SourceCitation]


class LiteLLMGenerator:
    """Generate answers with LiteLLM while keeping provider selection configurable."""

    def __init__(
        self,
        settings: AppSettings,
        completion_client: CompletionClient = litellm.completion,
    ) -> None:
        self.settings = settings
        self.completion_client = completion_client

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        """Call the LiteLLM provider selected by ``settings`` and return assistant text.

        Provider selection reads the *passed* settings (a per-request override may
        differ from the ``settings`` captured at construction), so a single cached
        generator instance can serve both the Ollama and OpenAI paths.
        """

        model, provider_kwargs = completion_model_and_kwargs(settings)
        try:
            response = self.completion_client(
                model=model,
                messages=list(messages),
                temperature=float(settings.llm_temperature),
                max_tokens=int(settings.llm_max_tokens),
                timeout=float(settings.llm_request_timeout_seconds),
                num_retries=int(settings.llm_num_retries),
                # Some models (e.g. the gpt-5 family) reject params other models
                # accept fine (temperature != 1). Let LiteLLM drop an unsupported
                # param for the selected model rather than raising, so provider
                # quirks don't turn into a hard failure.
                drop_params=True,
                **provider_kwargs,
            )
        except LiteLLMAPIConnectionError as exc:
            if settings.llm_provider.lower().strip() == "ollama":
                raise GenerationError(
                    f"Cannot reach Ollama at {settings.ollama_base_url}. Start Ollama "
                    "(`ollama serve`) and confirm OLLAMA_BASE_URL is reachable from "
                    "wherever the API process runs — use http://localhost:11434 when "
                    "the API runs directly on your host, or "
                    "http://host.docker.internal:11434 only when the API itself runs "
                    "inside Docker."
                ) from exc
            raise GenerationError(f"Generation provider request failed: {exc}") from exc
        except Exception as exc:
            # LiteLLM/provider SDKs raise many distinct exception types (auth,
            # connection, rate limit, ...); this boundary's job is translating all
            # of them into our domain error so main.py maps them to a clean 502
            # instead of an opaque 500.
            raise GenerationError(f"Generation provider request failed: {exc}") from exc
        return extract_completion_text(response)


class ChatGenerator(Protocol):
    """Generator surface used by the grounded answer pipeline."""

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str: ...


def generate_grounded_answer(
    query: str,
    context_chunks: Sequence[RerankedChunk],
    generator: ChatGenerator,
    settings: AppSettings,
) -> GroundedAnswer:
    """Generate a grounded answer from reranked context chunks."""

    normalized_query = query.strip()
    if not normalized_query:
        raise RetrievalError("Query text is required.")

    selected_chunks = list(context_chunks[: int(settings.max_context_chunks)])
    if not selected_chunks:
        return GroundedAnswer(answer=INSUFFICIENT_CONTEXT_ANSWER, sources=[])

    messages = build_grounded_messages(normalized_query, selected_chunks)
    answer = generator.complete(messages, settings).strip()
    if not answer:
        raise GenerationError("Generation provider returned an empty answer.")

    all_sources = source_citations(selected_chunks)
    return GroundedAnswer(
        answer=answer,
        sources=cited_sources(answer, all_sources),
    )


def build_grounded_messages(
    query: str,
    context_chunks: Sequence[RerankedChunk],
) -> list[ChatMessage]:
    """Build the prompt that constrains the model to provided context."""

    context = "\n\n".join(
        format_context_chunk(index, chunk)
        for index, chunk in enumerate(context_chunks, start=1)
    )
    return [
        {
            "role": "system",
            "content": (
                "You are DocRAG, a document question-answering assistant. "
                "Answer only from the provided context. Do not use outside knowledge. "
                "If the context is insufficient, explicitly say that the provided "
                "documents do not contain enough information. Cite supporting sources "
                "with bracketed source numbers like [1]."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{query}\n\n"
                f"Context:\n{context}\n\n"
                "Instructions:\n"
                "- Use only the context above.\n"
                "- Include citations using source numbers such as [1] or [2].\n"
                "- If the context is insufficient, say so directly."
            ),
        },
    ]


def format_context_chunk(index: int, chunk: RerankedChunk) -> str:
    """Format one chunk and its citation metadata for the generation prompt."""

    return (
        f"[{index}] filename={chunk.filename}; page={chunk.page}; "
        f"section={chunk.section}; chunk_id={chunk.chunk_id}\n"
        f"{chunk.text}"
    )


def source_citations(chunks: Sequence[RerankedChunk]) -> list[SourceCitation]:
    """Return source metadata for chunks included in the prompt context."""

    return [
        SourceCitation(
            source_number=index,
            filename=chunk.filename,
            page=chunk.page,
            section=chunk.section,
            chunk_id=chunk.chunk_id,
            text=_truncate_excerpt(chunk.text),
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


def _truncate_excerpt(text: str, limit: int = CITATION_EXCERPT_MAX_CHARS) -> str:
    """Collapse whitespace and truncate a chunk's text to a short citation excerpt."""

    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "..."


def cited_sources(
    answer: str,
    sources: Sequence[SourceCitation],
) -> list[SourceCitation]:
    """Return only the sources the answer actually cites via ``[n]`` markers.

    An answer with zero ``[n]`` markers is treated as ungrounded/insufficient — it
    gets zero sources rather than every candidate chunk attached regardless of
    relevance.
    """

    referenced = {int(match) for match in _CITATION_RE.findall(answer)}
    available = {source.source_number for source in sources}
    used = referenced & available
    return [source for source in sources if source.source_number in used]


def completion_model_and_kwargs(settings: AppSettings) -> tuple[str, dict[str, str]]:
    """Return LiteLLM model name and provider-specific keyword arguments."""

    provider = settings.llm_provider.lower().strip()
    if provider == "ollama":
        return f"ollama/{settings.llm_model}", {"base_url": settings.ollama_base_url}
    if provider == "openai":
        if not settings.openai_api_key:
            raise GenerationConfigError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        return settings.openai_model, {"api_key": settings.openai_api_key}
    raise GenerationConfigError(f"Unsupported LLM_PROVIDER '{settings.llm_provider}'.")


def extract_completion_text(response: object) -> str:
    """Extract assistant text from LiteLLM's response object or a compatible fake."""

    choices = read_value(response, "choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        raise GenerationError("Generation provider response did not include choices.")

    first_choice = choices[0]
    message = read_value(first_choice, "message")
    content = read_value(message, "content")
    if not isinstance(content, str):
        raise GenerationError("Generation provider response did not include message content.")
    return content


def read_value(source: object, key: str) -> object:
    """Read an attribute or dictionary key from a response object."""

    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)

