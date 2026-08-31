"""Grounded answer generation through LiteLLM."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict, cast

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

CONDENSE_SYSTEM_PROMPT = (
    "You rewrite a follow-up question into one standalone search query for document "
    "retrieval. Use the conversation to resolve pronouns and references. Output only "
    "the rewritten question — no preamble, quotes, or explanation. If the question is "
    "already self-contained, return it unchanged."
)

# Strips a leading bullet ("- ", "• ", "* ") or an enumerator ("1. ", "2) ") from a
# variant line — but only a real list prefix, so a variant like "3D printing basics"
# keeps its leading digit.
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-•*]\s+|\d+[.)]\s+)")


def _strip_list_marker(line: str) -> str:
    return _LIST_MARKER_RE.sub("", line, count=1)


EXPANSION_SYSTEM_PROMPT = (
    "You rewrite a search query into alternative phrasings for document retrieval. "
    "Output one variant per line — no numbering, bullets, quotes, or preamble. Vary "
    "the wording and emphasis while keeping the original meaning."
)

CITATION_RETRY_REMINDER = (
    "Your previous answer included no [n] citation markers. Rewrite it, keeping the "
    "same substance, and cite the specific numbered sources that support each claim "
    "(for example [1], [2]). If the context does not actually support an answer, say "
    "the provided documents do not contain enough information."
)

# Phrases that signal the model already declared the context insufficient — a valid
# no-citation answer that must not trigger a citation retry. Matched against
# _normalize_negation'd text, so only spelled-out forms need to be listed here;
# contractions ("don't", "can't") are expanded to match before comparison.
_INSUFFICIENCY_HINTS = (
    "enough information",
    "insufficient",
    "do not contain",
    "does not contain",
    "cannot answer",
)


def _normalize_negation(text: str) -> str:
    """Expand contractions so hint matching doesn't require every surface form.

    Order matters: ``can't`` is irregular (the generic rule alone would produce
    "ca not"), so it's special-cased before the generic ``n't`` -> " not" rule runs.
    The generic rule also mangles irregular contractions outside our hint vocabulary
    (``won't`` -> "wo not", ``shan't`` -> "sha not") but none of _INSUFFICIENCY_HINTS
    is reachable through a mangled stem, so those false expansions are harmless.
    """

    text = text.replace("’", "'")  # curly apostrophe, common in LLM output
    text = re.sub(r"\bcan't\b", "cannot", text)
    return re.sub(r"n't\b", " not", text)


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
class StageTimings:
    """Wall-clock time spent in each pipeline stage, for latency observability."""

    embed_ms: float
    search_ms: float
    # Cross-encoder scoring plus the diversity filter. Neighbour expansion used to be
    # folded in here, which hid a per-question Qdrant round trip inside "rerank".
    rerank_ms: float
    generate_ms: float
    total_ms: float
    # 0.0 when there was no history to condense, or condense was disabled.
    condense_ms: float = 0.0
    # LLM generation of alternative query phrasings, which happens BEFORE embedding.
    # 0.0 when multi-query expansion is disabled (the default) or produced no variants.
    # Named for what it measures: the old `expand_ms` read as context expansion, which
    # is a different stage at a different point in the pipeline (below).
    query_expansion_ms: float = 0.0
    # Neighbour/context expansion after rerank (small-to-big retrieval). 0.0 when
    # context_neighbor_radius is 0.
    context_expansion_ms: float = 0.0


@dataclass(frozen=True)
class GroundedAnswer:
    """Generated answer and the sources made available to the model."""

    answer: str
    sources: list[SourceCitation]
    timings: StageTimings | None = None
    trace_id: str | None = None
    # True when the answer initially lacked citations and a stricter retry supplied
    # them (see needs_citation_retry / retry_uncited_answer).
    citation_retry_used: bool = False
    # The exact message list of the call that produced `answer` — the retry's
    # continuation when citation_retry_used is True. Carried out so tracing records what
    # was really sent instead of re-deriving a prompt that may not be the one used.
    # Never surfaced through the HTTP response (see main.py's QuestionResponse), only
    # through the debug trace.
    prompt_messages: list[ChatMessage] = field(default_factory=list)


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
                max_tokens=effective_max_tokens(settings, model),
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

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings) -> Iterator[str]:
        """Call the LiteLLM provider with ``stream=True`` and yield assistant text deltas.

        Same provider selection and error-translation boundary as ``complete`` — the
        request is identical except for ``stream=True``. Because this is a generator
        function, the try/except below covers the whole streamed response: LiteLLM's
        ``CustomStreamWrapper`` is lazy, so a connection failure raised while iterating
        chunks is caught here exactly like a failure raised by the initial call.
        """

        model, provider_kwargs = completion_model_and_kwargs(settings)
        try:
            response = self.completion_client(
                model=model,
                messages=list(messages),
                temperature=float(settings.llm_temperature),
                max_tokens=effective_max_tokens(settings, model),
                timeout=float(settings.llm_request_timeout_seconds),
                num_retries=int(settings.llm_num_retries),
                drop_params=True,
                stream=True,
                **provider_kwargs,
            )
            # completion_client's return type is `object` (it must also cover the
            # non-streaming response `complete` uses) — with stream=True it is
            # actually an iterable of chunks; cast narrows that for the loop below.
            for chunk in cast(Iterable[object], response):
                delta = extract_delta_text(chunk)
                if delta:
                    yield delta
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
            raise GenerationError(f"Generation provider request failed: {exc}") from exc


class ChatGenerator(Protocol):
    """Generator surface used by the grounded answer pipeline (both response modes)."""

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str: ...

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings) -> Iterable[str]: ...


def generate_grounded_answer(
    query: str,
    context_chunks: Sequence[RerankedChunk],
    generator: ChatGenerator,
    settings: AppSettings,
    history: Sequence[ChatMessage] | None = None,
) -> GroundedAnswer:
    """Generate a grounded answer from reranked context chunks."""

    normalized_query = query.strip()
    if not normalized_query:
        raise RetrievalError("Query text is required.")

    selected_chunks = list(context_chunks[: int(settings.max_context_chunks)])
    if not selected_chunks:
        return GroundedAnswer(answer=INSUFFICIENT_CONTEXT_ANSWER, sources=[])

    messages = build_grounded_messages(normalized_query, selected_chunks, history)
    answer = generator.complete(messages, settings).strip()
    if not answer:
        raise GenerationError("Generation provider returned an empty answer.")

    all_sources = source_citations(selected_chunks)
    outcome = finalize_citations(messages, answer, all_sources, generator, settings)

    return GroundedAnswer(
        answer=outcome.answer,
        sources=outcome.cited,
        citation_retry_used=outcome.retry_used,
        prompt_messages=outcome.prompt_messages,
    )


def build_grounded_messages(
    query: str,
    context_chunks: Sequence[RerankedChunk],
    history: Sequence[ChatMessage] | None = None,
) -> list[ChatMessage]:
    """Build the prompt that constrains the model to provided context.

    ``history``, when given, is inserted verbatim between the system message and the
    final user message — the final user message always carries the raw ``query`` plus
    context plus instructions, unchanged by history's presence, so the context-only /
    mandatory-citation grounding rules never weaken for a follow-up question.
    """

    context = "\n\n".join(
        format_context_chunk(index, chunk)
        for index, chunk in enumerate(context_chunks, start=1)
    )
    system_content = (
        "You are DocRAG, a document question-answering assistant. "
        "Answer only from the provided context. Do not use outside knowledge. "
        "If the context is insufficient, explicitly say that the provided "
        "documents do not contain enough information. Cite supporting sources "
        "with bracketed source numbers like [1]."
    )
    if history:
        system_content += (
            " Prior conversation turns are provided for continuity only — still "
            "answer strictly from the provided context and cite sources like [1]."
        )
    messages: list[ChatMessage] = [{"role": "system", "content": system_content}]
    if history:
        messages.extend(history)
    messages.append(
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
        }
    )
    return messages


def build_condense_messages(
    question: str,
    history: Sequence[ChatMessage],
) -> list[ChatMessage]:
    """Build the prompt that rewrites a follow-up into a standalone retrieval query."""

    transcript = "\n".join(
        f"{'User' if message['role'] == 'user' else 'Assistant'}: {message['content']}"
        for message in history
    )
    return [
        {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{transcript}\n\n"
                f"Follow-up question: {question}\n\n"
                "Standalone question:"
            ),
        },
    ]


def condense_question(
    question: str,
    history: Sequence[ChatMessage],
    generator: ChatGenerator,
    settings: AppSettings,
) -> str:
    """Rewrite a follow-up question into a standalone retrieval query.

    Deterministic (temperature 0) regardless of the caller's own temperature
    override — this is a mechanical rewrite, not a creative generation. Returns ""
    (rather than raising) when the provider returns an empty response, so the
    caller's own fallback-to-raw-question logic handles both failure modes the
    same way.
    """

    messages = build_condense_messages(question, history)
    condense_settings = settings.model_copy(
        update={
            "llm_temperature": 0.0,
            "llm_max_tokens": int(settings.condense_max_tokens),
        }
    )
    result = generator.complete(messages, condense_settings).strip()
    return result.strip("\"'")


def format_context_chunk(index: int, chunk: RerankedChunk) -> str:
    """Format one chunk and its citation metadata for the generation prompt.

    Uses ``expanded_text`` (neighbor-context expansion) when present so the model
    sees richer surrounding context; the citation shown to the user always excerpts
    the original ``text`` (see ``source_citations``), independent of expansion.
    """

    text = chunk.expanded_text if chunk.expanded_text is not None else chunk.text
    return (
        f"[{index}] filename={chunk.filename}; page={chunk.page}; "
        f"section={chunk.section}; chunk_id={chunk.chunk_id}\n"
        f"{text}"
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


def needs_citation_retry(answer: str, source_count: int) -> bool:
    """True when an answer should be retried for missing citations.

    Only when sources were available, the answer contains no ``[n]`` marker, and the
    answer did not already declare the context insufficient (a valid uncited answer).
    """

    if source_count <= 0 or _CITATION_RE.search(answer):
        return False
    lowered = _normalize_negation(answer.lower())
    return not any(hint in lowered for hint in _INSUFFICIENCY_HINTS)


def retry_uncited_answer(
    prompt_messages: Sequence[ChatMessage],
    first_answer: str,
    generator: ChatGenerator,
    settings: AppSettings,
) -> tuple[str, list[ChatMessage]] | None:
    """One-shot retry that reminds the model to cite.

    Returns ``(cited_answer, messages_actually_sent)``, or None when the retry fails or
    still produces no citations — in which case the caller keeps the original answer.
    Never loops.

    Takes the already-built grounded prompt rather than rebuilding it from
    ``query``/``context_chunks``/``history``: the retry must continue the *same*
    conversation the first answer came from, and reconstructing it here was a second
    derivation that could silently drift from the one generation actually used.
    """

    messages: list[ChatMessage] = [
        *prompt_messages,
        {"role": "assistant", "content": first_answer},
        {"role": "user", "content": CITATION_RETRY_REMINDER},
    ]
    try:
        retried = generator.complete(messages, settings).strip()
    except GenerationError:
        return None
    if not retried or not _CITATION_RE.search(retried):
        return None
    return retried, messages


@dataclass(frozen=True)
class CitationOutcome:
    """Result of citation filtering, including what actually produced the answer.

    ``prompt_messages`` is the message list of the call the returned ``answer`` came
    from — the retry's continuation when ``retry_used`` is True, the original prompt
    otherwise. Traces record this rather than the first prompt, so a debug trace can
    never show a prompt the model was not asked.
    """

    answer: str
    cited: list[SourceCitation]
    retry_used: bool
    prompt_messages: list[ChatMessage]


def finalize_citations(
    prompt_messages: Sequence[ChatMessage],
    answer: str,
    all_sources: Sequence[SourceCitation],
    generator: ChatGenerator,
    settings: AppSettings,
) -> CitationOutcome:
    """Filter ``answer`` to its cited sources, retrying once if none are cited.

    Shared by both response modes (sync ``generate_grounded_answer`` and the streaming
    path in ``RagPipeline.answer_stream``) so the retry-trigger condition can't drift
    between them. ``prompt_messages`` is the prompt that produced ``answer``; the
    returned outcome carries whichever prompt produced its final answer.

    The ``not cited`` guard is redundant with ``needs_citation_retry`` — that returns
    False whenever the answer contains any ``[n]`` marker, and ``cited`` is only
    non-empty when a marker matched — so it never changes the outcome. It's kept as a
    cheap, explicit statement of the intent ("only retry when we ended up with zero
    sources"). Note the deliberate gap it does *not* close: an answer citing ``[9]``
    when only 3 sources exist yields empty ``cited`` but no retry, because the marker
    check in ``needs_citation_retry`` short-circuits first. Retrying there would mean
    re-prompting a model that did cite, just badly, so it's left alone.
    """

    cited = cited_sources(answer, all_sources)
    final_messages = list(prompt_messages)
    retry_used = False
    if (
        not cited
        and settings.citation_retry_enabled
        and needs_citation_retry(answer, len(all_sources))
    ):
        retried = retry_uncited_answer(prompt_messages, answer, generator, settings)
        if retried is not None:
            answer, final_messages = retried
            cited = cited_sources(answer, all_sources)
            retry_used = True
    return CitationOutcome(
        answer=answer, cited=cited, retry_used=retry_used, prompt_messages=final_messages
    )


def build_expansion_messages(question: str, count: int) -> list[ChatMessage]:
    """Build the prompt that asks for ``count`` alternative query phrasings."""

    return [
        {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Query: {question}\n\n"
                f"Write {count} alternative search queries, one per line:"
            ),
        },
    ]


def generate_query_variants(
    question: str,
    count: int,
    generator: ChatGenerator,
    settings: AppSettings,
) -> list[str]:
    """Generate up to ``count`` alternative query phrasings for multi-query retrieval.

    Best-effort: any provider error or empty output returns ``[]`` so retrieval falls
    back to the single original query (mirrors ``condense_question``'s contract).
    Deterministic-ish temperature 0.3 for mild variation.
    """

    if count <= 0:
        return []
    expansion_settings = settings.model_copy(
        update={
            "llm_temperature": 0.3,
            "llm_max_tokens": int(settings.query_expansion_max_tokens),
        }
    )
    try:
        raw = generator.complete(build_expansion_messages(question, count), expansion_settings)
    except GenerationError:
        return []

    original = question.strip().lower()
    seen: set[str] = set()
    variants: list[str] = []
    for line in raw.splitlines():
        cleaned = _strip_list_marker(line.strip()).strip("\"'").strip()
        key = cleaned.lower()
        if not cleaned or key == original or key in seen:
            continue
        seen.add(key)
        variants.append(cleaned)
    return variants[:count]


def supports_reasoning(model: str) -> bool:
    """True when LiteLLM reports ``model`` as a reasoning model (gpt-5 family, o-series).

    Wrapped so an unknown model or an offline metadata lookup degrades to False rather
    than raising — non-reasoning is the safe default (plain max_tokens, no effort knob).
    """

    try:
        return bool(litellm.supports_reasoning(model))
    except Exception:
        return False


def effective_max_tokens(settings: AppSettings, model: str) -> int:
    """Output-token budget for ``model``.

    Reasoning models spend hidden reasoning tokens out of the same budget as the visible
    answer, so they get ``llm_max_tokens`` plus ``reasoning_token_headroom`` — otherwise
    reasoning alone exhausts a small budget and the visible answer comes back empty. The
    additive form preserves the relative sizing of the cheaper condense/expansion calls,
    which lower ``llm_max_tokens`` via ``model_copy``.
    """

    base = int(settings.llm_max_tokens)
    if supports_reasoning(model):
        return base + int(settings.reasoning_token_headroom)
    return base


def completion_model_and_kwargs(settings: AppSettings) -> tuple[str, dict[str, str]]:
    """Return LiteLLM model name and provider-specific keyword arguments."""

    provider = settings.llm_provider.lower().strip()
    if provider == "ollama":
        return f"ollama/{settings.llm_model}", {
            "base_url": settings.ollama_base_url,
            "keep_alive": settings.ollama_keep_alive,
        }
    if provider == "openai":
        if not settings.openai_api_key:
            raise GenerationConfigError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        kwargs = {"api_key": settings.openai_api_key}
        effort = settings.openai_reasoning_effort.strip()
        if effort and supports_reasoning(settings.openai_model):
            kwargs["reasoning_effort"] = effort
        return settings.openai_model, kwargs
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


def extract_delta_text(chunk: object) -> str:
    """Extract the incremental assistant text from one streamed LiteLLM chunk.

    Returns "" for chunks that carry no content delta (e.g. the final chunk, or a
    role-only opening chunk) rather than raising — those are a normal part of a
    stream, unlike a malformed non-streaming response.
    """

    choices = read_value(chunk, "choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        return ""
    delta = read_value(choices[0], "delta")
    content = read_value(delta, "content")
    return content if isinstance(content, str) else ""


def read_value(source: object, key: str) -> object:
    """Read an attribute or dictionary key from a response object."""

    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)

