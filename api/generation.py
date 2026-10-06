"""Grounded answer generation.

The LLM client (``api.llm``), prompt building (``api.prompting``) and citation handling
(``api.citations``) live in their own modules; their public names are re-exported here,
where callers have always imported them from.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from api.citations import (
    CITATION_EXCERPT_MAX_CHARS,
    CITATION_RETRY_REMINDER,
    CitationOutcome,
    SourceCitation,
    cited_numbers,
    cited_sources,
    finalize_citations,
    needs_citation_retry,
    retry_uncited_answer,
    source_citations,
    will_retry_citations,
)
from api.llm import (
    ChatGenerator,
    ChatMessage,
    CompletionClient,
    GenerationConfigError,
    GenerationError,
    LiteLLMGenerator,
    completion_model_and_kwargs,
    effective_max_tokens,
    extract_completion_text,
    extract_delta_text,
    read_value,
    redact_url,
    supports_reasoning,
)
from api.prompting import (
    CONDENSE_SYSTEM_PROMPT,
    EXPANSION_SYSTEM_PROMPT,
    INSUFFICIENT_CONTEXT_ANSWER,
    FittedPrompt,
    build_condense_messages,
    build_expansion_messages,
    build_grounded_messages,
    condense_question,
    context_window_tokens,
    estimate_prompt_tokens,
    fit_prompt_to_context_window,
    format_context_chunk,
    generate_query_variants,
    needs_condense,
)
from api.reranking import RerankedChunk
from api.retrieval import RetrievalError
from api.settings import AppSettings

__all__ = [
    "CITATION_EXCERPT_MAX_CHARS",
    "CITATION_RETRY_REMINDER",
    "CONDENSE_SYSTEM_PROMPT",
    "ChatGenerator",
    "ChatMessage",
    "CitationOutcome",
    "CompletionClient",
    "EXPANSION_SYSTEM_PROMPT",
    "FittedPrompt",
    "GenerationConfigError",
    "GenerationError",
    "GroundedAnswer",
    "INSUFFICIENT_CONTEXT_ANSWER",
    "LiteLLMGenerator",
    "SourceCitation",
    "StageTimings",
    "build_condense_messages",
    "build_expansion_messages",
    "build_grounded_messages",
    "cited_numbers",
    "cited_sources",
    "completion_model_and_kwargs",
    "condense_question",
    "context_window_tokens",
    "effective_max_tokens",
    "estimate_prompt_tokens",
    "extract_completion_text",
    "extract_delta_text",
    "finalize_citations",
    "fit_prompt_to_context_window",
    "format_context_chunk",
    "generate_grounded_answer",
    "generate_query_variants",
    "needs_citation_retry",
    "needs_condense",
    "read_value",
    "redact_url",
    "retry_uncited_answer",
    "source_citations",
    "supports_reasoning",
    "will_retry_citations",
]


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
    # The zero-citation retry's own LLM round trip (see finalize_citations). 0.0 when
    # it didn't fire. Held apart from generate_ms because the two response modes used
    # to disagree about it: the sync path ran finalize_citations inside the block
    # generate_ms measures, while the streaming path ran it after generate_ms was
    # already stamped — so a retry silently inflated generate_ms in one mode and
    # vanished into the unaccounted total_ms remainder in the other. Now generate_ms
    # means "the first generation call" in both, and this is the retry.
    citation_retry_ms: float = 0.0


@dataclass(frozen=True)
class GroundedAnswer:
    """Generated answer and the sources made available to the model."""

    answer: str
    sources: list[SourceCitation]
    timings: StageTimings | None = None
    trace_id: str | None = None
    # Served from api.answer_cache rather than generated for this request.
    cached: bool = False
    # True when the answer initially lacked citations and a stricter retry supplied
    # them (see needs_citation_retry / retry_uncited_answer).
    citation_retry_used: bool = False
    # Wall-clock of that retry's LLM call, 0.0 when it didn't fire. Carried out so the
    # pipeline can subtract it from the block it measures as generate_ms — the retry
    # happens inside generate_grounded_answer, but it is not generation.
    citation_retry_ms: float = 0.0
    # The exact message list of the call that produced `answer` — the retry's
    # continuation when citation_retry_used is True. Carried out so tracing records what
    # was really sent instead of re-deriving a prompt that may not be the one used.
    # Never surfaced through the HTTP response (see api/schemas.py's QuestionResponse), only
    # through the debug trace.
    prompt_messages: list[ChatMessage] = field(default_factory=list)


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
        citation_retry_ms=outcome.retry_ms,
        prompt_messages=outcome.prompt_messages,
    )
