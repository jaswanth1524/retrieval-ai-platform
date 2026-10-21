"""Orchestration seams: query (retrieve -> rerank -> generate) and ingest (parse ->
chunk -> embed -> index), each behind a single call so handlers stay thin.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import time
import uuid
from collections.abc import Callable, Generator, Iterator, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from typing import Any, Literal, NotRequired, TypedDict

from api.answer_cache import (
    AnswerCache,
    CachedAnswer,
    CacheKey,
    answer_cache_key,
)
from api.citations import (
    answer_status,
    cited_numbers,
    declines_to_answer,
    opens_with_a_refusal,
    states_insufficiency,
    strip_citation_markers,
)

# Neighbour expansion and ingestion live in their own modules now; their names are
# re-exported here (``x as x``) so every existing ``from api.pipeline import`` works.
from api.context_expansion import _VARIANT_EXECUTOR as _VARIANT_EXECUTOR
from api.context_expansion import _clip_head as _clip_head
from api.context_expansion import _clip_tail as _clip_tail
from api.context_expansion import _map_concurrently as _map_concurrently
from api.context_expansion import _strip_overlap as _strip_overlap
from api.context_expansion import expand_with_neighbors as expand_with_neighbors
from api.diversity import select_diverse
from api.errors import public_error_message
from api.generation import (
    INSUFFICIENT_CONTEXT_ANSWER,
    ChatGenerator,
    ChatMessage,
    GenerationConfigError,
    GenerationError,
    GroundedAnswer,
    SourceCitation,
    StageTimings,
    attach_spans,
    build_grounded_messages,
    citation_from_dict,
    finalize_citations,
    fit_prompt_to_context_window,
    generate_followups,
    generate_grounded_answer,
    generate_query_variants,
    needs_condense,
    source_citations,
    will_call_llm_for_citations,
)
from api.ingest_service import DocumentIngestOutcome as DocumentIngestOutcome
from api.ingest_service import IngestService as IngestService
from api.llm import (
    GenerationSlot,
    begin_usage,
    expected_prefill_ms,
    reset_finish_reason,
    stopped_at_token_limit,
    usage_totals,
)
from api.prompting import (
    clip_text,
    condense_question_detailed,
    estimate_prompt_tokens,
    one_line,
    with_answer_language,
)
from api.qdrant_schema import check_dense_dimension
from api.repository import PassageFilter, VectorRepository, reciprocal_rank_fusion
from api.request_id import current_request_id, submit_in_context
from api.reranking import (
    RerankedChunk,
    Reranker,
    RerankOutcome,
    relative_cut,
    rerank_candidates_detailed,
)
from api.retrieval import (
    QueryEmbeddingProvider,
    RetrievalConfigError,
    RetrievalError,
    RetrievedChunk,
    embed_query,
    points_to_chunks,
)
from api.settings import AppSettings
from api.tracing import (
    QueryTrace,
    TraceConfig,
    TraceMode,
    TraceSink,
    TraceStatus,
    build_trace_candidates,
)
from api.workspaces import workspace_of

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnswerOverrides:
    """Optional per-request overrides for one ``RagPipeline.answer`` call.

    Defined here rather than in ``api.schemas`` so the pipeline layer never depends on
    the FastAPI request layer. ``rrf_k``, ``fused_top_n``, and the dense/sparse
    retrieval limits are intentionally not overridable here — hybrid retrieval and
    RRF k=60 are non-negotiable per CLAUDE.md.
    """

    llm_provider: str | None = None
    rerank_top_k: int | None = None
    max_context_chunks: int | None = None
    llm_temperature: float | None = None


@dataclass(frozen=True)
class RetrievalPhase:
    """Result of retrieve -> rerank -> neighbor-expand, kept as a single value so
    trace capture (see api.tracing) can see the full fused candidate list and every
    scored chunk, not just the ones that survived to generation.
    """

    context_chunks: list[RerankedChunk]
    fused_candidates: list[RetrievedChunk]
    scored_chunks: list[RerankedChunk]
    embed_ms: float
    search_ms: float
    rerank_ms: float
    # Query-variant generation (before embedding) and neighbour expansion (after
    # rerank) are separate stages at opposite ends of retrieval; they are timed apart
    # so a latency breakdown can't attribute one to the other.
    query_expansion_ms: float = 0.0
    context_expansion_ms: float = 0.0
    # Part of rerank_ms spent waiting for the reranker (another question was scoring).
    rerank_wait_ms: float = 0.0
    diversity_dropped_ids: frozenset[str] = frozenset()
    # Cleared the floor, cut for scoring far below the best (RERANK_RELATIVE_FLOOR).
    relative_cut_ids: frozenset[str] = frozenset()
    query_variants: list[str] = field(default_factory=list)
    # Search only: nothing cleared rerank_min_score, so context_chunks are the closest
    # passages instead (see _retrieve_and_rerank).
    below_min_score: bool = False


class SourcesEvent(TypedDict):
    """SSE event: citation metadata for the selected context, sent before generation.

    Carries ``trace_id`` (None when tracing is disabled) so the UI can link this turn
    to its debug trace even if generation fails before the ``done`` event is reached.
    """

    type: Literal["sources"]
    sources: list[dict[str, Any]]
    trace_id: str | None


class StageEvent(TypedDict):
    """SSE event: which pipeline stage just started, for the UI's progress indicator.

    Emitted *before* the stage runs, not after — the whole point is telling the reader
    what the server is doing during the seconds before any token exists. Retrieval can
    take tens of seconds on a whole-corpus question, and the UI previously showed one
    static label for all of it.

    ``stage`` is a free-form lowercase label rather than an enum: a client that doesn't
    recognise one must fall back to its own default, so adding a stage later can never
    break an older frontend.
    """

    type: Literal["stage"]
    stage: str
    # Progress through a multi-step stage (a summary's parts): step ``step`` of
    # ``steps``, both 1-based. Absent on single-step stages.
    step: NotRequired[int]
    steps: NotRequired[int]
    # "reading": the prompt the model is about to read, in estimated tokens, and how
    # long the local model should take to read it at its measured rate (absent before
    # one is measured, and for hosted providers).
    prompt_tokens: NotRequired[int]
    eta_ms: NotRequired[float]


class DeltaEvent(TypedDict):
    """SSE event: one incremental fragment of the generated answer."""

    type: Literal["delta"]
    text: str


class FollowupsEvent(TypedDict):
    """SSE event after ``done`` (FOLLOWUPS_ENABLED): questions to suggest next.

    After ``done``, not inside it, so the answer and its citations are final while the
    extra LLM call runs; a client that doesn't know the frame ignores it.
    """

    type: Literal["followups"]
    questions: list[str]


class DoneEvent(TypedDict):
    """SSE event: the final answer, cited-only sources, and stage timings."""

    type: Literal["done"]
    answer: str
    sources: list[dict[str, Any]]
    timings: dict[str, float]
    trace_id: str | None
    # Present (True) only when the answer came from the answer cache.
    cached: NotRequired[bool]
    # citations.answer_status, and the best cross-encoder score among the candidates.
    answer_status: NotRequired[str]
    top_rerank_score: NotRequired[float | None]
    # Answered from below-the-bar passages (RetrievalPhase.below_min_score).
    weak_matches: NotRequired[bool]
    # The provider cut the answer off at the token limit.
    truncated: NotRequired[bool]
    # The answer's citations were added by matching it to the sources (backfill).
    citation_backfilled: NotRequired[bool]
    usage: NotRequired[dict[str, int] | None]
    # Summaries only: passages read, of the passages the documents hold. Fewer than
    # the total when a long document was read at evenly spaced passages.
    chunks_read: NotRequired[int]
    chunks_total: NotRequired[int]


StreamEvent = StageEvent | SourcesEvent | DeltaEvent | DoneEvent | FollowupsEvent


def _cited_chunks(
    cited: Sequence[SourceCitation], selected: Sequence[RerankedChunk]
) -> list[RerankedChunk]:
    """The prompt chunks an answer cites (source ``n`` is ``selected[n - 1]``)."""

    return [
        selected[source.source_number - 1]
        for source in cited
        if 0 < source.source_number <= len(selected)
    ]


def _without_unanswered_exchanges(history: list[ChatMessage]) -> list[ChatMessage]:
    """``history`` minus every exchange whose answer was "not enough information".

    Sent back, a refusal primed the next question twice over: the condense step folded
    it into the search query ("What if they miss it?" became "What are the consequences
    if they miss the key points?") and the answering model read it as the
    conversation's own conclusion and refused again. A refusal that cites the sources
    it looked at counts too (``declines_to_answer``); an answer that says something is
    missing but gives something else from a source answered part of it, and stays.
    """

    kept: list[ChatMessage] = []
    for message in history:
        content = message["content"]
        if message["role"] == "assistant" and (
            (
                states_insufficiency(content)
                and not cited_numbers(content, _HISTORY_CITATION_CHECK_MAX)
            )
            or declines_to_answer(content)
        ):
            if kept and kept[-1]["role"] == "user":
                kept.pop()
            continue
        kept.append(message)
    return kept


# Only whether an earlier answer cites anything matters, so ranges are clamped small.
_HISTORY_CITATION_CHECK_MAX = 1000


def _clip_earlier_answers(history: list[ChatMessage], settings: AppSettings) -> list[ChatMessage]:
    """``history`` with each earlier answer cut to GENERATION_HISTORY_ASSISTANT_CHARS
    (0: unchanged). The answering model reads all of it before the first token."""

    limit = int(settings.generation_history_assistant_chars)
    if limit <= 0:
        return history
    return [
        {"role": "assistant", "content": clip_text(message["content"], limit)}
        if message["role"] == "assistant"
        else message
        for message in history
    ]


def _with_interpretation(question: str, condensed: str | None) -> str:
    """The question as the answering prompt shows it: a follow-up that was rewritten for
    search carries the rewrite, so a small model reading "Explain in more detail" next
    to sources about vacation days knows what "more" is about."""

    if not condensed or condensed.casefold() == question.casefold():
        return question
    return f"{question}\n(Read with the conversation as: {one_line(condensed)})"


def _condense_skip_reason(question: str, settings: AppSettings) -> str | None:
    """Why a question with history is not condensed, or None when it is. The one copy
    of the gate: ``_will_condense`` announces the stage from it, ``_condense_query``
    runs (or skips) the call from it."""

    if not settings.conversation_condense_enabled:
        return "disabled"
    if not needs_condense(question):
        return "standalone"
    return None


def _trace_status(status: str) -> TraceStatus:
    """A finished answer's trace status: a model that said the sources don't answer it
    is "insufficient_context" too, not "ok" (the query log counts those as uncovered)."""

    return "insufficient_context" if status in ("no_context", "insufficient_context") else "ok"


def _top_rerank_score(phase: RetrievalPhase) -> float | None:
    """The best cross-encoder score among the scored candidates, kept or not."""

    return phase.scored_chunks[0].rerank_score if phase.scored_chunks else None


def _cached_timings(started: float) -> StageTimings:
    """Timings for an answer served from the cache: no stage ran."""

    return StageTimings(
        embed_ms=0.0,
        search_ms=0.0,
        rerank_ms=0.0,
        generate_ms=0.0,
        total_ms=(time.monotonic() - started) * 1000,
    )


def _citation_dict(source: SourceCitation) -> dict[str, Any]:
    # Every field, spans included (as lists of dicts), as CitationResponse has them.
    return dataclasses.asdict(source)


def usage_dict(usage: tuple[int, int] | None) -> dict[str, int] | None:
    """Token usage as the ``done`` frame and responses carry it."""

    if usage is None:
        return None
    return {"prompt_tokens": usage[0], "completion_tokens": usage[1]}


def usage_tuple(usage: object) -> tuple[int, int] | None:
    """``usage_dict``'s inverse (a ``done`` frame's usage back to the pair)."""

    if not isinstance(usage, dict):
        return None
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if isinstance(prompt, int) and isinstance(completion, int):
        return prompt, completion
    return None


def timings_dict(timings: StageTimings) -> dict[str, float]:
    # Every StageTimings field, so a new stage reaches traces and the done frame without
    # another hand-kept list (TimingsResponse is tested to have the same fields).
    return dataclasses.asdict(timings)


class QuestionAbandoned(Exception):
    """The client left mid-question; stop before the next expensive stage."""


class RagPipeline:
    """Composes retrieval, reranking, and generation behind a single call."""

    def __init__(
        self,
        repository: VectorRepository,
        embedding_provider: QueryEmbeddingProvider,
        reranker: Reranker,
        generator: ChatGenerator,
        settings: AppSettings,
        trace_store: TraceSink | None = None,
        answer_cache: AnswerCache | None = None,
    ) -> None:
        self._repository = repository
        # None disables answer caching (ANSWER_CACHE_SIZE=0).
        self._answer_cache = answer_cache
        self._embedding_provider = embedding_provider
        self._reranker = reranker
        self._generator = generator
        self._settings = settings
        # None disables trace capture entirely (the default) — every trace-related
        # call below is skipped rather than merely producing an empty trace, so
        # tracing costs nothing when the operator has turned it off.
        self._trace_store = trace_store

    def _effective_settings(self, overrides: AnswerOverrides) -> AppSettings:
        """Fold per-request overrides into a settings copy without mutating the base."""

        settings = self._settings
        if overrides.rerank_top_k is not None and overrides.rerank_top_k > int(
            settings.fused_top_n
        ):
            raise RetrievalConfigError(
                f"rerank_top_k ({overrides.rerank_top_k}) cannot exceed the server's "
                f"fused_top_n ({settings.fused_top_n})."
            )

        update: dict[str, Any] = {
            key: value
            for key, value in {
                "llm_provider": overrides.llm_provider,
                "rerank_top_k": overrides.rerank_top_k,
                "max_context_chunks": overrides.max_context_chunks,
                "llm_temperature": overrides.llm_temperature,
            }.items()
            if value is not None
        }
        return settings.model_copy(update=update) if update else settings

    def _trace_config(
        self,
        effective_settings: AppSettings,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None = None,
    ) -> TraceConfig:
        return TraceConfig(
            llm_provider=effective_settings.llm_provider,
            rerank_top_k=int(effective_settings.rerank_top_k),
            max_context_chunks=int(effective_settings.max_context_chunks),
            llm_temperature=float(effective_settings.llm_temperature),
            rerank_min_score=float(effective_settings.rerank_min_score),
            fused_top_n=int(effective_settings.fused_top_n),
            filenames=list(filenames) if filenames else None,
            tags=list(tags) if tags else None,
            reranker_model=effective_settings.reranker_model,
        )

    def _phase_timings(
        self,
        phase: RetrievalPhase,
        *,
        total_start: float,
        condense_ms: float,
        generate_ms: float = 0.0,
        citation_retry_ms: float = 0.0,
        first_token_ms: float = 0.0,
    ) -> StageTimings:
        """Stage timings for a finished question; ``total_ms`` measured now."""

        return StageTimings(
            embed_ms=phase.embed_ms,
            search_ms=phase.search_ms,
            rerank_ms=phase.rerank_ms,
            generate_ms=generate_ms,
            total_ms=(time.monotonic() - total_start) * 1000,
            condense_ms=condense_ms,
            query_expansion_ms=phase.query_expansion_ms,
            context_expansion_ms=phase.context_expansion_ms,
            citation_retry_ms=citation_retry_ms,
            rerank_wait_ms=phase.rerank_wait_ms,
            first_token_ms=first_token_ms,
        )

    def _error_trace(
        self,
        *,
        trace_id: str,
        question: str,
        mode: TraceMode,
        effective_settings: AppSettings | None,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None,
        error: str,
        answer: str | None,
        condensed_question: str | None,
        history_message_count: int,
        condense_note: str | None = None,
    ) -> QueryTrace:
        """The trace of a question that failed (or was abandoned) before finishing."""

        return QueryTrace(
            trace_id=trace_id,
            created_at=time.time(),
            question=question,
            mode=mode,
            status="error",
            config=self._trace_config(effective_settings, filenames, tags)
            if effective_settings is not None
            else None,
            candidates=[],
            prompt_messages=None,
            answer=answer,
            cited_source_numbers=[],
            timings=None,
            error=error,
            condensed_question=condensed_question,
            condense_note=condense_note,
            history_message_count=history_message_count,
            request_id=current_request_id(),
            workspace=workspace_of(self._settings),
        )

    def _finished_trace(
        self,
        *,
        trace_id: str,
        question: str,
        mode: TraceMode,
        status: TraceStatus,
        effective_settings: AppSettings,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None,
        phase: RetrievalPhase,
        selected: Sequence[RerankedChunk],
        context_budget_dropped_ids: AbstractSet[str] = frozenset(),
        prompt_messages: list[ChatMessage] | None,
        answer: str,
        cited_source_numbers: list[int],
        timings: StageTimings,
        condensed_question: str | None,
        history_message_count: int,
        citation_retry_used: bool = False,
        usage: tuple[int, int] | None = None,
        condense_note: str | None = None,
        citation_backfilled: bool = False,
    ) -> QueryTrace:
        """The trace of a question that ran to an answer (or to "insufficient context")."""

        return QueryTrace(
            trace_id=trace_id,
            created_at=time.time(),
            question=question,
            mode=mode,
            status=status,
            config=self._trace_config(effective_settings, filenames, tags),
            candidates=build_trace_candidates(
                phase.fused_candidates,
                phase.scored_chunks,
                phase.context_chunks,
                selected,
                min_score=float(effective_settings.rerank_min_score),
                diversity_dropped_ids=phase.diversity_dropped_ids,
                relative_cut_ids=phase.relative_cut_ids,
                context_budget_dropped_ids=context_budget_dropped_ids,
            ),
            prompt_messages=prompt_messages,
            answer=answer,
            cited_source_numbers=cited_source_numbers,
            timings=timings_dict(timings),
            error=None,
            condensed_question=condensed_question,
            condense_note=condense_note,
            weak_matches=phase.below_min_score,
            history_message_count=history_message_count,
            query_variants=phase.query_variants,
            citation_retry_used=citation_retry_used,
            citation_backfilled=citation_backfilled,
            request_id=current_request_id(),
            workspace=workspace_of(self._settings),
            usage=usage,
        )

    def _truncate_history(
        self, history: Sequence[ChatMessage] | None, effective_settings: AppSettings
    ) -> list[ChatMessage]:
        """Keep only the most recent history messages, per server-side config.

        Earlier answers lose their ``[n]`` markers: those numbered the previous turn's
        sources, and this turn's prompt numbers its own from 1 again, so a model
        continuing the conversation could copy a stale ``[2]`` onto an unrelated source.
        An exchange whose answer was "not enough information" is left out (see
        ``_without_unanswered_exchanges``).
        """

        if not history:
            return []
        cap = int(effective_settings.conversation_max_history_messages)
        kept = _without_unanswered_exchanges(list(history)[-cap:] if cap > 0 else [])
        return [
            {
                "role": "assistant",
                # A bare "[1]" answer still holds its place, so the turns keep alternating.
                "content": strip_citation_markers(message["content"]) or "(answer omitted)",
            }
            if message["role"] == "assistant"
            else message
            for message in kept
        ]

    def _will_condense(
        self,
        question: str,
        truncated_history: Sequence[ChatMessage],
        effective_settings: AppSettings,
    ) -> bool:
        """Whether ``_condense_query`` will actually make its LLM call.

        Split out purely so ``answer_stream`` can announce the ``condensing`` stage
        *before* the call blocks it, without restating the gate — two copies of this
        condition would drift the first time either side changed.
        """

        return bool(truncated_history) and (
            _condense_skip_reason(question, effective_settings) is None
        )

    def _condense_query(
        self,
        question: str,
        truncated_history: Sequence[ChatMessage],
        effective_settings: AppSettings,
    ) -> tuple[str, float, str | None, str | None]:
        """Rewrite a follow-up into a standalone retrieval query, when history exists.

        ``truncated_history`` is expected to already be capped (see
        ``_truncate_history``) — the same truncated list is what generation sees.
        Returns ``(retrieval_query, condense_ms, condensed_question, note)`` — the third
        element is the trace-visible rewrite, or None when condense was skipped
        (no history, or disabled) or failed/returned nothing usable (falls back to the
        raw question); ``note`` says why for the trace. A condense failure must never
        fail the whole request — retrieval just proceeds on the raw question, exactly
        as it did before this feature.
        """

        if not truncated_history:
            return question, 0.0, None, None
        skip_reason = _condense_skip_reason(question, effective_settings)
        if skip_reason is not None:
            return question, 0.0, None, skip_reason

        condense_start = time.monotonic()
        try:
            condensed = condense_question_detailed(
                question, truncated_history, self._generator, effective_settings
            )
        except GenerationError as exc:
            logger.warning("Condense step failed, falling back to raw question: %s", exc)
            return question, (time.monotonic() - condense_start) * 1000, None, "failed"
        condense_ms = (time.monotonic() - condense_start) * 1000
        if (
            condensed.text
            and not condensed.text.rstrip().endswith("?")
            and opens_with_a_refusal(condensed.text)
        ):
            # The model answered from the history ("The documents do not contain ...")
            # instead of rewriting; searched for, it matches nothing. A question that
            # merely uses the words ("What if a receipt does not contain a date?") is
            # a rewrite.
            condensed = replace(condensed, text="", note="rejected: not a question")
        if not condensed.text:
            logger.warning(
                "Condense step returned nothing usable (%s), falling back to raw question.",
                condensed.note,
            )
            return question, condense_ms, None, condensed.note
        return condensed.text, condense_ms, condensed.text, condensed.note

    def retrieval_query(self, question: str, history: Sequence[ChatMessage] | None = None) -> str:
        """The query a question is retrieved with: a follow-up rewritten into a
        standalone one, as ``answer`` does (the eval harness scores this)."""

        settings = self._effective_settings(AnswerOverrides())
        truncated = self._truncate_history(history, settings)
        query, _, _, _ = self._condense_query(question.strip(), truncated, settings)
        return query

    def _stored_tag_variants(self, tags: Sequence[str], settings: AppSettings) -> list[str]:
        """The stored spellings of ``tags``, matched the way tags are de-duplicated.

        Tags are stored as first typed ("Finance") and de-duplicated case-insensitively,
        but the payload filter matches exactly, so asking for "finance" found nothing.
        Tags no document carries are kept as given: they still match nothing, where an
        empty list would have dropped the filter and searched every document.
        """

        wanted = {" ".join(tag.split()).casefold() for tag in tags}
        stored = self._repository.stored_tags(settings)
        return sorted(tag for tag in stored if tag.casefold() in wanted) or list(tags)

    def _retrieve_and_rerank(
        self,
        question: str,
        effective_settings: AppSettings,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None = None,
        should_stop: Callable[[], bool] | None = None,
        *,
        for_generation: bool = True,
        narrow: PassageFilter | None = None,
        also_search: str | None = None,
    ) -> RetrievalPhase:
        """Run retrieve -> rerank -> neighbor-expand.

        ``also_search`` is a second query searched alongside ``question`` and fused
        with it (the raw follow-up next to its condensed rewrite); the rerank scores
        against ``question`` alone.

        ``for_generation=False`` (search without an answer) skips the two stages that
        only serve the prompt: LLM query expansion and neighbour expansion.

        Embedding and search are separate steps so each can be timed for
        observability.
        ``should_stop`` is checked before the rerank, the costliest stage, and raises
        ``QuestionAbandoned`` when the client has gone.
        """

        normalized_query = question.strip()
        if not normalized_query:
            raise RetrievalError("Query text is required.")
        self._repository.ensure_ready(effective_settings)
        # Looked up alongside query expansion and embedding, which don't depend on it;
        # only the search below needs the stored spellings.
        tag_variants = (
            submit_in_context(
                _VARIANT_EXECUTOR,
                functools.partial(self._stored_tag_variants, tags, effective_settings),
            )
            if tags
            else None
        )

        # Multi-query expansion (opt-in): rewrite into variant phrasings, retrieve each,
        # and RRF-fuse the result lists with the same pinned rrf_k before one rerank.
        # Left un-timed (query_expansion_ms stays 0.0) when disabled, like condense_ms.
        variants: list[str] = []
        query_expansion_ms = 0.0
        if effective_settings.query_expansion_enabled and for_generation:
            query_expansion_start = time.monotonic()
            variants = generate_query_variants(
                normalized_query,
                int(effective_settings.query_expansion_count),
                self._generator,
                effective_settings,
            )
            query_expansion_ms = (time.monotonic() - query_expansion_start) * 1000
        queries = [normalized_query, *variants]
        # The raw follow-up searched next to its rewrite: a rewrite that lost or changed
        # what the user asked no longer takes the raw question's matches with it.
        extra_query = (also_search or "").strip()
        if extra_query and extra_query.casefold() not in {q.casefold() for q in queries}:
            queries.insert(1, extra_query)

        embed_start = time.monotonic()
        query_embeddings = _map_concurrently(
            lambda query: embed_query(query, self._embedding_provider), queries
        )
        embed_ms = (time.monotonic() - embed_start) * 1000
        check_dense_dimension(len(query_embeddings[0].dense), effective_settings)
        if tag_variants is not None:
            tags = tag_variants.result()

        search_start = time.monotonic()
        # Passed only when set, so a repository double without the parameter still works.
        narrowing = {"narrow": narrow} if narrow is not None else {}
        result_lists = _map_concurrently(
            lambda embedding: self._repository.hybrid_search(
                effective_settings, embedding, filenames, tags, **narrowing
            ),
            query_embeddings,
        )
        points = (
            result_lists[0]
            if len(result_lists) == 1
            else reciprocal_rank_fusion(
                result_lists,
                k=int(effective_settings.rrf_k),
                limit=int(effective_settings.fused_top_n),
            )
        )
        candidates = points_to_chunks(points)
        search_ms = (time.monotonic() - search_start) * 1000
        if should_stop is not None and should_stop():
            raise QuestionAbandoned

        rerank_start = time.monotonic()
        outcome = rerank_candidates_detailed(
            query=question,
            candidates=candidates,
            reranker=self._reranker,
            settings=effective_settings,
        )
        # A chunk next to a kept one is dropped on the promise that neighbour expansion
        # puts its text back — true only when expansion brings the whole neighbour.
        # With CONTEXT_NEIGHBOR_MAX_CHARS capping it, a well-scored passage was lost.
        diverse_outcome, diversity_dropped = select_diverse(
            outcome,
            effective_settings,
            drop_neighbours=for_generation
            and int(effective_settings.context_neighbor_max_chars) == 0,
        )
        # Nothing cleared rerank_min_score: the closest passages are used anyway,
        # flagged. A search (no answer) shows up to rerank_top_k of them; the cross-
        # encoder scores a bare keyword low against a passage that has it ("laptop":
        # 0.11 for the IT equipment policy, which it ranks first). An answer gets
        # rerank_fallback_count of them, and the prompt tells the model they match only
        # weakly, so it answers from them only when they hold the answer: the floor is
        # on one model's score scale, and a fixed one refused questions the documents
        # answer ("what are foundational models": 0.006 with MiniLM, 0.154 with jina).
        below_min_score = False
        fallback_count = min(
            int(effective_settings.rerank_top_k),
            int(effective_settings.rerank_fallback_count)
            if for_generation
            else int(effective_settings.rerank_top_k),
        )
        if fallback_count > 0 and not diverse_outcome.kept and outcome.scored:
            # rerank_top_k too: select_diverse fills up to it from the scored list, so
            # an answer got 8 below-the-bar passages instead of rerank_fallback_count.
            unfloored = effective_settings.model_copy(
                update={"rerank_min_score": 0.0, "rerank_top_k": fallback_count}
            )
            diverse_outcome, diversity_dropped = select_diverse(
                RerankOutcome(scored=outcome.scored, kept=outcome.scored[:fallback_count]),
                unfloored,
                drop_neighbours=False,
            )
            below_min_score = bool(diverse_outcome.kept)
        relative_cut_ids: set[str] = set()
        if for_generation and not below_min_score:
            trimmed, relative_cut_ids = relative_cut(
                diverse_outcome.kept,
                outcome.scored,
                float(effective_settings.rerank_relative_floor),
            )
            diverse_outcome = RerankOutcome(scored=diverse_outcome.scored, kept=trimmed)
        rerank_ms = (time.monotonic() - rerank_start) * 1000
        # Read on this thread right after the call: the reranker keeps it per thread.
        rerank_wait_ms = float(getattr(self._reranker, "last_wait_ms", 0.0))

        # Timed apart from rerank: neighbour expansion is a Qdrant round trip, not model
        # scoring, and folding it into rerank_ms hid it from every latency breakdown.
        # Only the max_context_chunks prefix reaches the prompt, so only it is expanded;
        # the tail stays in the list unexpanded because traces report it as kept.
        context_expansion_start = time.monotonic()
        context_limit = int(effective_settings.max_context_chunks)
        expanded = (
            [
                *expand_with_neighbors(
                    diverse_outcome.kept[:context_limit], self._repository, effective_settings
                ),
                *diverse_outcome.kept[context_limit:],
            ]
            if for_generation
            else list(diverse_outcome.kept)
        )
        context_expansion_ms = (time.monotonic() - context_expansion_start) * 1000

        return RetrievalPhase(
            context_chunks=expanded,
            fused_candidates=candidates,
            scored_chunks=outcome.scored,
            embed_ms=embed_ms,
            search_ms=search_ms,
            rerank_ms=rerank_ms,
            rerank_wait_ms=rerank_wait_ms,
            query_expansion_ms=query_expansion_ms,
            context_expansion_ms=context_expansion_ms,
            diversity_dropped_ids=frozenset(diversity_dropped),
            relative_cut_ids=frozenset(relative_cut_ids),
            query_variants=queries[1:],
            below_min_score=below_min_score,
        )

    def retrieve(
        self,
        question: str,
        overrides: AnswerOverrides | None = None,
        filenames: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        *,
        for_generation: bool = True,
        narrow: PassageFilter | None = None,
        also_search: str | None = None,
    ) -> RetrievalPhase:
        """Retrieval exactly as ``answer`` runs it — query expansion, hybrid search,
        rerank, diversity filter, neighbour expansion — without generating. The eval
        harness scores this, so it measures the pipeline users actually get.
        ``for_generation=False`` is ``POST /search``: the same hybrid search, RRF and
        rerank, minus the LLM query expansion and the neighbour expansion that only
        feed a prompt."""

        effective_settings = self._effective_settings(overrides or AnswerOverrides())
        return self._retrieve_and_rerank(
            question,
            effective_settings,
            filenames,
            tags,
            for_generation=for_generation,
            narrow=narrow,
            also_search=also_search,
        )

    def _cache_key(
        self,
        question: str,
        overrides: AnswerOverrides | None,
        filenames: Sequence[str] | None,
        tags: Sequence[str] | None,
        history: Sequence[ChatMessage] | None,
    ) -> CacheKey | None:
        """The answer-cache key for this request, or None when it must not be cached."""

        if self._answer_cache is None or history or not question.strip():
            return None
        try:
            settings = self._effective_settings(overrides or AnswerOverrides())
            return answer_cache_key(question, settings, filenames, tags)
        except (RetrievalConfigError, GenerationConfigError):
            # The uncached path raises it with its proper handling, error trace
            # included (a provider without its key failed here, before the trace).
            return None

    def answer(
        self,
        question: str,
        overrides: AnswerOverrides | None = None,
        filenames: Sequence[str] | None = None,
        history: Sequence[ChatMessage] | None = None,
        *,
        tags: Sequence[str] | None = None,
        use_cache: bool = True,
    ) -> GroundedAnswer:
        """``_answer_uncached``, served from the answer cache when it has this question.

        ``use_cache=False`` (Regenerate) skips the lookup but still stores the new
        answer, so the next plain ask gets the regenerated one.
        """

        started = time.monotonic()
        key = self._cache_key(question, overrides, filenames, tags, history)
        if key is not None and use_cache and self._answer_cache is not None:
            hit = self._answer_cache.get(key)
            if hit is not None:
                return GroundedAnswer(
                    answer=hit.answer,
                    sources=list(hit.sources),
                    timings=_cached_timings(started),
                    trace_id=hit.trace_id,
                    cached=True,
                    answer_status=hit.answer_status,
                    top_rerank_score=hit.top_rerank_score,
                    weak_matches=hit.weak_matches,
                    truncated=hit.truncated,
                    citation_backfilled=hit.citation_backfilled,
                    followups=list(hit.followups),
                )
        grounded = self._answer_uncached(question, overrides, filenames, history, tags)
        if key is not None and self._answer_cache is not None:
            self._answer_cache.put(
                key,
                CachedAnswer(
                    grounded.answer,
                    tuple(grounded.sources),
                    grounded.trace_id,
                    grounded.answer_status,
                    grounded.top_rerank_score,
                    grounded.truncated,
                    weak_matches=grounded.weak_matches,
                    citation_backfilled=grounded.citation_backfilled,
                    followups=tuple(grounded.followups),
                ),
            )
        return grounded

    def answer_stream(
        self,
        question: str,
        overrides: AnswerOverrides | None = None,
        filenames: Sequence[str] | None = None,
        history: Sequence[ChatMessage] | None = None,
        *,
        tags: Sequence[str] | None = None,
        use_cache: bool = True,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[StreamEvent]:
        """``_answer_stream_uncached``, replayed from the answer cache on a hit.

        ``should_stop`` (the client has gone) is polled between stages, so an
        abandoned question stops before the rerank or the LLM call instead of
        finishing work nobody will read.
        """

        started = time.monotonic()
        key = self._cache_key(question, overrides, filenames, tags, history)
        if key is not None and use_cache and self._answer_cache is not None:
            hit = self._answer_cache.get(key)
            if hit is not None:
                sources = [_citation_dict(source) for source in hit.sources]
                yield {"type": "sources", "sources": sources, "trace_id": hit.trace_id}
                yield {"type": "delta", "text": hit.answer}
                yield {
                    "type": "done",
                    "answer": hit.answer,
                    "sources": sources,
                    "timings": timings_dict(_cached_timings(started)),
                    "trace_id": hit.trace_id,
                    "cached": True,
                    "answer_status": hit.answer_status,
                    "top_rerank_score": hit.top_rerank_score,
                    "weak_matches": hit.weak_matches,
                    "truncated": hit.truncated,
                    "citation_backfilled": hit.citation_backfilled,
                }
                if hit.followups:
                    yield {"type": "followups", "questions": list(hit.followups)}
                return
        done: DoneEvent | None = None
        events = self._answer_stream_uncached(
            question, overrides, filenames, history, tags, should_stop
        )

        def store(followups: Sequence[str]) -> None:
            if key is None or done is None or self._answer_cache is None:
                return
            self._answer_cache.put(
                key,
                CachedAnswer(
                    done["answer"],
                    tuple(citation_from_dict(source) for source in done["sources"]),
                    done["trace_id"],
                    done.get("answer_status", "answered"),
                    done.get("top_rerank_score"),
                    done.get("truncated", False),
                    weak_matches=done.get("weak_matches", False),
                    citation_backfilled=done.get("citation_backfilled", False),
                    followups=tuple(followups),
                ),
            )

        try:
            for event in events:
                # Cached as soon as the answer is complete, before its frame goes out:
                # follow-up questions are generated after it, and a client that leaves
                # then (or as `done` arrives) closes this generator at the yield, so a
                # store after the loop never ran. The follow-ups, if any come, are added.
                if event["type"] == "done":
                    done = event
                    store(())
                elif event["type"] == "followups":
                    store(event["questions"])
                yield event
        finally:
            # Closing this generator (the client left) must close the inner one now: its
            # `finally` records the "closed early" trace. Left to garbage collection, it
            # ran whenever the inner generator was freed — never, while anything (a
            # tracer, a traceback) still held it.
            events.close()

    def _answer_uncached(
        self,
        question: str,
        overrides: AnswerOverrides | None = None,
        filenames: Sequence[str] | None = None,
        history: Sequence[ChatMessage] | None = None,
        tags: Sequence[str] | None = None,
    ) -> GroundedAnswer:
        """Retrieve, rerank, and generate a grounded, cited answer.

        ``overrides`` are folded into a single ``model_copy`` of the shared settings —
        the singleton is never mutated, so concurrent requests with different
        overrides stay isolated. Only ``llm_provider``, ``rerank_top_k``,
        ``max_context_chunks``, and ``llm_temperature`` can vary per request; the
        dense/sparse retrieval limits, ``fused_top_n``, and ``rrf_k`` always come from
        the base settings (embeddings and hybrid fusion stay fixed regardless).
        ``filenames``, when given, restricts retrieval to those documents. ``history``,
        when given, drives one extra condense call that rewrites ``question`` into a
        standalone retrieval query — the server itself stores no conversation state.
        """

        total_start = time.monotonic()
        trace_store = self._trace_store
        trace_id = str(uuid.uuid4()) if trace_store is not None else None
        effective_settings: AppSettings | None = None
        condensed_question: str | None = None
        condense_note: str | None = None
        truncated_history: list[ChatMessage] = []

        try:
            effective_settings = self._effective_settings(overrides or AnswerOverrides())
            truncated_history = self._truncate_history(history, effective_settings)
            retrieval_query, condense_ms, condensed_question, condense_note = self._condense_query(
                question, truncated_history, effective_settings
            )
            phase = self._retrieve_and_rerank(
                retrieval_query,
                effective_settings,
                filenames,
                tags,
                also_search=question if condensed_question else None,
            )
            prompt_question = _with_interpretation(question.strip(), condensed_question)
            fitted = fit_prompt_to_context_window(
                prompt_question,
                phase.context_chunks[: int(effective_settings.max_context_chunks)],
                _clip_earlier_answers(truncated_history, effective_settings),
                effective_settings,
                weak_matches=phase.below_min_score,
            )
            selected = fitted.context_chunks

            generate_start = time.monotonic()
            reset_finish_reason()
            begin_usage()
            grounded = generate_grounded_answer(
                query=prompt_question,
                context_chunks=selected,
                generator=self._generator,
                settings=effective_settings,
                history=fitted.history or None,
                weak_matches=phase.below_min_score,
            )
            # generate_grounded_answer runs the zero-citation retry internally, so the
            # block just timed covers two LLM calls when it fired. Split it back out so
            # generate_ms means the same thing here as it does on the streaming path.
            generate_ms = (time.monotonic() - generate_start) * 1000 - grounded.citation_retry_ms
            # The last call made the answer: the citation retry's when it fired.
            truncated = stopped_at_token_limit()
            usage = usage_totals()
            timings = self._phase_timings(
                phase,
                total_start=total_start,
                condense_ms=condense_ms,
                generate_ms=generate_ms,
                citation_retry_ms=grounded.citation_retry_ms,
            )
        except Exception as exc:
            if trace_store is not None and trace_id is not None:
                trace_store.add(
                    self._error_trace(
                        trace_id=trace_id,
                        question=question,
                        mode="sync",
                        effective_settings=effective_settings,
                        filenames=filenames,
                        tags=tags,
                        # Traces are readable with the read-only key: the same rule
                        # as a response body, so no internal detail.
                        error=public_error_message(exc),
                        answer=None,
                        condensed_question=condensed_question,
                        condense_note=condense_note,
                        history_message_count=len(truncated_history),
                    )
                )
            raise

        if trace_store is not None and trace_id is not None:
            # Captured from generation, not rebuilt here. A second
            # build_grounded_messages call would agree today but is a re-derivation that
            # can drift, and it could never reproduce the citation-retry continuation
            # that actually produced the answer when grounded.citation_retry_used is set.
            trace_store.add(
                self._finished_trace(
                    trace_id=trace_id,
                    question=question,
                    mode="sync",
                    status=_trace_status(
                        answer_status(grounded.answer, len(grounded.sources), len(selected))
                    ),
                    effective_settings=effective_settings,
                    filenames=filenames,
                    tags=tags,
                    phase=phase,
                    selected=selected,
                    context_budget_dropped_ids=fitted.dropped_point_ids,
                    prompt_messages=grounded.prompt_messages or None,
                    answer=grounded.answer,
                    cited_source_numbers=[source.source_number for source in grounded.sources],
                    timings=timings,
                    condensed_question=condensed_question,
                    condense_note=condense_note,
                    history_message_count=len(truncated_history),
                    citation_retry_used=grounded.citation_retry_used,
                    citation_backfilled=grounded.citation_backfilled,
                    usage=usage,
                )
            )
        status = answer_status(grounded.answer, len(grounded.sources), len(selected))
        followups = (
            generate_followups(
                question.strip(),
                grounded.answer,
                _cited_chunks(grounded.sources, selected),
                self._generator,
                effective_settings,
            )
            if effective_settings.followups_enabled and status == "answered"
            else []
        )
        return replace(
            grounded,
            timings=timings,
            trace_id=trace_id,
            answer_status=status,
            top_rerank_score=_top_rerank_score(phase),
            weak_matches=phase.below_min_score,
            truncated=truncated,
            followups=followups,
            usage=usage,
        )

    def _answer_stream_uncached(
        self,
        question: str,
        overrides: AnswerOverrides | None = None,
        filenames: Sequence[str] | None = None,
        history: Sequence[ChatMessage] | None = None,
        tags: Sequence[str] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Generator[StreamEvent]:
        """Stream a grounded answer as SSE-ready events: sources, deltas, then done.

        Sources are emitted immediately after rerank (before the LLM call starts) so
        the UI can render citations while the answer is still generating. ``done``
        carries the final answer, the citation-filtered source list, and timings.
        ``history``, when given, drives the same condense step as ``answer``.
        """

        total_start = time.monotonic()
        trace_store = self._trace_store
        trace_id = str(uuid.uuid4()) if trace_store is not None else None
        effective_settings: AppSettings | None = None
        parts: list[str] = []
        condensed_question: str | None = None
        condense_note: str | None = None
        truncated_history: list[ChatMessage] = []
        slot: GenerationSlot | None = None
        # Set the moment a trace has been stored (success, insufficient-context, or a
        # caught error below) — the `finally` clause uses this to detect the one case
        # none of those cover: the client disconnecting (GeneratorExit, a BaseException
        # that `except Exception` never sees) before any of them ran.
        finalized = False

        def _store_error_trace(error: str) -> None:
            if trace_store is None or trace_id is None:
                return
            trace_store.add(
                self._error_trace(
                    trace_id=trace_id,
                    question=question,
                    mode="stream",
                    effective_settings=effective_settings,
                    filenames=filenames,
                    tags=tags,
                    error=error,
                    answer="".join(parts).strip() or None,
                    condensed_question=condensed_question,
                    condense_note=condense_note,
                    history_message_count=len(truncated_history),
                )
            )

        try:
            effective_settings = self._effective_settings(overrides or AnswerOverrides())
            truncated_history = self._truncate_history(history, effective_settings)
            # Announced before the call, and only when it will actually happen — an
            # unconditional frame would report a stage that never ran for the majority
            # of questions, which is worse than no frame.
            if self._will_condense(question, truncated_history, effective_settings):
                yield {"type": "stage", "stage": "condensing"}
            retrieval_query, condense_ms, condensed_question, condense_note = self._condense_query(
                question, truncated_history, effective_settings
            )
            yield {"type": "stage", "stage": "searching"}
            phase = self._retrieve_and_rerank(
                retrieval_query,
                effective_settings,
                filenames,
                tags,
                should_stop,
                also_search=question if condensed_question else None,
            )
            prompt_question = _with_interpretation(question.strip(), condensed_question)
            fitted = fit_prompt_to_context_window(
                prompt_question,
                phase.context_chunks[: int(effective_settings.max_context_chunks)],
                _clip_earlier_answers(truncated_history, effective_settings),
                effective_settings,
                weak_matches=phase.below_min_score,
            )
            selected = fitted.context_chunks

            if not selected:
                yield {"type": "sources", "sources": [], "trace_id": trace_id}
                timings = self._phase_timings(
                    phase, total_start=total_start, condense_ms=condense_ms
                )
                if trace_store is not None and trace_id is not None:
                    trace_store.add(
                        self._finished_trace(
                            trace_id=trace_id,
                            question=question,
                            mode="stream",
                            status="insufficient_context",
                            effective_settings=effective_settings,
                            filenames=filenames,
                            tags=tags,
                            phase=phase,
                            selected=[],
                            prompt_messages=None,
                            answer=INSUFFICIENT_CONTEXT_ANSWER,
                            cited_source_numbers=[],
                            timings=timings,
                            condensed_question=condensed_question,
                            condense_note=condense_note,
                            history_message_count=len(truncated_history),
                        )
                    )
                finalized = True
                yield {
                    "type": "done",
                    "answer": INSUFFICIENT_CONTEXT_ANSWER,
                    "sources": [],
                    "timings": timings_dict(timings),
                    "trace_id": trace_id,
                    "answer_status": "no_context",
                    "top_rerank_score": _top_rerank_score(phase),
                }
                return

            all_sources = source_citations(selected)
            yield {
                "type": "sources",
                "sources": [_citation_dict(source) for source in all_sources],
                "trace_id": trace_id,
            }

            messages = with_answer_language(
                build_grounded_messages(
                    prompt_question,
                    selected,
                    fitted.history or None,
                    weak_matches=phase.below_min_score,
                ),
                effective_settings,
            )
            if should_stop is not None and should_stop():
                return  # the finally below records the trace as closed early
            # Held until the answer, its citation retry and follow-ups are done, so
            # another question's call can't slip in between them; released below.
            slot = GenerationSlot(effective_settings)
            if not slot.try_acquire():
                # Another question has the local model. Waiting inside Ollama ran this
                # request's timeout; waiting here doesn't, and the reader is told why.
                yield {"type": "stage", "stage": "waiting_for_model"}
                if not slot.acquire(should_stop):
                    return  # the finally below records the trace as closed early
            reading: StageEvent = {
                "type": "stage",
                "stage": "reading",
                "prompt_tokens": estimate_prompt_tokens(messages),
            }
            eta_ms = expected_prefill_ms(messages, effective_settings)
            if eta_ms is not None:
                reading["eta_ms"] = round(eta_ms)
            yield reading
            generate_start = time.monotonic()
            first_token_ms = 0.0
            reset_finish_reason()
            begin_usage()
            for delta in self._generator.stream(messages, effective_settings):
                if not parts:
                    first_token_ms = (time.monotonic() - total_start) * 1000
                parts.append(delta)
                yield {"type": "delta", "text": delta}
            generate_ms = (time.monotonic() - generate_start) * 1000

            answer = "".join(parts).strip()
            if not answer:
                raise GenerationError("Generation provider returned an empty answer.")

            # Streaming can't retry before the deltas already sent, and buffering would
            # destroy the streaming UX for the common case. Instead, if the streamed
            # answer carries no citations, run one non-streaming retry and swap the
            # corrected answer/sources into the `done` event — the frontend replaces the
            # streamed content with `done.answer`, so it snaps to the cited version.
            # finalize_citations is the same helper generate_grounded_answer (sync path)
            # uses, so the retry-trigger condition can't drift between response modes.
            # The retry is a whole unstreamed generation after the last delta; without
            # this frame the UI sits on its streaming label with nothing moving.
            if will_call_llm_for_citations(
                answer,
                all_sources,
                effective_settings,
                weak_matches=phase.below_min_score,
                context_chunks=selected,
            ):
                yield {"type": "stage", "stage": "verifying_citations"}
            outcome = finalize_citations(
                messages,
                answer,
                all_sources,
                self._generator,
                effective_settings,
                weak_matches=phase.below_min_score,
                context_chunks=selected,
            )
            answer = outcome.answer
            # The last call made the answer: the citation retry's when it fired.
            truncated = stopped_at_token_limit()
            usage = usage_totals()
            # Spans only now: the sources frame went out before there was an answer.
            cited = attach_spans(answer, outcome.cited, selected)
            citation_retry_used = outcome.retry_used

            timings = self._phase_timings(
                phase,
                total_start=total_start,
                condense_ms=condense_ms,
                generate_ms=generate_ms,
                citation_retry_ms=outcome.retry_ms,
                first_token_ms=first_token_ms,
            )
            if trace_store is not None and trace_id is not None:
                trace_store.add(
                    self._finished_trace(
                        trace_id=trace_id,
                        question=question,
                        mode="stream",
                        status=_trace_status(answer_status(answer, len(cited), len(selected))),
                        effective_settings=effective_settings,
                        filenames=filenames,
                        tags=tags,
                        phase=phase,
                        selected=selected,
                        context_budget_dropped_ids=fitted.dropped_point_ids,
                        # outcome.prompt_messages, not the `messages` that were streamed:
                        # when the citation retry fired, the streamed answer was replaced
                        # by the retry's, and the retry's continuation is what produced
                        # the text stored just below.
                        prompt_messages=list(outcome.prompt_messages),
                        answer=answer,
                        cited_source_numbers=[source.source_number for source in cited],
                        timings=timings,
                        condensed_question=condensed_question,
                        condense_note=condense_note,
                        history_message_count=len(truncated_history),
                        citation_retry_used=citation_retry_used,
                        citation_backfilled=outcome.backfilled,
                        usage=usage,
                    )
                )
            finalized = True
            status = answer_status(answer, len(cited), len(selected))
            yield {
                "type": "done",
                "answer": answer,
                "sources": [_citation_dict(source) for source in cited],
                "timings": timings_dict(timings),
                "trace_id": trace_id,
                "answer_status": status,
                "top_rerank_score": _top_rerank_score(phase),
                "weak_matches": phase.below_min_score,
                "truncated": truncated,
                "citation_backfilled": outcome.backfilled,
                "usage": usage_dict(usage),
            }
            if (
                effective_settings.followups_enabled
                and status == "answered"
                and not (should_stop is not None and should_stop())
            ):
                followups = generate_followups(
                    question.strip(),
                    answer,
                    _cited_chunks(cited, selected),
                    self._generator,
                    effective_settings,
                )
                if followups:
                    yield {"type": "followups", "questions": followups}
        except QuestionAbandoned:
            return  # the finally below records the trace as closed early
        except Exception as exc:
            if not finalized:
                _store_error_trace(public_error_message(exc))
                finalized = True
            raise
        finally:
            if not finalized:
                # Reached via GeneratorExit (the client disconnected, or the generator
                # was otherwise closed mid-stream without completing or raising an
                # `Exception`) — `finally` re-raises it automatically once this block
                # returns, so no explicit `raise` is needed here — or via an early
                # return once `should_stop` reported the client gone.
                _store_error_trace("Stream closed before completion.")
                finalized = True
            if slot is not None:
                slot.release()
