"""Orchestration seams: query (retrieve -> rerank -> generate) and ingest (parse ->
chunk -> embed -> index), each behind a single call so handlers stay thin.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypedDict

from api.chunking import HeuristicTokenCounter, TokenCounter
from api.diversity import select_diverse
from api.documents import CHUNKER_VERSION, chunk_sections, parse_document_bytes
from api.generation import (
    INSUFFICIENT_CONTEXT_ANSWER,
    ChatGenerator,
    ChatMessage,
    GenerationError,
    GroundedAnswer,
    SourceCitation,
    StageTimings,
    build_grounded_messages,
    condense_question,
    finalize_citations,
    generate_grounded_answer,
    generate_query_variants,
    needs_condense,
    source_citations,
)
from api.ingestion import EmbeddingProvider as IngestEmbeddingProvider
from api.ingestion import IngestResult, ingest_chunks
from api.repository import VectorRepository, reciprocal_rank_fusion
from api.reranking import RerankedChunk, Reranker, rerank_candidates_detailed
from api.retrieval import (
    QueryEmbeddingProvider,
    RetrievalConfigError,
    RetrievalError,
    RetrievedChunk,
    embed_query,
    points_to_chunks,
)
from api.settings import AppSettings
from api.tracing import QueryTrace, TraceConfig, TraceSink, build_trace_candidates

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
    diversity_dropped_ids: frozenset[str] = frozenset()
    query_variants: list[str] = field(default_factory=list)


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


class DeltaEvent(TypedDict):
    """SSE event: one incremental fragment of the generated answer."""

    type: Literal["delta"]
    text: str


class DoneEvent(TypedDict):
    """SSE event: the final answer, cited-only sources, and stage timings."""

    type: Literal["done"]
    answer: str
    sources: list[dict[str, Any]]
    timings: dict[str, float]
    trace_id: str | None


StreamEvent = StageEvent | SourcesEvent | DeltaEvent | DoneEvent


def _citation_dict(source: SourceCitation) -> dict[str, Any]:
    return {
        "source_number": source.source_number,
        "filename": source.filename,
        "page": source.page,
        "section": source.section,
        "chunk_id": source.chunk_id,
        "text": source.text,
    }


def _timings_dict(timings: StageTimings) -> dict[str, float]:
    return {
        "embed_ms": timings.embed_ms,
        "search_ms": timings.search_ms,
        "rerank_ms": timings.rerank_ms,
        "generate_ms": timings.generate_ms,
        "total_ms": timings.total_ms,
        "condense_ms": timings.condense_ms,
        "query_expansion_ms": timings.query_expansion_ms,
        "context_expansion_ms": timings.context_expansion_ms,
        "citation_retry_ms": timings.citation_retry_ms,
    }


def expand_with_neighbors(
    chunks: Sequence[RerankedChunk],
    repository: VectorRepository,
    settings: AppSettings,
) -> list[RerankedChunk]:
    """Expand each reranked chunk's generation context with its immediate neighbors.

    Small-to-big retrieval: rerank on tight chunks, but let the model generate from a
    wider window around each one. Chunks with no recorded ``chunk_ordinal`` (indexed
    before this field existed) are returned unchanged rather than erroring.

    Neighbours are fetched **once per distinct filename**, not once per chunk: the
    surviving chunks are usually a handful drawn from one or two documents, so a
    per-chunk round trip meant up to ``rerank_top_k`` sequential Qdrant scrolls on every
    question for data one query per file could return.
    """

    radius = int(settings.context_neighbor_radius)
    if radius <= 0:
        return list(chunks)

    def wanted_ordinals(chunk: RerankedChunk) -> list[int]:
        assert chunk.chunk_ordinal is not None
        return [
            chunk.chunk_ordinal + delta
            for delta in range(-radius, radius + 1)
            if delta != 0 and chunk.chunk_ordinal + delta >= 1
        ]

    # Union every chunk's neighbour ordinals per filename, then one fetch per filename.
    ordinals_by_file: dict[str, set[int]] = {}
    for chunk in chunks:
        if chunk.chunk_ordinal is None:
            continue
        ordinals_by_file.setdefault(chunk.filename, set()).update(wanted_ordinals(chunk))

    # (filename, ordinal) -> text, so each chunk can pick its own neighbours back out.
    neighbor_text: dict[tuple[str, int], str] = {}
    for filename, ordinals in ordinals_by_file.items():
        for point in repository.fetch_neighbors(settings, filename, sorted(ordinals)):
            payload = point.payload or {}
            ordinal = payload.get("chunk_ordinal")
            text = payload.get("text")
            if isinstance(ordinal, int) and not isinstance(ordinal, bool) and text:
                neighbor_text[(filename, ordinal)] = str(text)

    expanded: list[RerankedChunk] = []
    for chunk in chunks:
        if chunk.chunk_ordinal is None:
            expanded.append(chunk)
            continue
        neighbor_texts = sorted(
            (ordinal, neighbor_text[(chunk.filename, ordinal)])
            for ordinal in wanted_ordinals(chunk)
            if (chunk.filename, ordinal) in neighbor_text
        )
        if not neighbor_texts:
            expanded.append(chunk)
            continue

        merged_parts: list[str] = []
        inserted_self = False
        for ordinal_value, text in neighbor_texts:
            if not inserted_self and ordinal_value > chunk.chunk_ordinal:
                merged_parts.append(chunk.text)
                inserted_self = True
            merged_parts.append(str(text))
        if not inserted_self:
            merged_parts.append(chunk.text)
        expanded.append(replace(chunk, expanded_text="\n\n".join(merged_parts)))
    return expanded


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
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._reranker = reranker
        self._generator = generator
        self._settings = settings
        # None disables trace capture entirely (the default) — every trace-related
        # call below is skipped rather than merely producing an empty trace, so
        # tracing costs nothing when the operator has turned it off.
        self._trace_store = trace_store

    def _effective_settings(
        self, overrides: AnswerOverrides
    ) -> AppSettings:
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
        self, effective_settings: AppSettings, filenames: Sequence[str] | None
    ) -> TraceConfig:
        return TraceConfig(
            llm_provider=effective_settings.llm_provider,
            rerank_top_k=int(effective_settings.rerank_top_k),
            max_context_chunks=int(effective_settings.max_context_chunks),
            llm_temperature=float(effective_settings.llm_temperature),
            rerank_min_score=float(effective_settings.rerank_min_score),
            fused_top_n=int(effective_settings.fused_top_n),
            filenames=list(filenames) if filenames else None,
        )

    def _truncate_history(
        self, history: Sequence[ChatMessage] | None, effective_settings: AppSettings
    ) -> list[ChatMessage]:
        """Keep only the most recent history messages, per server-side config."""

        if not history:
            return []
        cap = int(effective_settings.conversation_max_history_messages)
        return list(history)[-cap:] if cap > 0 else []

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

        if not truncated_history or not effective_settings.conversation_condense_enabled:
            return False
        return needs_condense(question)

    def _condense_query(
        self,
        question: str,
        truncated_history: Sequence[ChatMessage],
        effective_settings: AppSettings,
    ) -> tuple[str, float, str | None]:
        """Rewrite a follow-up into a standalone retrieval query, when history exists.

        ``truncated_history`` is expected to already be capped (see
        ``_truncate_history``) — the same truncated list is what generation sees.
        Returns ``(retrieval_query, condense_ms, condensed_question)`` — the third
        element is the trace-visible rewrite, or None when condense was skipped
        (no history, or disabled) or failed/returned empty (falls back to the raw
        question). A condense failure must never fail the whole request — retrieval
        just proceeds on the raw question, exactly as it did before this feature.
        """

        if not self._will_condense(question, truncated_history, effective_settings):
            return question, 0.0, None

        condense_start = time.monotonic()
        try:
            condensed = condense_question(
                question, truncated_history, self._generator, effective_settings
            )
        except GenerationError as exc:
            logger.warning("Condense step failed, falling back to raw question: %s", exc)
            return question, (time.monotonic() - condense_start) * 1000, None
        condense_ms = (time.monotonic() - condense_start) * 1000
        if not condensed:
            logger.warning("Condense step returned an empty result, falling back to raw question.")
            return question, condense_ms, None
        return condensed, condense_ms, condensed

    def _retrieve_and_rerank(
        self,
        question: str,
        effective_settings: AppSettings,
        filenames: Sequence[str] | None,
    ) -> RetrievalPhase:
        """Run retrieve -> rerank -> neighbor-expand.

        Inlines ``retrieve_candidates``' steps (rather than calling it as one call)
        so embedding and search can be timed separately for observability.
        """

        normalized_query = question.strip()
        if not normalized_query:
            raise RetrievalError("Query text is required.")
        self._repository.ensure_ready(effective_settings)

        # Multi-query expansion (opt-in): rewrite into variant phrasings, retrieve each,
        # and RRF-fuse the result lists with the same pinned rrf_k before one rerank.
        # Left un-timed (query_expansion_ms stays 0.0) when disabled, like condense_ms.
        variants: list[str] = []
        query_expansion_ms = 0.0
        if effective_settings.query_expansion_enabled:
            query_expansion_start = time.monotonic()
            variants = generate_query_variants(
                normalized_query,
                int(effective_settings.query_expansion_count),
                self._generator,
                effective_settings,
            )
            query_expansion_ms = (time.monotonic() - query_expansion_start) * 1000
        queries = [normalized_query, *variants]

        embed_start = time.monotonic()
        query_embeddings = [embed_query(query, self._embedding_provider) for query in queries]
        embed_ms = (time.monotonic() - embed_start) * 1000

        search_start = time.monotonic()
        result_lists = [
            self._repository.hybrid_search(effective_settings, embedding, filenames)
            for embedding in query_embeddings
        ]
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

        rerank_start = time.monotonic()
        outcome = rerank_candidates_detailed(
            query=question,
            candidates=candidates,
            reranker=self._reranker,
            settings=effective_settings,
        )
        diverse_outcome, diversity_dropped = select_diverse(outcome, effective_settings)
        rerank_ms = (time.monotonic() - rerank_start) * 1000

        # Timed apart from rerank: neighbour expansion is a Qdrant round trip, not model
        # scoring, and folding it into rerank_ms hid it from every latency breakdown.
        context_expansion_start = time.monotonic()
        expanded = expand_with_neighbors(
            diverse_outcome.kept, self._repository, effective_settings
        )
        context_expansion_ms = (time.monotonic() - context_expansion_start) * 1000

        return RetrievalPhase(
            context_chunks=expanded,
            fused_candidates=candidates,
            scored_chunks=outcome.scored,
            embed_ms=embed_ms,
            search_ms=search_ms,
            rerank_ms=rerank_ms,
            query_expansion_ms=query_expansion_ms,
            context_expansion_ms=context_expansion_ms,
            diversity_dropped_ids=frozenset(diversity_dropped),
            query_variants=variants,
        )

    def answer(
        self,
        question: str,
        overrides: AnswerOverrides | None = None,
        filenames: Sequence[str] | None = None,
        history: Sequence[ChatMessage] | None = None,
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
        truncated_history: list[ChatMessage] = []

        try:
            effective_settings = self._effective_settings(overrides or AnswerOverrides())
            truncated_history = self._truncate_history(history, effective_settings)
            retrieval_query, condense_ms, condensed_question = self._condense_query(
                question, truncated_history, effective_settings
            )
            phase = self._retrieve_and_rerank(retrieval_query, effective_settings, filenames)
            selected = list(phase.context_chunks[: int(effective_settings.max_context_chunks)])

            generate_start = time.monotonic()
            grounded = generate_grounded_answer(
                query=question,
                context_chunks=phase.context_chunks,
                generator=self._generator,
                settings=effective_settings,
                history=truncated_history or None,
            )
            # generate_grounded_answer runs the zero-citation retry internally, so the
            # block just timed covers two LLM calls when it fired. Split it back out so
            # generate_ms means the same thing here as it does on the streaming path.
            generate_ms = (
                time.monotonic() - generate_start
            ) * 1000 - grounded.citation_retry_ms
            total_ms = (time.monotonic() - total_start) * 1000

            timings = StageTimings(
                embed_ms=phase.embed_ms,
                search_ms=phase.search_ms,
                rerank_ms=phase.rerank_ms,
                generate_ms=generate_ms,
                total_ms=total_ms,
                condense_ms=condense_ms,
                query_expansion_ms=phase.query_expansion_ms,
                context_expansion_ms=phase.context_expansion_ms,
                citation_retry_ms=grounded.citation_retry_ms,
            )
        except Exception as exc:
            if trace_store is not None and trace_id is not None:
                trace_store.add(
                    QueryTrace(
                        trace_id=trace_id,
                        created_at=time.time(),
                        question=question,
                        mode="sync",
                        status="error",
                        config=self._trace_config(effective_settings, filenames)
                        if effective_settings is not None
                        else None,
                        candidates=[],
                        prompt_messages=None,
                        answer=None,
                        cited_source_numbers=[],
                        timings=None,
                        error=str(exc),
                        condensed_question=condensed_question,
                        history_message_count=len(truncated_history),
                    )
                )
            raise

        if trace_store is not None and trace_id is not None:
            # Captured from generation, not rebuilt here. A second
            # build_grounded_messages call would agree today but is a re-derivation that
            # can drift, and it could never reproduce the citation-retry continuation
            # that actually produced the answer when grounded.citation_retry_used is set.
            prompt_messages = grounded.prompt_messages or None
            trace_store.add(
                QueryTrace(
                    trace_id=trace_id,
                    created_at=time.time(),
                    question=question,
                    mode="sync",
                    status="ok" if selected else "insufficient_context",
                    config=self._trace_config(effective_settings, filenames),
                    candidates=build_trace_candidates(
                        phase.fused_candidates,
                        phase.scored_chunks,
                        phase.context_chunks,
                        selected,
                        min_score=float(effective_settings.rerank_min_score),
                        diversity_dropped_ids=phase.diversity_dropped_ids,
                    ),
                    prompt_messages=prompt_messages,
                    answer=grounded.answer,
                    cited_source_numbers=[source.source_number for source in grounded.sources],
                    timings=_timings_dict(timings),
                    error=None,
                    condensed_question=condensed_question,
                    history_message_count=len(truncated_history),
                    query_variants=phase.query_variants,
                    citation_retry_used=grounded.citation_retry_used,
                )
            )
        return replace(grounded, timings=timings, trace_id=trace_id)

    def answer_stream(
        self,
        question: str,
        overrides: AnswerOverrides | None = None,
        filenames: Sequence[str] | None = None,
        history: Sequence[ChatMessage] | None = None,
    ) -> Iterator[StreamEvent]:
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
        truncated_history: list[ChatMessage] = []
        # Set the moment a trace has been stored (success, insufficient-context, or a
        # caught error below) — the `finally` clause uses this to detect the one case
        # none of those cover: the client disconnecting (GeneratorExit, a BaseException
        # that `except Exception` never sees) before any of them ran.
        finalized = False

        def _store_error_trace(error: BaseException) -> None:
            if trace_store is None or trace_id is None:
                return
            trace_store.add(
                QueryTrace(
                    trace_id=trace_id,
                    created_at=time.time(),
                    question=question,
                    mode="stream",
                    status="error",
                    config=self._trace_config(effective_settings, filenames)
                    if effective_settings is not None
                    else None,
                    candidates=[],
                    prompt_messages=None,
                    answer="".join(parts).strip() or None,
                    cited_source_numbers=[],
                    timings=None,
                    error=str(error),
                    condensed_question=condensed_question,
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
            retrieval_query, condense_ms, condensed_question = self._condense_query(
                question, truncated_history, effective_settings
            )
            yield {"type": "stage", "stage": "searching"}
            phase = self._retrieve_and_rerank(retrieval_query, effective_settings, filenames)
            selected = list(phase.context_chunks[: int(effective_settings.max_context_chunks)])

            if not selected:
                yield {"type": "sources", "sources": [], "trace_id": trace_id}
                total_ms = (time.monotonic() - total_start) * 1000
                timings = StageTimings(
                    embed_ms=phase.embed_ms,
                    search_ms=phase.search_ms,
                    rerank_ms=phase.rerank_ms,
                    generate_ms=0.0,
                    total_ms=total_ms,
                    condense_ms=condense_ms,
                    query_expansion_ms=phase.query_expansion_ms,
                    context_expansion_ms=phase.context_expansion_ms,
                )
                if trace_store is not None and trace_id is not None:
                    trace_store.add(
                        QueryTrace(
                            trace_id=trace_id,
                            created_at=time.time(),
                            question=question,
                            mode="stream",
                            status="insufficient_context",
                            config=self._trace_config(effective_settings, filenames),
                            candidates=build_trace_candidates(
                                phase.fused_candidates,
                                phase.scored_chunks,
                                phase.context_chunks,
                                [],
                                min_score=float(effective_settings.rerank_min_score),
                                diversity_dropped_ids=phase.diversity_dropped_ids,
                            ),
                            prompt_messages=None,
                            answer=INSUFFICIENT_CONTEXT_ANSWER,
                            cited_source_numbers=[],
                            timings=_timings_dict(timings),
                            error=None,
                            condensed_question=condensed_question,
                            history_message_count=len(truncated_history),
                            query_variants=phase.query_variants,
                        )
                    )
                finalized = True
                yield {
                    "type": "done",
                    "answer": INSUFFICIENT_CONTEXT_ANSWER,
                    "sources": [],
                    "timings": _timings_dict(timings),
                    "trace_id": trace_id,
                }
                return

            all_sources = source_citations(selected)
            yield {
                "type": "sources",
                "sources": [_citation_dict(source) for source in all_sources],
                "trace_id": trace_id,
            }

            messages = build_grounded_messages(question, selected, truncated_history or None)
            generate_start = time.monotonic()
            for delta in self._generator.stream(messages, effective_settings):
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
            outcome = finalize_citations(
                messages, answer, all_sources, self._generator, effective_settings
            )
            answer = outcome.answer
            cited = outcome.cited
            citation_retry_used = outcome.retry_used

            total_ms = (time.monotonic() - total_start) * 1000
            timings = StageTimings(
                embed_ms=phase.embed_ms,
                search_ms=phase.search_ms,
                rerank_ms=phase.rerank_ms,
                generate_ms=generate_ms,
                total_ms=total_ms,
                condense_ms=condense_ms,
                query_expansion_ms=phase.query_expansion_ms,
                context_expansion_ms=phase.context_expansion_ms,
                citation_retry_ms=outcome.retry_ms,
            )
            if trace_store is not None and trace_id is not None:
                trace_store.add(
                    QueryTrace(
                        trace_id=trace_id,
                        created_at=time.time(),
                        question=question,
                        mode="stream",
                        status="ok",
                        config=self._trace_config(effective_settings, filenames),
                        candidates=build_trace_candidates(
                            phase.fused_candidates,
                            phase.scored_chunks,
                            phase.context_chunks,
                            selected,
                            min_score=float(effective_settings.rerank_min_score),
                            diversity_dropped_ids=phase.diversity_dropped_ids,
                        ),
                        # outcome.prompt_messages, not the `messages` that were streamed:
                        # when the citation retry fired, the streamed answer was replaced
                        # by the retry's, and the retry's continuation is what produced
                        # the text stored just below.
                        prompt_messages=list(outcome.prompt_messages),
                        answer=answer,
                        cited_source_numbers=[source.source_number for source in cited],
                        timings=_timings_dict(timings),
                        error=None,
                        condensed_question=condensed_question,
                        history_message_count=len(truncated_history),
                        query_variants=phase.query_variants,
                        citation_retry_used=citation_retry_used,
                    )
                )
            finalized = True
            yield {
                "type": "done",
                "answer": answer,
                "sources": [_citation_dict(source) for source in cited],
                "timings": _timings_dict(timings),
                "trace_id": trace_id,
            }
        except Exception as exc:
            if not finalized:
                _store_error_trace(exc)
                finalized = True
            raise
        finally:
            if not finalized:
                # Reached only via GeneratorExit (the client disconnected, or the
                # generator was otherwise closed mid-stream without completing or
                # raising an `Exception`) — `finally` re-raises it automatically once
                # this block returns, so no explicit `raise` is needed here.
                _store_error_trace(RuntimeError("Stream closed before completion."))
                finalized = True


@dataclass(frozen=True)
class DocumentIngestOutcome:
    """Result of parsing, chunking, and indexing one uploaded document."""

    filename: str
    sections_parsed: int
    chunks_ingested: int
    collection_name: str


class IngestService:
    """Composes parsing, chunking, embedding, and indexing behind a single call."""

    def __init__(
        self,
        repository: VectorRepository,
        embedding_provider: IngestEmbeddingProvider,
        settings: AppSettings,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._settings = settings
        # Default to the heuristic (no model fetch) so direct constructions in tests
        # stay hermetic; production DI passes the dense model's real tokenizer.
        self._token_counter = token_counter or HeuristicTokenCounter()

    def ingest(
        self,
        filename: str,
        content: bytes,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> DocumentIngestOutcome:
        """Parse, chunk, embed, and index an uploaded document. Blocking — run off-loop."""

        sections = parse_document_bytes(filename, content, self._settings)
        chunks = chunk_sections(sections, self._settings, self._token_counter)
        # Stamped here (not in chunk_sections) because this is the first point in the
        # pipeline that has both the raw upload's byte count and the upload instant —
        # chunking itself works purely from parsed text.
        upload_stamp = time.time()
        upload_size = len(content)
        chunks = [
            replace(chunk, byte_size=upload_size, uploaded_at=upload_stamp) for chunk in chunks
        ]
        logger.info(
            "Chunked %s into %d chunks (chunker v%d).",
            sections[0].filename,
            len(chunks),
            CHUNKER_VERSION,
        )
        result: IngestResult = ingest_chunks(
            self._repository, self._settings, chunks, self._embedding_provider, on_progress
        )
        return DocumentIngestOutcome(
            filename=sections[0].filename,
            sections_parsed=len(sections),
            chunks_ingested=result.points_count,
            collection_name=result.collection_name,
        )
