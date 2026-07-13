"""Orchestration seams: query (retrieve -> rerank -> generate) and ingest (parse ->
chunk -> embed -> index), each behind a single call so handlers stay thin.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, TypedDict

from api.documents import chunk_sections, parse_document_bytes
from api.generation import (
    INSUFFICIENT_CONTEXT_ANSWER,
    ChatGenerator,
    GenerationError,
    GroundedAnswer,
    SourceCitation,
    StageTimings,
    build_grounded_messages,
    cited_sources,
    generate_grounded_answer,
    source_citations,
)
from api.ingestion import EmbeddingProvider as IngestEmbeddingProvider
from api.ingestion import IngestResult, ingest_chunks
from api.repository import VectorRepository
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


class SourcesEvent(TypedDict):
    """SSE event: citation metadata for the selected context, sent before generation.

    Carries ``trace_id`` (None when tracing is disabled) so the UI can link this turn
    to its debug trace even if generation fails before the ``done`` event is reached.
    """

    type: Literal["sources"]
    sources: list[dict[str, Any]]
    trace_id: str | None


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


StreamEvent = SourcesEvent | DeltaEvent | DoneEvent


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
    """

    radius = int(settings.context_neighbor_radius)
    if radius <= 0:
        return list(chunks)

    expanded: list[RerankedChunk] = []
    for chunk in chunks:
        if chunk.chunk_ordinal is None:
            expanded.append(chunk)
            continue
        neighbor_ordinals = [
            chunk.chunk_ordinal + delta
            for delta in range(-radius, radius + 1)
            if delta != 0 and chunk.chunk_ordinal + delta >= 1
        ]
        neighbor_points = repository.fetch_neighbors(settings, chunk.filename, neighbor_ordinals)
        neighbor_texts = sorted(
            (point.payload.get("chunk_ordinal", 0), point.payload.get("text", ""))
            for point in neighbor_points
            if point.payload and point.payload.get("text")
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

        embed_start = time.monotonic()
        query_embedding = embed_query(normalized_query, self._embedding_provider)
        embed_ms = (time.monotonic() - embed_start) * 1000

        search_start = time.monotonic()
        points = self._repository.hybrid_search(effective_settings, query_embedding, filenames)
        candidates = points_to_chunks(points)
        search_ms = (time.monotonic() - search_start) * 1000

        rerank_start = time.monotonic()
        outcome = rerank_candidates_detailed(
            query=question,
            candidates=candidates,
            reranker=self._reranker,
            settings=effective_settings,
        )
        expanded = expand_with_neighbors(outcome.kept, self._repository, effective_settings)
        rerank_ms = (time.monotonic() - rerank_start) * 1000
        return RetrievalPhase(
            context_chunks=expanded,
            fused_candidates=candidates,
            scored_chunks=outcome.scored,
            embed_ms=embed_ms,
            search_ms=search_ms,
            rerank_ms=rerank_ms,
        )

    def answer(
        self,
        question: str,
        overrides: AnswerOverrides | None = None,
        filenames: Sequence[str] | None = None,
    ) -> GroundedAnswer:
        """Retrieve, rerank, and generate a grounded, cited answer.

        ``overrides`` are folded into a single ``model_copy`` of the shared settings —
        the singleton is never mutated, so concurrent requests with different
        overrides stay isolated. Only ``llm_provider``, ``rerank_top_k``,
        ``max_context_chunks``, and ``llm_temperature`` can vary per request; the
        dense/sparse retrieval limits, ``fused_top_n``, and ``rrf_k`` always come from
        the base settings (embeddings and hybrid fusion stay fixed regardless).
        ``filenames``, when given, restricts retrieval to those documents.
        """

        total_start = time.monotonic()
        trace_store = self._trace_store
        trace_id = str(uuid.uuid4()) if trace_store is not None else None
        effective_settings: AppSettings | None = None

        try:
            effective_settings = self._effective_settings(overrides or AnswerOverrides())
            phase = self._retrieve_and_rerank(question, effective_settings, filenames)
            selected = list(phase.context_chunks[: int(effective_settings.max_context_chunks)])

            generate_start = time.monotonic()
            grounded = generate_grounded_answer(
                query=question,
                context_chunks=phase.context_chunks,
                generator=self._generator,
                settings=effective_settings,
            )
            generate_ms = (time.monotonic() - generate_start) * 1000
            total_ms = (time.monotonic() - total_start) * 1000

            timings = StageTimings(
                embed_ms=phase.embed_ms,
                search_ms=phase.search_ms,
                rerank_ms=phase.rerank_ms,
                generate_ms=generate_ms,
                total_ms=total_ms,
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
                    )
                )
            raise

        if trace_store is not None and trace_id is not None:
            prompt_messages = (
                build_grounded_messages(question.strip(), selected) if selected else None
            )
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
                    ),
                    prompt_messages=prompt_messages,
                    answer=grounded.answer,
                    cited_source_numbers=[source.source_number for source in grounded.sources],
                    timings=_timings_dict(timings),
                    error=None,
                )
            )
        return replace(grounded, timings=timings, trace_id=trace_id)

    def answer_stream(
        self,
        question: str,
        overrides: AnswerOverrides | None = None,
        filenames: Sequence[str] | None = None,
    ) -> Iterator[StreamEvent]:
        """Stream a grounded answer as SSE-ready events: sources, deltas, then done.

        Sources are emitted immediately after rerank (before the LLM call starts) so
        the UI can render citations while the answer is still generating. ``done``
        carries the final answer, the citation-filtered source list, and timings.
        """

        total_start = time.monotonic()
        trace_store = self._trace_store
        trace_id = str(uuid.uuid4()) if trace_store is not None else None
        effective_settings: AppSettings | None = None
        parts: list[str] = []
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
                )
            )

        try:
            effective_settings = self._effective_settings(overrides or AnswerOverrides())
            phase = self._retrieve_and_rerank(question, effective_settings, filenames)
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
                            ),
                            prompt_messages=None,
                            answer=INSUFFICIENT_CONTEXT_ANSWER,
                            cited_source_numbers=[],
                            timings=_timings_dict(timings),
                            error=None,
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

            messages = build_grounded_messages(question, selected)
            generate_start = time.monotonic()
            for delta in self._generator.stream(messages, effective_settings):
                parts.append(delta)
                yield {"type": "delta", "text": delta}
            generate_ms = (time.monotonic() - generate_start) * 1000

            answer = "".join(parts).strip()
            if not answer:
                raise GenerationError("Generation provider returned an empty answer.")

            total_ms = (time.monotonic() - total_start) * 1000
            cited = cited_sources(answer, all_sources)
            timings = StageTimings(
                embed_ms=phase.embed_ms,
                search_ms=phase.search_ms,
                rerank_ms=phase.rerank_ms,
                generate_ms=generate_ms,
                total_ms=total_ms,
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
                        ),
                        prompt_messages=list(messages),
                        answer=answer,
                        cited_source_numbers=[source.source_number for source in cited],
                        timings=_timings_dict(timings),
                        error=None,
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
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._settings = settings

    def ingest(
        self,
        filename: str,
        content: bytes,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> DocumentIngestOutcome:
        """Parse, chunk, embed, and index an uploaded document. Blocking — run off-loop."""

        sections = parse_document_bytes(filename, content)
        chunks = chunk_sections(sections, self._settings)
        result: IngestResult = ingest_chunks(
            self._repository, self._settings, chunks, self._embedding_provider, on_progress
        )
        return DocumentIngestOutcome(
            filename=sections[0].filename,
            sections_parsed=len(sections),
            chunks_ingested=result.points_count,
            collection_name=result.collection_name,
        )
