"""Orchestration seams: query (retrieve -> rerank -> generate) and ingest (parse ->
chunk -> embed -> index), each behind a single call so handlers stay thin.
"""

from __future__ import annotations

import time
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
from api.reranking import RerankedChunk, Reranker, rerank_candidates
from api.retrieval import (
    QueryEmbeddingProvider,
    RetrievalConfigError,
    RetrievalError,
    embed_query,
    points_to_chunks,
)
from api.settings import AppSettings


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


class SourcesEvent(TypedDict):
    """SSE event: citation metadata for the selected context, sent before generation."""

    type: Literal["sources"]
    sources: list[dict[str, Any]]


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
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._reranker = reranker
        self._generator = generator
        self._settings = settings

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

    def _retrieve_and_rerank(
        self,
        question: str,
        effective_settings: AppSettings,
        filenames: Sequence[str] | None,
    ) -> tuple[list[RerankedChunk], float, float, float]:
        """Run retrieve -> rerank -> neighbor-expand, returning
        (chunks, embed_ms, search_ms, rerank_ms).

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
        reranked = rerank_candidates(
            query=question,
            candidates=candidates,
            reranker=self._reranker,
            settings=effective_settings,
        )
        expanded = expand_with_neighbors(reranked, self._repository, effective_settings)
        rerank_ms = (time.monotonic() - rerank_start) * 1000
        return expanded, embed_ms, search_ms, rerank_ms

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
        effective_settings = self._effective_settings(overrides or AnswerOverrides())

        reranked, embed_ms, search_ms, rerank_ms = self._retrieve_and_rerank(
            question, effective_settings, filenames
        )

        generate_start = time.monotonic()
        grounded = generate_grounded_answer(
            query=question,
            context_chunks=reranked,
            generator=self._generator,
            settings=effective_settings,
        )
        generate_ms = (time.monotonic() - generate_start) * 1000
        total_ms = (time.monotonic() - total_start) * 1000

        timings = StageTimings(
            embed_ms=embed_ms,
            search_ms=search_ms,
            rerank_ms=rerank_ms,
            generate_ms=generate_ms,
            total_ms=total_ms,
        )
        return replace(grounded, timings=timings)

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
        effective_settings = self._effective_settings(overrides or AnswerOverrides())

        reranked, embed_ms, search_ms, rerank_ms = self._retrieve_and_rerank(
            question, effective_settings, filenames
        )
        selected = reranked[: int(effective_settings.max_context_chunks)]

        if not selected:
            yield {"type": "sources", "sources": []}
            total_ms = (time.monotonic() - total_start) * 1000
            yield {
                "type": "done",
                "answer": INSUFFICIENT_CONTEXT_ANSWER,
                "sources": [],
                "timings": _timings_dict(
                    StageTimings(
                        embed_ms=embed_ms,
                        search_ms=search_ms,
                        rerank_ms=rerank_ms,
                        generate_ms=0.0,
                        total_ms=total_ms,
                    )
                ),
            }
            return

        all_sources = source_citations(selected)
        yield {"type": "sources", "sources": [_citation_dict(source) for source in all_sources]}

        messages = build_grounded_messages(question, selected)
        generate_start = time.monotonic()
        parts: list[str] = []
        for delta in self._generator.stream(messages, effective_settings):
            parts.append(delta)
            yield {"type": "delta", "text": delta}
        generate_ms = (time.monotonic() - generate_start) * 1000

        answer = "".join(parts).strip()
        if not answer:
            raise GenerationError("Generation provider returned an empty answer.")

        total_ms = (time.monotonic() - total_start) * 1000
        cited = cited_sources(answer, all_sources)
        yield {
            "type": "done",
            "answer": answer,
            "sources": [_citation_dict(source) for source in cited],
            "timings": _timings_dict(
                StageTimings(
                    embed_ms=embed_ms,
                    search_ms=search_ms,
                    rerank_ms=rerank_ms,
                    generate_ms=generate_ms,
                    total_ms=total_ms,
                )
            ),
        }


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
