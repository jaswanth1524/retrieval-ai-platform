"""Orchestration seams: query (retrieve -> rerank -> generate) and ingest (parse ->
chunk -> embed -> index), each behind a single call so handlers stay thin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from api.documents import chunk_sections, parse_document_bytes
from api.generation import ChatGenerator, GroundedAnswer, generate_grounded_answer
from api.ingestion import EmbeddingProvider as IngestEmbeddingProvider
from api.ingestion import IngestResult, ingest_chunks
from api.repository import VectorRepository
from api.reranking import Reranker, rerank_candidates
from api.retrieval import QueryEmbeddingProvider, RetrievalConfigError, retrieve_candidates
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

    def answer(self, question: str, overrides: AnswerOverrides | None = None) -> GroundedAnswer:
        """Retrieve, rerank, and generate a grounded, cited answer.

        ``overrides`` are folded into a single ``model_copy`` of the shared settings —
        the singleton is never mutated, so concurrent requests with different
        overrides stay isolated. Only ``llm_provider``, ``rerank_top_k``,
        ``max_context_chunks``, and ``llm_temperature`` can vary per request; the
        dense/sparse retrieval limits, ``fused_top_n``, and ``rrf_k`` always come from
        the base settings (embeddings and hybrid fusion stay fixed regardless).
        """

        overrides = overrides or AnswerOverrides()
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
        effective_settings = settings.model_copy(update=update) if update else settings

        candidates = retrieve_candidates(
            repository=self._repository,
            settings=effective_settings,
            query=question,
            embedding_provider=self._embedding_provider,
        )
        reranked = rerank_candidates(
            query=question,
            candidates=candidates,
            reranker=self._reranker,
            settings=effective_settings,
        )
        return generate_grounded_answer(
            query=question,
            context_chunks=reranked,
            generator=self._generator,
            settings=effective_settings,
        )


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

    def ingest(self, filename: str, content: bytes) -> DocumentIngestOutcome:
        """Parse, chunk, embed, and index an uploaded document. Blocking — run off-loop."""

        sections = parse_document_bytes(filename, content)
        chunks = chunk_sections(sections, self._settings)
        result: IngestResult = ingest_chunks(
            self._repository, self._settings, chunks, self._embedding_provider
        )
        return DocumentIngestOutcome(
            filename=sections[0].filename,
            sections_parsed=len(sections),
            chunks_ingested=result.points_count,
            collection_name=result.collection_name,
        )
