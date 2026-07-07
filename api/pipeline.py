"""Orchestration seams: query (retrieve -> rerank -> generate) and ingest (parse ->
chunk -> embed -> index), each behind a single call so handlers stay thin.
"""

from __future__ import annotations

from dataclasses import dataclass

from api.documents import chunk_sections, parse_document_bytes
from api.generation import ChatGenerator, GroundedAnswer, generate_grounded_answer
from api.ingestion import EmbeddingProvider as IngestEmbeddingProvider
from api.ingestion import IngestResult, ingest_chunks
from api.repository import VectorRepository
from api.reranking import Reranker, rerank_candidates
from api.retrieval import QueryEmbeddingProvider, retrieve_candidates
from api.settings import AppSettings


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

    def answer(self, question: str, llm_provider: str | None = None) -> GroundedAnswer:
        """Retrieve, rerank, and generate a grounded, cited answer.

        ``llm_provider`` overrides the generation provider for this call only. It is
        applied via ``model_copy`` — the shared settings singleton is never mutated,
        so concurrent requests choosing different providers stay isolated. Retrieval
        and reranking always use the base settings (embeddings stay local regardless).
        """

        candidates = retrieve_candidates(
            repository=self._repository,
            settings=self._settings,
            query=question,
            embedding_provider=self._embedding_provider,
        )
        reranked = rerank_candidates(
            query=question,
            candidates=candidates,
            reranker=self._reranker,
            settings=self._settings,
        )
        generation_settings = (
            self._settings.model_copy(update={"llm_provider": llm_provider})
            if llm_provider is not None
            else self._settings
        )
        return generate_grounded_answer(
            query=question,
            context_chunks=reranked,
            generator=self._generator,
            settings=generation_settings,
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
