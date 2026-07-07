from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from api.embeddings import EmbeddedText
from api.generation import ChatMessage
from api.pipeline import IngestService, RagPipeline
from api.repository import VectorRepository
from api.settings import AppSettings


class StaticEmbeddingProvider:
    def __init__(self, embeddings: list[EmbeddedText]) -> None:
        self.embeddings = embeddings
        self.seen_texts: list[str] = []

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]:
        self.seen_texts = list(texts)
        return self.embeddings


class FakeReranker:
    def __init__(self) -> None:
        self.seen_documents: list[str] = []

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.seen_documents = list(documents)
        return [1.0 for _ in documents]


class FakeGenerator:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.messages: list[ChatMessage] = []
        self.seen_provider: str | None = None

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        self.messages = list(messages)
        self.seen_provider = settings.llm_provider
        return self.answer


def make_settings(**overrides: Any) -> AppSettings:
    defaults: dict[str, Any] = {
        "qdrant_url": ":memory:",
        "qdrant_collection": "pipeline_documents",
        "qdrant_dense_vector_name": "dense",
        "qdrant_sparse_vector_name": "sparse",
        "qdrant_dense_vector_size": 3,
        "embedding_model_tag": "test-embedding:v1",
        "chunk_size_tokens": 20,
        "chunk_overlap_tokens": 0,
        "fused_top_n": 5,
        "rerank_top_k": 3,
        "max_context_chunks": 2,
    }
    defaults.update(overrides)
    return AppSettings(_env_file=None, **defaults)  # type: ignore[call-arg]


def make_embedding(value: float) -> EmbeddedText:
    return EmbeddedText(
        dense=[value, 0.0, 0.0],
        sparse=models.SparseVector(indices=[1], values=[1.0]),
    )


def test_ingest_service_parses_chunks_and_indexes_them() -> None:
    settings = make_settings()
    repository = VectorRepository(QdrantClient(":memory:"))
    service = IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings)

    outcome = service.ingest("guide.txt", b"Intro\nalpha beta")

    assert outcome.filename == "guide.txt"
    assert outcome.sections_parsed == 1
    assert outcome.chunks_ingested == 1
    assert outcome.collection_name == "pipeline_documents"


def test_rag_pipeline_answers_from_ingested_document() -> None:
    settings = make_settings()
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )

    reranker = FakeReranker()
    generator = FakeGenerator("Alpha is documented [1].")
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=reranker,
        generator=generator,
        settings=settings,
    )

    grounded = pipeline.answer("alpha")

    assert grounded.answer == "Alpha is documented [1]."
    assert [source.filename for source in grounded.sources] == ["guide.txt"]
    assert reranker.seen_documents == ["Intro alpha beta"]
    assert "Intro alpha beta" in generator.messages[1]["content"]
    # No override → generation uses the base provider.
    assert generator.seen_provider == "ollama"


def test_rag_pipeline_provider_override_does_not_mutate_base_settings() -> None:
    settings = make_settings()  # llm_provider defaults to "ollama"
    client = QdrantClient(":memory:")
    repository = VectorRepository(client)
    IngestService(repository, StaticEmbeddingProvider([make_embedding(1.0)]), settings).ingest(
        "guide.txt", b"Intro\nalpha beta"
    )

    generator = FakeGenerator("Alpha is documented [1].")
    pipeline = RagPipeline(
        repository=repository,
        embedding_provider=StaticEmbeddingProvider([make_embedding(1.0)]),
        reranker=FakeReranker(),
        generator=generator,
        settings=settings,
    )

    pipeline.answer("alpha", llm_provider="openai")

    # The override reached generation, but the shared singleton is untouched.
    assert generator.seen_provider == "openai"
    assert settings.llm_provider == "ollama"
