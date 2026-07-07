from __future__ import annotations

from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from api.dependencies import (
    clear_dependency_caches,
    get_app_settings,
    get_embedding_provider,
    get_generator,
    get_qdrant_client,
    get_reranker,
)
from api.embeddings import EmbeddedText
from api.generation import ChatMessage, LiteLLMGenerator
from api.main import create_app
from api.qdrant_schema import EMBEDDING_MODEL_TAG_KEY, dense_vectors_config, sparse_vectors_config
from api.settings import AppSettings


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.seen_text_batches: list[list[str]] = []

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]:
        self.seen_text_batches.append(list(texts))
        embeddings: list[EmbeddedText] = []
        for text in texts:
            if text.strip().lower() == "alpha":
                embeddings.append(make_embedding([1.0, 0.0, 0.0], [10], [1.0]))
            else:
                embeddings.append(make_embedding([1.0, 0.0, 0.0], [10], [1.0]))
        return embeddings


class FakeReranker:
    def __init__(self) -> None:
        self.seen_documents: list[str] = []

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.seen_documents = list(documents)
        return [1.0 for _ in documents]


class FakeGenerator:
    def __init__(self, answer: str = "Alpha is documented [1].") -> None:
        self.answer = answer
        self.messages: list[ChatMessage] = []
        self.seen_provider: str | None = None

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        self.messages = list(messages)
        self.seen_provider = settings.llm_provider
        return self.answer


@dataclass
class ApiTestContext:
    client: TestClient
    settings: AppSettings
    qdrant: QdrantClient
    embeddings: FakeEmbeddingProvider
    reranker: FakeReranker
    generator: FakeGenerator


@pytest.fixture
def api_context() -> Generator[ApiTestContext]:
    clear_dependency_caches()
    settings = make_settings()
    qdrant = QdrantClient(":memory:")
    embeddings = FakeEmbeddingProvider()
    reranker = FakeReranker()
    generator = FakeGenerator()
    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant
    app.dependency_overrides[get_embedding_provider] = lambda: embeddings
    app.dependency_overrides[get_reranker] = lambda: reranker
    app.dependency_overrides[get_generator] = lambda: generator

    with TestClient(app) as client:
        yield ApiTestContext(
            client=client,
            settings=settings,
            qdrant=qdrant,
            embeddings=embeddings,
            reranker=reranker,
            generator=generator,
        )

    clear_dependency_caches()


def make_settings(**overrides: Any) -> AppSettings:
    defaults: dict[str, Any] = {
        "qdrant_url": ":memory:",
        "qdrant_collection": "api_documents",
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
    return AppSettings(**defaults)


def make_embedding(
    dense: list[float],
    sparse_indices: list[int],
    sparse_values: list[float],
) -> EmbeddedText:
    return EmbeddedText(
        dense=dense,
        sparse=models.SparseVector(indices=sparse_indices, values=sparse_values),
    )


def test_health_endpoint(api_context: ApiTestContext) -> None:
    response = api_context.client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_config_endpoint_exposes_non_secret_settings(api_context: ApiTestContext) -> None:
    response = api_context.client.get("/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["qdrant_collection"] == "api_documents"
    assert payload["dense_embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert payload["llm_provider"] == "ollama"
    assert payload["openai_available"] is False
    assert "openai_api_key" not in payload


def test_document_upload_ingests_chunks(api_context: ApiTestContext) -> None:
    response = api_context.client.post(
        "/documents",
        files={"file": ("guide.txt", b"Intro\nalpha beta", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "guide.txt",
        "sections_parsed": 1,
        "chunks_ingested": 1,
        "collection_name": "api_documents",
    }
    records, _ = api_context.qdrant.scroll(
        collection_name="api_documents",
        with_payload=True,
        with_vectors=False,
    )
    assert len(records) == 1
    assert records[0].payload is not None
    assert records[0].payload["filename"] == "guide.txt"
    assert records[0].payload["chunk_id"]


def test_document_upload_rejects_file_over_size_limit() -> None:
    """A single oversized upload must be rejected with 413 before it can OOM the
    process — proven by capping the limit far below the test payload's size."""

    clear_dependency_caches()
    settings = make_settings(max_upload_bytes=10)
    qdrant = QdrantClient(":memory:")
    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_generator] = lambda: FakeGenerator()
    with TestClient(app) as client:
        response = client.post(
            "/documents",
            files={
                "file": ("guide.txt", b"this content is definitely over ten bytes", "text/plain")
            },
        )
    clear_dependency_caches()

    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"]


def test_document_upload_rejects_unsupported_file(api_context: ApiTestContext) -> None:
    response = api_context.client.post(
        "/documents",
        files={"file": ("archive.zip", b"not supported", "application/zip")},
    )

    assert response.status_code == 400
    assert "Unsupported document type" in response.json()["detail"]


def test_question_endpoint_runs_grounded_pipeline(api_context: ApiTestContext) -> None:
    upload_response = api_context.client.post(
        "/documents",
        files={"file": ("guide.txt", b"Intro\nalpha beta", "text/plain")},
    )
    assert upload_response.status_code == 200

    response = api_context.client.post("/questions", json={"question": "alpha"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Alpha is documented [1]."
    assert payload["sources"] == [
        {
            "source_number": 1,
            "filename": "guide.txt",
            "page": 1,
            "section": "Intro",
            "chunk_id": payload["sources"][0]["chunk_id"],
            "text": "Intro alpha beta",
        }
    ]
    assert api_context.reranker.seen_documents == ["Intro alpha beta"]
    assert "Intro alpha beta" in api_context.generator.messages[1]["content"]


def test_question_endpoint_rejects_blank_query(api_context: ApiTestContext) -> None:
    response = api_context.client.post("/questions", json={"question": "   "})

    assert response.status_code == 400
    assert "Query text is required" in response.json()["detail"]


def test_config_reports_openai_available_when_key_present() -> None:
    clear_dependency_caches()
    settings = make_settings(openai_api_key="sk-test")
    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: settings
    with TestClient(app) as client:
        payload = client.get("/config").json()
    clear_dependency_caches()

    assert payload["openai_available"] is True
    assert "openai_api_key" not in payload


class CapturingCompletionClient:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def __call__(self, **kwargs: Any) -> object:
        self.kwargs = kwargs
        return {"choices": [{"message": {"content": "Grounded answer [1]."}}]}


def _ingested_client(
    settings: AppSettings,
    completion_client: CapturingCompletionClient,
) -> tuple[TestClient, QdrantClient]:
    """Build a TestClient wired with a real LiteLLMGenerator and one ingested doc.

    Uses the real generator (not the fake) so provider routing actually flows through
    ``completion_model_and_kwargs`` — the seam the per-request override drives.
    """

    qdrant = QdrantClient(":memory:")
    embeddings = FakeEmbeddingProvider()
    generator = LiteLLMGenerator(settings, completion_client=completion_client)
    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant
    app.dependency_overrides[get_embedding_provider] = lambda: embeddings
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_generator] = lambda: generator
    client = TestClient(app)
    upload = client.post(
        "/documents",
        files={"file": ("guide.txt", b"Intro\nalpha beta", "text/plain")},
    )
    assert upload.status_code == 200
    return client, qdrant


def test_question_endpoint_routes_to_openai_when_selected() -> None:
    clear_dependency_caches()
    settings = make_settings(openai_api_key="sk-test", openai_model="gpt-test")
    completion_client = CapturingCompletionClient()
    client, _ = _ingested_client(settings, completion_client)

    response = client.post("/questions", json={"question": "alpha", "llm_provider": "openai"})

    assert response.status_code == 200
    assert completion_client.kwargs is not None
    assert completion_client.kwargs["model"] == "gpt-test"
    assert completion_client.kwargs["api_key"] == "sk-test"
    # Per-request override must not mutate the shared settings singleton.
    assert settings.llm_provider == "ollama"
    clear_dependency_caches()


def test_question_endpoint_openai_selected_without_key_returns_400() -> None:
    clear_dependency_caches()
    settings = make_settings(openai_api_key=None)
    completion_client = CapturingCompletionClient()
    client, _ = _ingested_client(settings, completion_client)

    response = client.post("/questions", json={"question": "alpha", "llm_provider": "openai"})

    assert response.status_code == 400
    assert "OPENAI_API_KEY" in response.json()["detail"]
    # The provider never got called since routing rejected the request first.
    assert completion_client.kwargs is None
    clear_dependency_caches()


def test_question_endpoint_rejects_unknown_provider() -> None:
    clear_dependency_caches()
    settings = make_settings()
    completion_client = CapturingCompletionClient()
    client, _ = _ingested_client(settings, completion_client)

    response = client.post("/questions", json={"question": "alpha", "llm_provider": "anthropic"})

    # Literal["ollama","openai"] rejects unknown values at the schema boundary.
    assert response.status_code == 422
    clear_dependency_caches()


def test_question_endpoint_returns_conflict_for_embedding_mismatch(
    api_context: ApiTestContext,
) -> None:
    api_context.qdrant.create_collection(
        collection_name=api_context.settings.qdrant_collection,
        vectors_config=dense_vectors_config(api_context.settings),
        sparse_vectors_config=sparse_vectors_config(api_context.settings),
        metadata={EMBEDDING_MODEL_TAG_KEY: "other-model"},
    )

    response = api_context.client.post("/questions", json={"question": "alpha"})

    assert response.status_code == 409
    assert "Re-ingest documents" in response.json()["detail"]

