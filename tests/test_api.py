from __future__ import annotations

import json
import time
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from api.chunking import HeuristicTokenCounter
from api.dependencies import (
    clear_dependency_caches,
    get_app_settings,
    get_embedding_provider,
    get_generator,
    get_ollama_reachability_checker,
    get_qdrant_client,
    get_qdrant_reachability_checker,
    get_reranker,
    get_token_counter,
)
from api.embeddings import EmbeddedText
from api.generation import ChatMessage, GenerationError, LiteLLMGenerator
from api.main import create_app, run_model_warmup
from api.qdrant_schema import EMBEDDING_MODEL_TAG_KEY, dense_vectors_config, sparse_vectors_config
from api.reranking import RerankingError
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

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings):
        self.messages = list(messages)
        self.seen_provider = settings.llm_provider
        yield self.answer


@dataclass
class ApiTestContext:
    client: TestClient
    app: FastAPI
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
    # Hermetic ingest: use the word heuristic instead of fetching the bge tokenizer
    # from the HF Hub over the network on every ingest-path test.
    app.dependency_overrides[get_token_counter] = lambda: HeuristicTokenCounter()
    # Hermetic by default: never let /config make a real network call to Ollama.
    app.dependency_overrides[get_ollama_reachability_checker] = lambda: (lambda _: False)

    with TestClient(app) as client:
        yield ApiTestContext(
            client=client,
            app=app,
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
    return AppSettings(_env_file=None, **defaults)  # type: ignore[call-arg]


def make_embedding(
    dense: list[float],
    sparse_indices: list[int],
    sparse_values: list[float],
) -> EmbeddedText:
    return EmbeddedText(
        dense=dense,
        sparse=models.SparseVector(indices=sparse_indices, values=sparse_values),
    )


def _upload_and_wait(
    client: TestClient,
    filename: str,
    content: bytes,
    content_type: str = "text/plain",
) -> dict[str, Any]:
    """Upload a document via the async job endpoint and poll until it terminates.

    Ingestion now runs on a background executor (see api/jobs.py), so POST
    /documents only returns 202 + a job id — tests that need the document actually
    indexed (or need to see why ingestion failed) poll the job status endpoint
    instead of asserting on the upload response directly.
    """

    response = client.post(
        "/documents", files={"file": (filename, content, content_type)}
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    for _ in range(500):
        status = client.get(f"/documents/jobs/{job_id}").json()
        if status["state"] in ("done", "failed"):
            return status
        time.sleep(0.01)
    raise AssertionError(f"ingest job {job_id} did not finish in time")


def test_health_endpoint(api_context: ApiTestContext) -> None:
    response = api_context.client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_ready_returns_ok_when_qdrant_and_provider_are_reachable(
    api_context: ApiTestContext,
) -> None:
    api_context.app.dependency_overrides[get_ollama_reachability_checker] = (
        lambda: (lambda _: True)
    )

    response = api_context.client.get("/health/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "status": "ok",
        "qdrant": True,
        "generation_provider": True,
        "llm_provider": "ollama",
    }


def test_health_ready_returns_503_when_generation_provider_unreachable(
    api_context: ApiTestContext,
) -> None:
    # api_context's default Ollama checker override already returns False.
    response = api_context.client.get("/health/ready")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["qdrant"] is True
    assert payload["generation_provider"] is False


def test_health_ready_returns_503_when_qdrant_unreachable(api_context: ApiTestContext) -> None:
    api_context.app.dependency_overrides[get_qdrant_reachability_checker] = (
        lambda: (lambda _: False)
    )
    api_context.app.dependency_overrides[get_ollama_reachability_checker] = (
        lambda: (lambda _: True)
    )

    response = api_context.client.get("/health/ready")

    assert response.status_code == 503
    payload = response.json()
    assert payload["qdrant"] is False
    assert payload["generation_provider"] is True


def test_health_endpoint_never_probes_dependencies(api_context: ApiTestContext) -> None:
    def raise_if_called(_client: object) -> bool:
        raise AssertionError("check_qdrant_reachable must not run for /health")

    api_context.app.dependency_overrides[get_qdrant_reachability_checker] = (
        lambda: raise_if_called
    )

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
    status = _upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")

    assert status["state"] == "done"
    assert status["result"] == {
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
    app.dependency_overrides[get_ollama_reachability_checker] = lambda: (lambda _: False)
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
    """Parsing happens inside the background job, so the upload itself is still
    accepted (202) — the unsupported-type failure surfaces on the job status."""

    status = _upload_and_wait(
        api_context.client, "archive.zip", b"not supported", "application/zip"
    )

    assert status["state"] == "failed"
    assert "Unsupported document type" in status["error"]


def test_question_endpoint_runs_grounded_pipeline(api_context: ApiTestContext) -> None:
    status = _upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"

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


def test_question_endpoint_rejects_empty_string_query(api_context: ApiTestContext) -> None:
    """Distinct from the whitespace case above: truly empty fails schema validation
    (min_length=1) before the pipeline's own strip-and-check ever runs."""

    response = api_context.client.post("/questions", json={"question": ""})

    assert response.status_code == 422


def test_question_endpoint_rejects_oversized_query(api_context: ApiTestContext) -> None:
    response = api_context.client.post("/questions", json={"question": "a" * 4001})

    assert response.status_code == 422


def test_question_endpoint_accepts_history(api_context: ApiTestContext) -> None:
    status = _upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"

    response = api_context.client.post(
        "/questions",
        json={
            "question": "alpha",
            "history": [
                {"role": "user", "content": "Tell me about doc A."},
                {"role": "assistant", "content": "Doc A covers setup [1]."},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["timings"]["condense_ms"] > 0.0


def test_question_endpoint_rejects_more_than_twelve_history_messages(
    api_context: ApiTestContext,
) -> None:
    history = [{"role": "user", "content": f"turn {i}"} for i in range(13)]

    response = api_context.client.post(
        "/questions", json={"question": "alpha", "history": history}
    )

    assert response.status_code == 422


def test_question_endpoint_rejects_invalid_history_role(api_context: ApiTestContext) -> None:
    response = api_context.client.post(
        "/questions",
        json={"question": "alpha", "history": [{"role": "system", "content": "nope"}]},
    )

    assert response.status_code == 422


def test_question_endpoint_rejects_out_of_bounds_rerank_top_k(
    api_context: ApiTestContext,
) -> None:
    response = api_context.client.post(
        "/questions", json={"question": "alpha", "rerank_top_k": 999}
    )

    assert response.status_code == 422


def test_question_endpoint_rejects_max_context_chunks_above_rerank_top_k(
    api_context: ApiTestContext,
) -> None:
    response = api_context.client.post(
        "/questions",
        json={"question": "alpha", "rerank_top_k": 3, "max_context_chunks": 5},
    )

    assert response.status_code == 422


def test_question_endpoint_rejects_rerank_top_k_above_fused_top_n(
    api_context: ApiTestContext,
) -> None:
    """api_context's settings default to fused_top_n=5 — a business rule known only
    inside the pipeline, so this is a 400 (RetrievalConfigError), not the schema's 422."""

    status = _upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"

    response = api_context.client.post(
        "/questions", json={"question": "alpha", "rerank_top_k": 45}
    )

    assert response.status_code == 400
    assert "fused_top_n" in response.json()["detail"]


def test_question_endpoint_accepts_valid_overrides_and_limits_sources(
    api_context: ApiTestContext,
) -> None:
    status = _upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"

    response = api_context.client.post(
        "/questions",
        json={"question": "alpha", "rerank_top_k": 1, "max_context_chunks": 1},
    )

    assert response.status_code == 200
    assert len(response.json()["sources"]) <= 1


class RaisingReranker:
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        raise RerankingError("Reranker model failed: out of memory")


class RaisingGenerator:
    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        raise GenerationError("Generation provider request failed: connection refused")


def test_question_endpoint_returns_502_shape_for_reranking_failure(
    api_context: ApiTestContext,
) -> None:
    """Pins the exact response shape (status + string `detail`) the frontend's
    extractErrorDetail relies on — a regression here would silently break error
    rendering in the UI without any type system catching it."""

    status = _upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"
    api_context.app.dependency_overrides[get_reranker] = lambda: RaisingReranker()

    response = api_context.client.post("/questions", json={"question": "alpha"})

    assert response.status_code == 502
    assert isinstance(response.json()["detail"], str)
    assert "out of memory" in response.json()["detail"]


def test_question_endpoint_returns_502_shape_for_generation_failure(
    api_context: ApiTestContext,
) -> None:
    status = _upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"
    api_context.app.dependency_overrides[get_generator] = lambda: RaisingGenerator()

    response = api_context.client.post("/questions", json={"question": "alpha"})

    assert response.status_code == 502
    assert isinstance(response.json()["detail"], str)
    assert "connection refused" in response.json()["detail"]


def test_question_endpoint_returns_503_when_qdrant_unreachable(
    api_context: ApiTestContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_connection(*args: object, **kwargs: object) -> bool:
        raise ResponseHandlingException(ConnectionError("connection refused"))

    monkeypatch.setattr(api_context.qdrant, "collection_exists", fail_connection)

    response = api_context.client.post("/questions", json={"question": "alpha"})

    assert response.status_code == 503
    assert isinstance(response.json()["detail"], str)
    assert "unreachable" in response.json()["detail"]


def test_config_reports_openai_available_when_key_present() -> None:
    clear_dependency_caches()
    settings = make_settings(openai_api_key="sk-test")
    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_ollama_reachability_checker] = lambda: (lambda _: False)
    with TestClient(app) as client:
        payload = client.get("/config").json()
    clear_dependency_caches()

    assert payload["openai_available"] is True
    assert "openai_api_key" not in payload


def test_config_reports_ollama_available_true_or_false_from_checker() -> None:
    clear_dependency_caches()
    settings = make_settings()
    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: settings

    app.dependency_overrides[get_ollama_reachability_checker] = lambda: (lambda _: True)
    with TestClient(app) as client:
        assert client.get("/config").json()["ollama_available"] is True

    app.dependency_overrides[get_ollama_reachability_checker] = lambda: (lambda _: False)
    with TestClient(app) as client:
        assert client.get("/config").json()["ollama_available"] is False
    clear_dependency_caches()


def test_config_exposes_override_limit_fields() -> None:
    clear_dependency_caches()
    settings = make_settings(fused_top_n=5)
    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_ollama_reachability_checker] = lambda: (lambda _: False)
    with TestClient(app) as client:
        payload = client.get("/config").json()
    clear_dependency_caches()

    # rerank_top_k_limit is capped by the server's own fused_top_n (5 here), not the
    # global REQUEST_RERANK_TOP_K_MAX (50).
    assert payload["rerank_top_k_limit"] == 5
    assert payload["max_context_chunks_limit"] == 20
    assert payload["llm_temperature_max"] == 2.0
    assert payload["llm_temperature"] == settings.llm_temperature


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
    app.dependency_overrides[get_ollama_reachability_checker] = lambda: (lambda _: False)
    client = TestClient(app)
    status = _upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"
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


def test_run_model_warmup_touches_both_models() -> None:
    embeddings = FakeEmbeddingProvider()
    reranker = FakeReranker()

    run_model_warmup(embeddings, reranker)

    assert embeddings.seen_text_batches == [["warmup"]]
    assert reranker.seen_documents == ["warmup"]


def test_run_model_warmup_swallows_failures_instead_of_raising() -> None:
    class RaisingEmbeddingProvider:
        def embed_texts(self, texts: Sequence[str]) -> list[EmbeddedText]:
            raise RuntimeError("model download failed")

    # Must not raise — a warmup failure is logged, not fatal to app startup.
    run_model_warmup(RaisingEmbeddingProvider(), FakeReranker())


def test_lifespan_runs_warmup_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lifespan calls get_embedding_provider()/get_reranker() directly (not via
    FastAPI Depends), so dependency_overrides can't intercept them — patch the names
    api.main actually holds instead, same as production code would resolve them."""

    clear_dependency_caches()
    settings = make_settings(warmup_models=True)
    embeddings = FakeEmbeddingProvider()
    reranker = FakeReranker()
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    monkeypatch.setattr("api.main.get_embedding_provider", lambda: embeddings)
    monkeypatch.setattr("api.main.get_reranker", lambda: reranker)

    app = create_app()
    with TestClient(app):
        pass

    assert embeddings.seen_text_batches == [["warmup"]]
    assert reranker.seen_documents == ["warmup"]
    clear_dependency_caches()


def test_lifespan_skips_warmup_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_dependency_caches()
    settings = make_settings(warmup_models=False)
    embeddings = FakeEmbeddingProvider()
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    monkeypatch.setattr("api.main.get_embedding_provider", lambda: embeddings)
    monkeypatch.setattr("api.main.get_reranker", lambda: FakeReranker())

    app = create_app()
    with TestClient(app):
        pass

    assert embeddings.seen_text_batches == []
    clear_dependency_caches()


def test_document_job_status_returns_404_for_unknown_job(api_context: ApiTestContext) -> None:
    response = api_context.client.get("/documents/jobs/does-not-exist")

    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]


def test_list_documents_returns_indexed_filenames(api_context: ApiTestContext) -> None:
    assert _upload_and_wait(api_context.client, "a.txt", b"alpha")["state"] == "done"
    assert _upload_and_wait(api_context.client, "b.txt", b"beta")["state"] == "done"

    response = api_context.client.get("/documents")

    assert response.status_code == 200
    assert response.json()["filenames"] == ["a.txt", "b.txt"]


def test_list_documents_empty_for_no_uploads(api_context: ApiTestContext) -> None:
    response = api_context.client.get("/documents")

    assert response.status_code == 200
    assert response.json()["filenames"] == []


def test_delete_document_removes_indexed_chunks(api_context: ApiTestContext) -> None:
    assert _upload_and_wait(api_context.client, "a.txt", b"alpha")["state"] == "done"

    response = api_context.client.delete("/documents/a.txt")

    assert response.status_code == 200
    assert response.json() == {"filename": "a.txt", "points_deleted": 1}
    assert api_context.client.get("/documents").json()["filenames"] == []


def test_delete_document_returns_404_for_unknown_filename(api_context: ApiTestContext) -> None:
    response = api_context.client.delete("/documents/does-not-exist.txt")

    assert response.status_code == 404
    assert "does-not-exist.txt" in response.json()["detail"]


def test_delete_document_leaves_other_filenames_searchable(api_context: ApiTestContext) -> None:
    assert _upload_and_wait(api_context.client, "a.txt", b"alpha")["state"] == "done"
    assert _upload_and_wait(api_context.client, "b.txt", b"beta")["state"] == "done"

    response = api_context.client.delete("/documents/a.txt")

    assert response.status_code == 200
    assert response.json()["points_deleted"] == 1
    assert api_context.client.get("/documents").json()["filenames"] == ["b.txt"]

    question_response = api_context.client.post("/questions", json={"question": "beta"})
    assert question_response.status_code == 200


def test_document_content_returns_chunks_in_ordinal_order(api_context: ApiTestContext) -> None:
    status = _upload_and_wait(api_context.client, "a.md", b"Intro\nalpha beta gamma")
    assert status["state"] == "done"

    response = api_context.client.get("/documents/a.md/content")

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "a.md"
    assert len(body["chunks"]) >= 1
    ordinals = [chunk["chunk_ordinal"] for chunk in body["chunks"]]
    assert ordinals == sorted(o for o in ordinals if o is not None)
    assert all({"chunk_id", "page", "section", "text"} <= set(chunk) for chunk in body["chunks"])


def test_document_content_returns_404_for_unknown_filename(api_context: ApiTestContext) -> None:
    response = api_context.client.get("/documents/missing.md/content")

    assert response.status_code == 404


def test_metrics_endpoint_exposes_prometheus_text(api_context: ApiTestContext) -> None:
    response = api_context.client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_question_endpoint_includes_stage_timings(api_context: ApiTestContext) -> None:
    status = _upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"

    response = api_context.client.post("/questions", json={"question": "alpha"})

    assert response.status_code == 200
    timings = response.json()["timings"]
    assert timings is not None
    assert set(timings) == {
        "embed_ms",
        "search_ms",
        "rerank_ms",
        "generate_ms",
        "total_ms",
        "condense_ms",
        "expand_ms",
    }
    assert timings["condense_ms"] == 0.0
    assert timings["expand_ms"] == 0.0


def test_question_endpoint_filters_by_filenames(api_context: ApiTestContext) -> None:
    assert _upload_and_wait(api_context.client, "a.txt", b"alpha in a")["state"] == "done"
    assert _upload_and_wait(api_context.client, "b.txt", b"alpha in b")["state"] == "done"

    response = api_context.client.post(
        "/questions", json={"question": "alpha", "filenames": ["b.txt"]}
    )

    assert response.status_code == 200
    assert [source["filename"] for source in response.json()["sources"]] == ["b.txt"]


def _parse_sse_events(text: str) -> list[dict[str, Any]]:
    events = []
    for block in text.strip().split("\n\n"):
        if not block:
            continue
        _, _, data = block.partition("data: ")
        events.append(json.loads(data))
    return events


def test_question_stream_endpoint_emits_sources_delta_done(
    api_context: ApiTestContext,
) -> None:
    status = _upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"

    response = api_context.client.post("/questions/stream", json={"question": "alpha"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(response.text)

    assert events[0]["type"] == "sources"
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"] == "Alpha is documented [1]."
    assert "timings" in events[-1]


def test_question_endpoint_returns_a_trace_id(api_context: ApiTestContext) -> None:
    status = _upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"

    response = api_context.client.post("/questions", json={"question": "alpha"})

    assert response.status_code == 200
    assert response.json()["trace_id"]


def test_traces_endpoint_lists_and_fetches_a_recorded_trace(
    api_context: ApiTestContext,
) -> None:
    status = _upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"
    question_response = api_context.client.post("/questions", json={"question": "alpha"})
    trace_id = question_response.json()["trace_id"]

    list_response = api_context.client.get("/traces")
    assert list_response.status_code == 200
    summaries = list_response.json()["traces"]
    assert len(summaries) == 1
    assert summaries[0]["trace_id"] == trace_id
    assert summaries[0]["status"] == "ok"
    assert summaries[0]["mode"] == "sync"

    detail_response = api_context.client.get(f"/traces/{trace_id}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["trace_id"] == trace_id
    assert detail["question"] == "alpha"
    assert detail["answer"] == "Alpha is documented [1]."
    assert detail["cited_source_numbers"] == [1]
    assert len(detail["candidates"]) == 1
    assert detail["candidates"][0]["filename"] == "guide.txt"
    assert detail["candidates"][0]["kept"] is True
    assert detail["prompt_messages"] == api_context.generator.messages
    assert detail["config"]["llm_provider"] == "ollama"
    assert detail["timings"]["total_ms"] >= 0


def test_traces_endpoint_returns_404_for_unknown_trace_id(api_context: ApiTestContext) -> None:
    response = api_context.client.get("/traces/does-not-exist")

    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]


def test_question_stream_endpoint_trace_id_matches_across_events_and_is_fetchable(
    api_context: ApiTestContext,
) -> None:
    status = _upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"

    response = api_context.client.post("/questions/stream", json={"question": "alpha"})
    events = _parse_sse_events(response.text)

    trace_id = events[0]["trace_id"]
    assert trace_id
    assert events[-1]["trace_id"] == trace_id

    detail_response = api_context.client.get(f"/traces/{trace_id}")
    assert detail_response.status_code == 200
    assert detail_response.json()["mode"] == "stream"


def test_question_stream_endpoint_done_timings_include_condense_ms(
    api_context: ApiTestContext,
) -> None:
    status = _upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"

    response = api_context.client.post(
        "/questions/stream",
        json={
            "question": "What about the second one?",
            "history": [
                {"role": "user", "content": "Tell me about doc A."},
                {"role": "assistant", "content": "Doc A covers setup [1]."},
            ],
        },
    )
    events = _parse_sse_events(response.text)

    done = events[-1]
    assert done["type"] == "done"
    assert done["timings"]["condense_ms"] > 0.0

    detail = api_context.client.get(f"/traces/{done['trace_id']}").json()
    assert detail["history_message_count"] == 2
    assert detail["condensed_question"]


def test_question_endpoint_trace_id_null_and_traces_empty_when_tracing_disabled() -> None:
    clear_dependency_caches()
    settings = make_settings(trace_enabled=False)
    qdrant = QdrantClient(":memory:")
    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_generator] = lambda: FakeGenerator()
    app.dependency_overrides[get_ollama_reachability_checker] = lambda: (lambda _: False)
    with TestClient(app) as client:
        _upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")
        response = client.post("/questions", json={"question": "alpha"})
        traces_response = client.get("/traces")
    clear_dependency_caches()

    assert response.json()["trace_id"] is None
    assert traces_response.json()["traces"] == []

