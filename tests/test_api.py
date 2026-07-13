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

from api.dependencies import (
    clear_dependency_caches,
    get_app_settings,
    get_embedding_provider,
    get_generator,
    get_ollama_reachability_checker,
    get_qdrant_client,
    get_reranker,
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
    assert set(timings) == {"embed_ms", "search_ms", "rerank_ms", "generate_ms", "total_ms"}


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

