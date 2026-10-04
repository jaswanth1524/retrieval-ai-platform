"""Shared fakes and helpers for the HTTP route tests (split out of the old test_api.py)."""

from __future__ import annotations

import json
import time
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from api.chunking import HeuristicTokenCounter
from api.dependencies import (
    clear_dependency_caches,
    get_app_settings,
    get_embedding_provider,
    get_generator,
    get_ollama_reachability_checker,
    get_qdrant_client,
    get_reranker,
    get_token_counter,
)
from api.embeddings import EmbeddedText
from api.generation import ChatMessage, GenerationError, LiteLLMGenerator
from api.main import create_app
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


def hermetic_app(settings: AppSettings, qdrant: QdrantClient) -> FastAPI:
    """An app wired to ``settings`` and ``qdrant`` with the standard hermetic fakes.

    Every hand-rolled copy of this block drifted at some point (one was missing
    ``get_token_counter`` and fetched the bge tokenizer from the HF Hub on cold CI), so
    tests override only what they need on top of this.
    """

    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    app.dependency_overrides[get_generator] = lambda: FakeGenerator()
    # Hermetic ingest: the word heuristic, not the bge tokenizer from the HF Hub.
    app.dependency_overrides[get_token_counter] = lambda: HeuristicTokenCounter()
    # Never let /config make a real network call to Ollama.
    app.dependency_overrides[get_ollama_reachability_checker] = lambda: lambda _: False
    return app


def upload_and_wait(
    client: TestClient,
    filename: str,
    content: bytes,
    content_type: str = "text/plain",
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Upload a document via the async job endpoint and poll until it terminates.

    Ingestion now runs on a background executor (see api/jobs.py), so POST
    /documents only returns 202 + a job id — tests that need the document actually
    indexed (or need to see why ingestion failed) poll the job status endpoint
    instead of asserting on the upload response directly.
    """

    response = client.post(
        "/documents", files={"file": (filename, content, content_type)}, headers=headers
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    for _ in range(500):
        # Same headers as the upload: the job-status route is key-guarded too, so
        # polling it bare would 401 here exactly as it would in a real client.
        status = client.get(f"/documents/jobs/{job_id}", headers=headers).json()
        if status["state"] in ("done", "failed"):
            return status
        time.sleep(0.01)
    raise AssertionError(f"ingest job {job_id} did not finish in time")


@contextmanager
def keyed_client(
    api_key: str, extra_overrides: dict[Any, Any] | None = None, **overrides: Any
) -> Generator[TestClient]:
    """Client for an app with ``api_key`` configured and the standard hermetic overrides.

    The ``api_context`` fixture can't serve these tests because ``make_settings()``
    leaves ``api_key`` empty, but hand-rolling the override block per test drifted: the
    copies were missing ``get_token_counter``, so any test that uploaded ran the real
    chunker and fetched the bge tokenizer from the HF Hub — passing locally off a warm
    cache and reaching the network on cold CI. One helper keeps the override set in one
    place so a future addition to the fixture can't silently skip these.
    """

    clear_dependency_caches()
    settings = make_settings(api_key=api_key, **overrides)
    # Built once and closed over: a `lambda: QdrantClient(":memory:")` would hand every
    # request its own empty store, so an upload and the read that checks it would land in
    # different databases and the read would 404 while looking like an auth failure.
    qdrant = QdrantClient(":memory:")
    app = hermetic_app(settings, qdrant)
    app.dependency_overrides.update(extra_overrides or {})
    try:
        with TestClient(app) as client:
            yield client
    finally:
        clear_dependency_caches()


class RaisingReranker:
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        raise RerankingError("Reranker model failed: out of memory")


class RaisingGenerator:
    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        raise GenerationError("Generation provider request failed: connection refused")


def cors_app(monkeypatch: pytest.MonkeyPatch, origin: str) -> FastAPI:
    """Build an app whose CORS middleware really is configured for ``origin``.

    ``create_app`` reads settings through a direct ``get_app_settings()`` call to
    install the middleware, before any route exists — so ``dependency_overrides``
    (which only intercepts route dependencies) cannot reach it. Patching the name as
    imported into ``api.main`` is the seam that can, and matches how the warmup tests
    below control the same call.
    """

    clear_dependency_caches()
    settings = make_settings(cors_allow_origins=[origin])
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    return create_app()


class CapturingCompletionClient:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def __call__(self, **kwargs: Any) -> object:
        self.kwargs = kwargs
        return {"choices": [{"message": {"content": "Grounded answer [1]."}}]}


def ingested_client(
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
    app = hermetic_app(settings, qdrant)
    app.dependency_overrides[get_embedding_provider] = lambda: embeddings
    app.dependency_overrides[get_generator] = lambda: generator
    client = TestClient(app)
    status = upload_and_wait(client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"
    return client, qdrant


@contextmanager
def raw_storage_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    clear_dependency_caches()
    settings = make_settings(raw_document_dir=str(tmp_path))
    qdrant = QdrantClient(":memory:")
    monkeypatch.setattr("api.main.get_app_settings", lambda: settings)
    monkeypatch.setattr("api.dependencies.get_app_settings", lambda: settings)
    app = hermetic_app(settings, qdrant)
    try:
        with TestClient(app) as client:
            yield client
    finally:
        monkeypatch.undo()
        clear_dependency_caches()


def parse_sse_events(text: str) -> list[dict[str, Any]]:
    events = []
    for block in text.strip().split("\n\n"):
        if not block:
            continue
        _, _, data = block.partition("data: ")
        events.append(json.loads(data))
    return events


def without_stage_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop `stage` progress frames so positional assertions stay about the payload."""

    return [event for event in events if event["type"] != "stage"]


class FakeInactiveRpcError(grpc.RpcError):
    """Stands in for grpc._channel._InactiveRpcError, which has no public constructor.

    The handler probes for a callable .code() rather than the private concrete class,
    so this reproduces exactly what it keys on.
    """

    def __init__(self, code: object) -> None:
        self._code = code

    def code(self) -> object:
        return self._code


class StreamRaisingGenerator:
    """Generator whose stream() raises on first iteration, i.e. mid-response.

    The raise lands inside RagPipeline.answer_stream, after /questions/stream has
    already committed its 200 — the exact window the SSE error frame exists for.
    """

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def complete(self, messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        raise self._exc

    def stream(self, messages: Sequence[ChatMessage], settings: AppSettings) -> Any:
        raise self._exc
        yield ""  # unreachable; makes this a generator function


def stream_error_detail(api_context: ApiTestContext, exc: Exception) -> str:
    status = upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"
    api_context.app.dependency_overrides[get_generator] = lambda: StreamRaisingGenerator(exc)

    response = api_context.client.post("/questions/stream", json={"question": "alpha"})

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert events[-1]["type"] == "error"
    detail = events[-1]["detail"]
    assert isinstance(detail, str)
    return detail


def wait_for_job(client: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(300):
        status: dict[str, Any] = client.get(f"/documents/jobs/{job_id}").json()
        if status["state"] in ("done", "failed"):
            return status
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} never finished")


def count_generations(generator: FakeGenerator) -> list[int]:
    calls = [0]
    complete, stream = generator.complete, generator.stream

    def counting_complete(messages: Sequence[ChatMessage], settings: AppSettings) -> str:
        calls[0] += 1
        return complete(messages, settings)

    def counting_stream(messages: Sequence[ChatMessage], settings: AppSettings) -> Any:
        calls[0] += 1
        return stream(messages, settings)

    generator.complete = counting_complete  # type: ignore[method-assign]
    generator.stream = counting_stream  # type: ignore[method-assign]
    return calls
