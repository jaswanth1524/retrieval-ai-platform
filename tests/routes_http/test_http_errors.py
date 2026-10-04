"""Error mapping: status codes, redaction and error classification."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import grpc
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from api.chunking import HeuristicTokenCounter
from api.dependencies import (
    clear_dependency_caches,
    get_app_settings,
    get_embedding_provider,
    get_generator,
    get_ingest_backlog,
    get_ollama_reachability_checker,
    get_qdrant_client,
    get_qdrant_reachability_checker,
    get_reranker,
    get_token_counter,
)
from api.documents import EmptyDocumentError
from api.generation import GenerationConfigError, GenerationError
from api.ingestion import IngestionError
from api.jobs import IngestBacklog
from api.main import (
    _ingest_error_type,
    _job_error_message,
    _question_error_type,
    create_app,
)
from api.qdrant_schema import (
    CollectionSchemaError,
    VectorStoreUnavailableError,
)
from api.reranking import RerankingError
from api.retrieval import RetrievalConfigError, RetrievalError, RetrievalPayloadError
from tests.routes_http.support import (
    ApiTestContext,
    FakeEmbeddingProvider,
    FakeInactiveRpcError,
    RaisingGenerator,
    RaisingReranker,
    keyed_client,
    make_settings,
    raw_storage_client,
    stream_error_detail,
    upload_and_wait,
)


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
    api_context.app.dependency_overrides[get_qdrant_reachability_checker] = lambda: lambda _: False
    api_context.app.dependency_overrides[get_ollama_reachability_checker] = lambda: lambda _: True

    response = api_context.client.get("/health/ready")

    assert response.status_code == 503
    payload = response.json()
    assert payload["qdrant"] is False
    assert payload["generation_provider"] is True


def test_document_upload_returns_503_when_the_ingest_backlog_is_full(
    api_context: ApiTestContext,
) -> None:
    backlog = IngestBacklog(max_bytes=10)
    assert backlog.try_reserve(6)  # something already waiting
    api_context.app.dependency_overrides[get_ingest_backlog] = lambda: backlog

    response = api_context.client.post(
        "/documents", files={"file": ("guide.txt", b"0123456789", "text/plain")}
    )

    assert response.status_code == 503
    assert response.headers["retry-after"]
    # Rejected uploads reserve nothing.
    assert backlog.pending_bytes == 6


def test_question_endpoint_returns_502_shape_for_reranking_failure(
    api_context: ApiTestContext,
) -> None:
    """Pins the exact response shape (status + string `detail`) the frontend's
    extractErrorDetail relies on — a regression here would silently break error
    rendering in the UI without any type system catching it."""

    status = upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
    assert status["state"] == "done"
    api_context.app.dependency_overrides[get_reranker] = lambda: RaisingReranker()

    response = api_context.client.post("/questions", json={"question": "alpha"})

    assert response.status_code == 502
    assert isinstance(response.json()["detail"], str)
    assert "out of memory" in response.json()["detail"]


def test_question_endpoint_returns_502_shape_for_generation_failure(
    api_context: ApiTestContext,
) -> None:
    status = upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")
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


def test_document_original_deleted_after_the_existence_check_is_a_404_not_a_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The race: path() saw the file, a concurrent DELETE removed it, then the response
    went to open it. FileResponse opened it only while sending, so that was a 500."""

    from api.raw_documents import RawDocumentStore

    with raw_storage_client(tmp_path, monkeypatch) as client:
        assert upload_and_wait(client, "guide.txt", b"alpha beta")["state"] == "done"
        stale_path = tmp_path / "guide.txt"
        stale_path.unlink()  # the DELETE that won the race
        monkeypatch.setattr(RawDocumentStore, "path", lambda self, filename: stale_path)

        response = client.get("/documents/guide.txt/original")

        assert response.status_code == 404


def test_api_key_non_ascii_header_is_401_not_500() -> None:
    """A non-ASCII X-API-Key must be rejected, not crash the request.

    ``secrets.compare_digest`` raises TypeError when handed a non-ASCII *str*, and
    Starlette decodes header bytes with latin-1, so comparing strings turned any
    unauthenticated request carrying a non-ASCII byte in X-API-Key into a 500 (plus a
    traceback in the logs) on every guarded route. The comparison is on bytes now.
    """

    # Raw bytes, not str: a bare 0xe9 isn't valid utf-8, and "🔑" exercises the
    # multi-byte path. Both reach require_api_key latin-1-decoded.
    non_ascii_values = [b"\xe9", "é".encode(), "🔑".encode()]
    with keyed_client("secret-key") as client:
        responses = [
            client.get("/documents", headers={b"X-API-Key": value}) for value in non_ascii_values
        ]

    for value, response in zip(non_ascii_values, responses, strict=True):
        assert response.status_code == 401, (value, response.status_code, response.text)


def test_job_error_message_keeps_known_domain_error_text() -> None:
    exc = EmptyDocumentError("Document has no extractable text to chunk.")
    assert _job_error_message(exc) == "Document has no extractable text to chunk."


def test_job_error_message_redacts_unexpected_exception_text() -> None:
    """An exception type the ingest path never anticipated (a bare KeyError, an SDK
    internal error, ...) may carry a path/config value in its str() — that must not
    reach the client verbatim; the logger.warning at the raise site keeps the detail
    server-side."""

    exc = ValueError("unexpected: /internal/config/path leaked here")
    message = _job_error_message(exc)
    assert message == "Ingestion failed due to an unexpected server error."
    assert "/internal/config/path" not in message


def test_ingest_error_type_classifies_known_domain_errors() -> None:
    assert _ingest_error_type(EmptyDocumentError("x")) == "document"
    assert _ingest_error_type(IngestionError("x")) == "ingestion"
    assert _ingest_error_type(VectorStoreUnavailableError("x")) == "vector_store_unavailable"
    assert _ingest_error_type(CollectionSchemaError("x")) == "collection_schema"
    assert _ingest_error_type(ValueError("x")) == "unexpected"


def test_question_error_type_prefers_the_most_specific_subclass() -> None:
    """RetrievalPayloadError/RetrievalConfigError and GenerationConfigError are
    subclasses of RetrievalError/GenerationError respectively — a parent-first check
    would misclassify every subclass instance as its broader parent."""

    assert _question_error_type(RetrievalPayloadError("x")) == "retrieval_payload"
    assert _question_error_type(RetrievalConfigError("x")) == "retrieval_config"
    assert _question_error_type(RetrievalError("x")) == "retrieval"
    assert _question_error_type(GenerationConfigError("x")) == "generation_config"
    assert _question_error_type(GenerationError("x")) == "generation"
    assert _question_error_type(RerankingError("x")) == "reranking"
    assert _question_error_type(ValueError("x")) == "unexpected"


def test_question_stream_redacts_an_unexpected_exception(api_context: ApiTestContext) -> None:
    """/questions can let an unanticipated exception fall through to Starlette's
    default 500, which leaks nothing. /questions/stream cannot — its 200 is already
    committed — so it catches everything, and without an allowlist that meant putting
    an arbitrary str(exc) on the wire.
    """

    detail = stream_error_detail(
        api_context, ValueError("unexpected: /internal/config/path leaked here")
    )

    assert detail == "The question could not be answered due to an unexpected server error."
    assert "/internal/config/path" not in detail


def test_question_stream_keeps_a_known_domain_error_message(
    api_context: ApiTestContext,
) -> None:
    """The redaction must not swallow the messages that are useful and safe — these
    are the same types register_exception_handlers already returns verbatim on the
    synchronous route."""

    detail = stream_error_detail(
        api_context, GenerationError("Cannot reach Ollama at http://localhost:11434.")
    )

    assert detail == "Cannot reach Ollama at http://localhost:11434."


def test_documents_returns_503_not_500_when_qdrant_drops_after_startup(
    api_context: ApiTestContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only VectorRepository.hybrid_search translates ResponseHandlingException into
    VectorStoreUnavailableError; filename_metadata and friends let it escape. ensure_ready
    can't cover for them either, because ensure_collection caches success per client — so
    an outage that starts after the first successful request was a bare 500 here while
    /questions correctly reported 503.
    """

    upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")

    def fail_connection(*args: object, **kwargs: object) -> Any:
        raise ResponseHandlingException(ConnectionError("connection refused"))

    monkeypatch.setattr(api_context.qdrant, "scroll", fail_connection)

    response = api_context.client.get("/documents")

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert isinstance(detail, str)
    # Fixed message, not str(exc): this is a blanket handler, so it also catches a bug
    # that merely surfaces as this type, where naming the cause would be wrong.
    assert "connection refused" not in detail


def test_question_stream_names_the_vector_store_when_it_is_unreachable(
    api_context: ApiTestContext,
) -> None:
    """A raw transport failure carries no authored message, so the generic redaction
    would report "unexpected server error" for something the synchronous route names
    properly — hiding an actionable cause from anyone watching the stream."""

    detail = stream_error_detail(api_context, FakeInactiveRpcError(grpc.StatusCode.UNAVAILABLE))

    assert detail == "The vector store is unavailable. Try again shortly."


def test_documents_returns_503_when_qdrant_grpc_is_unavailable(
    api_context: ApiTestContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Found by running against a real container, not by reading the code: the
    existing ResponseHandlingException translations are REST-only. docker-compose sets
    QDRANT_PREFER_GRPC=true, and on that path an unreachable Qdrant raises
    grpc.RpcError instead — which ResponseHandlingException never matches, so even
    /questions returned a bare 500 on the project's own default deployment.
    """

    upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")

    def fail_grpc(*args: object, **kwargs: object) -> Any:
        raise FakeInactiveRpcError(grpc.StatusCode.UNAVAILABLE)

    monkeypatch.setattr(api_context.qdrant, "scroll", fail_grpc)

    response = api_context.client.get("/documents")

    assert response.status_code == 503
    assert "The vector store is unavailable" in response.json()["detail"]


def test_a_non_transport_grpc_error_is_a_redacted_500_not_a_false_503(
    api_context: ApiTestContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The handler is blanket over a type the app doesn't own, so it also catches bugs
    that merely surface as gRPC errors. Those are still redacted and logged, but must
    not be reported as the vector store being down."""

    upload_and_wait(api_context.client, "guide.txt", b"Intro\nalpha beta")

    def fail_grpc(*args: object, **kwargs: object) -> Any:
        raise FakeInactiveRpcError(grpc.StatusCode.INVALID_ARGUMENT)

    monkeypatch.setattr(api_context.qdrant, "scroll", fail_grpc)

    response = api_context.client.get("/documents")

    assert response.status_code == 500
    assert "unavailable" not in response.json()["detail"]


def test_more_uploads_than_the_job_store_retains_are_503_not_evicted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past ingest_jobs_max_retained the store evicted still-queued jobs, so a bulk
    re-index or a big drop handed back job ids that already 404'd — and the UI treats a
    404 while polling as a failed upload, for work that was still going to run."""

    from concurrent.futures import ThreadPoolExecutor

    from api.dependencies import get_ingest_executor

    clear_dependency_caches()
    settings = make_settings(ingest_jobs_max_retained=2)
    monkeypatch.setattr("api.dependencies.get_app_settings", lambda: settings)
    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.dependency_overrides[get_qdrant_client] = lambda: QdrantClient(":memory:")
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
    app.dependency_overrides[get_token_counter] = lambda: HeuristicTokenCounter()
    app.dependency_overrides[get_ollama_reachability_checker] = lambda: lambda _: False
    single = ThreadPoolExecutor(max_workers=1)
    busy = threading.Event()
    single.submit(busy.wait, 5)
    app.dependency_overrides[get_ingest_executor] = lambda: single
    try:
        with TestClient(app) as client:
            accepted = [
                client.post("/documents", files={"file": (f"doc{i}.txt", b"Intro\nalpha")})
                for i in range(3)
            ]
            assert [response.status_code for response in accepted] == [202, 202, 503]
            for response in accepted[:2]:
                job = client.get(f"/documents/jobs/{response.json()['job_id']}")
                assert job.status_code == 200
            busy.set()
    finally:
        busy.set()
        single.shutdown(wait=True)
        monkeypatch.undo()
        clear_dependency_caches()
