"""Exception handlers mapping domain errors to HTTP status codes."""

from __future__ import annotations

import logging

import grpc
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from api.documents import (
    DocumentError,
    DocumentNotFoundError,
)
from api.embeddings import EmbeddingError
from api.errors import (
    VECTOR_STORE_UNAVAILABLE,
    is_vector_store_unavailable,
)
from api.generation import GenerationConfigError, GenerationError
from api.ingestion import IngestionError
from api.jobs import JobNotFoundError
from api.qdrant_schema import CollectionSchemaError, VectorStoreUnavailableError
from api.reranking import RerankingError
from api.retrieval import RetrievalError, RetrievalPayloadError
from api.tracing import TraceNotFoundError
from api.upload import UploadTooLargeError

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register domain exception handlers."""

    app.add_exception_handler(DocumentError, bad_request_handler)
    app.add_exception_handler(UploadTooLargeError, payload_too_large_handler)
    app.add_exception_handler(IngestionError, bad_gateway_handler)
    app.add_exception_handler(RetrievalPayloadError, internal_error_handler)
    app.add_exception_handler(RetrievalError, bad_request_handler)
    app.add_exception_handler(CollectionSchemaError, conflict_handler)
    app.add_exception_handler(VectorStoreUnavailableError, service_unavailable_handler)
    app.add_exception_handler(ResponseHandlingException, vector_store_transport_handler)
    app.add_exception_handler(UnexpectedResponse, vector_store_response_handler)
    # grpcio is a hard dependency of qdrant-client and nothing else here speaks gRPC,
    # so this is effectively "any Qdrant gRPC failure" despite the broad type.
    app.add_exception_handler(grpc.RpcError, vector_store_transport_handler)
    app.add_exception_handler(GenerationConfigError, bad_request_handler)
    app.add_exception_handler(GenerationError, bad_gateway_handler)
    app.add_exception_handler(RerankingError, bad_gateway_handler)
    app.add_exception_handler(EmbeddingError, internal_error_handler)
    app.add_exception_handler(JobNotFoundError, not_found_handler)
    app.add_exception_handler(TraceNotFoundError, not_found_handler)
    app.add_exception_handler(DocumentNotFoundError, not_found_handler)


async def bad_request_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a 400 response for user-correctable request errors."""

    return error_response(400, exc)


async def conflict_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a 409 response for collection/index compatibility errors."""

    return error_response(409, exc)


async def bad_gateway_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a 502 response for provider failures."""

    return error_response(502, exc)


async def service_unavailable_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a 503 response when a required backing service is unreachable."""

    return error_response(503, exc)


async def vector_store_transport_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map a raw Qdrant transport failure to 503 instead of a bare 500.

    Two gaps, one handler. First, VectorRepository.hybrid_search translates
    ResponseHandlingException into VectorStoreUnavailableError itself but no other
    repository method does, and ensure_ready can't cover for them because
    ensure_collection caches success per client — so an outage starting after the
    first successful request made GET /documents, GET /documents/{f}/content and
    DELETE /documents/{f} return bare 500s.

    Second, and found only by running this against a real container: those existing
    translations are REST-only. Under QDRANT_PREFER_GRPC=true — which docker-compose
    sets by default — an unreachable Qdrant raises grpc.RpcError, which
    ResponseHandlingException never matches, so even /questions returned a bare 500 on
    the project's own default deployment path.

    Messages are fixed rather than str(exc): this is a blanket handler over an
    exception type the app doesn't own, so it also catches bugs that merely surface as
    one, where naming the vector store as the cause would be wrong. The real cause is
    logged.
    """

    logger.warning("Vector store transport failure: %s", exc, exc_info=True)
    if is_vector_store_unavailable(exc):
        return JSONResponse(
            status_code=503,
            content={"detail": VECTOR_STORE_UNAVAILABLE},
        )
    # A gRPC error that isn't a transport failure (a malformed query, say) is a bug,
    # not an outage — still redacted and logged, but not mislabeled as unavailability.
    return JSONResponse(
        status_code=500,
        content={"detail": "The request failed due to an unexpected vector store error."},
    )


QDRANT_REJECTED_KEY = "Qdrant rejected the request's credentials; check QDRANT_API_KEY."
QDRANT_SERVER_ERROR = "The vector store failed to handle the request. Try again shortly."
QDRANT_UNEXPECTED_ERROR = "The request failed due to an unexpected vector store error."


async def vector_store_response_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map an error status from Qdrant's REST API to a fixed message instead of a bare 500.

    Repository methods let ``UnexpectedResponse`` through (only a missing collection is
    recovered from), so a wrong ``QDRANT_API_KEY`` or a Qdrant 5xx surfaced as an
    unhandled exception. The real response is logged; the client gets a fixed message.
    """

    status = exc.status_code if isinstance(exc, UnexpectedResponse) else None
    logger.warning("Qdrant answered with an error (HTTP %s).", status, exc_info=exc)
    if status in (401, 403):
        return JSONResponse(status_code=503, content={"detail": QDRANT_REJECTED_KEY})
    if status is not None and status >= 500:
        return JSONResponse(status_code=503, content={"detail": QDRANT_SERVER_ERROR})
    return JSONResponse(status_code=500, content={"detail": QDRANT_UNEXPECTED_ERROR})


async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a 404 response for a referenced resource that doesn't exist."""

    return error_response(404, exc)


async def payload_too_large_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a 413 response when an upload exceeds the configured size limit."""

    return error_response(413, exc)


async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a 500 response for unexpected pipeline state."""

    return error_response(500, exc)


def error_response(status_code: int, exc: Exception) -> JSONResponse:
    """Build a consistent error response."""

    return JSONResponse(status_code=status_code, content={"detail": str(exc)})
