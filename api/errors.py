"""Which exception text may reach a client, and the metric label for each error.

Domain errors carry authored, client-safe messages; anything else may carry internal
detail (a path, a config value, a library's repr) and is replaced by a fixed message.
Shared by the HTTP routes and the trace store — traces are readable with the read-only
key, so a trace's ``error`` must follow the same rule as a response body.
"""

from __future__ import annotations

import grpc
from qdrant_client.http.exceptions import ResponseHandlingException

from api.embeddings import EmbeddingError
from api.generation import GenerationConfigError, GenerationError
from api.qdrant_schema import CollectionSchemaError, VectorStoreUnavailableError
from api.reranking import RerankingError
from api.retrieval import RetrievalConfigError, RetrievalError, RetrievalPayloadError

# Ordered most-specific-first: several of these are subclasses of one another
# (RetrievalPayloadError/RetrievalConfigError < RetrievalError,
# GenerationConfigError < GenerationError), so a parent check must not run first.
QUESTION_ERROR_TYPES: tuple[tuple[type[Exception], str], ...] = (
    (RetrievalPayloadError, "retrieval_payload"),
    (RetrievalConfigError, "retrieval_config"),
    (RetrievalError, "retrieval"),
    (GenerationConfigError, "generation_config"),
    (GenerationError, "generation"),
    (RerankingError, "reranking"),
    (EmbeddingError, "embedding"),
    (VectorStoreUnavailableError, "vector_store_unavailable"),
    (CollectionSchemaError, "collection_schema"),
)

# Derived from QUESTION_ERROR_TYPES above, which has two jobs: it orders the metric
# labels AND decides which messages are safe to return. Adding a type there for
# labeling silently widens what may be sent verbatim, so only add types whose str() is
# an authored, client-safe message.
#
# The streaming route needs this allowlist: /questions lets an unanticipated exception
# fall through to Starlette's default 500 ("Internal Server Error", nothing leaked), but
# /questions/stream has already committed a 200 by the time it can fail, so it must
# catch everything — and without an allowlist that means putting an arbitrary str(exc)
# on the wire.
QUESTION_USER_FACING_ERRORS: tuple[type[Exception], ...] = tuple(
    exc_type for exc_type, _ in QUESTION_ERROR_TYPES
)

# The gRPC status codes that mean "couldn't reach it / gave up waiting" rather than
# "it answered and said no" — the gRPC-transport equivalent of the REST client's
# ResponseHandlingException.
_GRPC_UNAVAILABLE_CODES = frozenset(
    {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED}
)

UNEXPECTED_QUESTION_ERROR = "The question could not be answered due to an unexpected server error."
VECTOR_STORE_UNAVAILABLE = "The vector store is unavailable. Try again shortly."


def question_error_type(exc: BaseException) -> str:
    """Coarse error_type label for docrag_questions_total{outcome="error"}."""

    for exc_type, label in QUESTION_ERROR_TYPES:
        if isinstance(exc, exc_type):
            return label
    return "unexpected"


def is_vector_store_unavailable(exc: BaseException) -> bool:
    if isinstance(exc, ResponseHandlingException):
        return True
    # grpc.RpcError itself declares no code(); the concrete _InactiveRpcError raised in
    # practice does, so probe rather than assume.
    code = getattr(exc, "code", None)
    return callable(code) and code() in _GRPC_UNAVAILABLE_CODES


def public_error_message(exc: BaseException) -> str:
    """The text a client (or a trace reader) may see for a failed question."""

    if isinstance(exc, QUESTION_USER_FACING_ERRORS):
        return str(exc)
    # A raw transport failure carries no authored message, but "unexpected server
    # error" would hide an actionable cause the synchronous route names properly.
    if is_vector_store_unavailable(exc):
        return VECTOR_STORE_UNAVAILABLE
    return UNEXPECTED_QUESTION_ERROR
