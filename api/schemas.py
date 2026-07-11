"""FastAPI request and response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Bounds for per-question retrieval/generation overrides accepted by QuestionRequest.
# rrf_k and fused_top_n stay server-only (hybrid retrieval / RRF fusion is spec-pinned
# by CLAUDE.md) — only these three fields are ever user-adjustable per request.
REQUEST_RERANK_TOP_K_MAX = 50
REQUEST_MAX_CONTEXT_CHUNKS_MAX = 20
REQUEST_TEMPERATURE_MIN = 0.0
REQUEST_TEMPERATURE_MAX = 2.0


class HealthResponse(BaseModel):
    """Health check response."""

    status: str


class PublicConfigResponse(BaseModel):
    """Non-secret runtime configuration exposed to the UI."""

    qdrant_collection: str
    dense_embedding_model: str
    sparse_embedding_model: str
    reranker_model: str
    embedding_model_tag: str
    llm_provider: str
    llm_model: str
    llm_temperature: float
    openai_model: str
    openai_available: bool
    ollama_available: bool
    rrf_k: int
    dense_retrieval_limit: int
    sparse_retrieval_limit: int
    fused_top_n: int
    rerank_top_k: int
    max_context_chunks: int
    rerank_min_score: float
    max_upload_bytes: int
    # Real ceilings the frontend should clamp sliders to, so a user can never submit
    # an override the backend would reject. rerank_top_k_limit is capped by the
    # server's own fused_top_n (RRF can't surface more candidates than that).
    rerank_top_k_limit: int
    max_context_chunks_limit: int
    llm_temperature_max: float


class DocumentIngestResponse(BaseModel):
    """Document upload and ingestion response."""

    filename: str
    sections_parsed: int
    chunks_ingested: int
    collection_name: str


class QuestionRequest(BaseModel):
    """Question-answering request."""

    question: str = Field(min_length=1, max_length=4000)
    llm_provider: Literal["ollama", "openai"] | None = None
    rerank_top_k: int | None = Field(default=None, ge=1, le=REQUEST_RERANK_TOP_K_MAX)
    max_context_chunks: int | None = Field(
        default=None, ge=1, le=REQUEST_MAX_CONTEXT_CHUNKS_MAX
    )
    llm_temperature: float | None = Field(
        default=None, ge=REQUEST_TEMPERATURE_MIN, le=REQUEST_TEMPERATURE_MAX
    )

    @model_validator(mode="after")
    def _validate_context_within_rerank(self) -> QuestionRequest:
        # Shape-level rule (both fields present and internally inconsistent) -> 422
        # via FastAPI's built-in validation error handler. This is distinct from the
        # fused_top_n business rule in RagPipeline.answer, which depends on live
        # server config and therefore surfaces as a 400 through RetrievalConfigError.
        if (
            self.rerank_top_k is not None
            and self.max_context_chunks is not None
            and self.max_context_chunks > self.rerank_top_k
        ):
            raise ValueError("max_context_chunks cannot exceed rerank_top_k.")
        return self


class CitationResponse(BaseModel):
    """Citation metadata returned with an answer."""

    source_number: int
    filename: str
    page: int
    section: str
    chunk_id: str
    text: str


class QuestionResponse(BaseModel):
    """Grounded answer response."""

    answer: str
    sources: list[CitationResponse]

