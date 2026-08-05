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

# Bounds for the client-sent conversation history on QuestionRequest. The server
# also has its own conversation_max_history_messages setting for further truncation;
# these are the hard request-shape limits enforced regardless of server config.
REQUEST_HISTORY_MAX_MESSAGES = 12
REQUEST_HISTORY_MESSAGE_MAX_CHARS = 4000


class HealthResponse(BaseModel):
    """Health check response."""

    status: str


class ReadinessResponse(BaseModel):
    """Readiness check response — unlike /health, this probes real dependencies."""

    status: Literal["ok", "degraded"]
    qdrant: bool
    generation_provider: bool
    llm_provider: str


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
    """Completed document ingestion result (also embedded in a finished job status)."""

    filename: str
    sections_parsed: int
    chunks_ingested: int
    collection_name: str


class DocumentJobAcceptedResponse(BaseModel):
    """Response returned immediately when an upload is accepted for background ingestion."""

    job_id: str
    filename: str
    state: Literal["queued"]


class DocumentJobStatusResponse(BaseModel):
    """Current status of a background ingestion job."""

    job_id: str
    filename: str
    state: Literal["queued", "parsing", "embedding", "done", "failed"]
    chunks_total: int
    chunks_done: int
    error: str | None = None
    result: DocumentIngestResponse | None = None


class DocumentListResponse(BaseModel):
    """Filenames currently indexed, for per-document query filtering."""

    filenames: list[str]
    chunk_counts: dict[str, int] = {}


class DocumentDeleteResponse(BaseModel):
    """Result of removing an indexed document's chunks."""

    filename: str
    points_deleted: int


class DocumentChunkResponse(BaseModel):
    """One stored chunk of a document, for the source viewer."""

    chunk_id: str
    page: int
    section: str
    text: str
    chunk_ordinal: int | None = None


class DocumentContentResponse(BaseModel):
    """A document reconstructed from its stored chunks, in ordinal order."""

    filename: str
    chunks: list[DocumentChunkResponse]


class PromptMessageResponse(BaseModel):
    """One message from the exact prompt sent to the generation provider."""

    role: Literal["system", "user", "assistant"]
    content: str


class TraceConfigResponse(BaseModel):
    """Effective per-request settings a trace was generated under."""

    llm_provider: str
    rerank_top_k: int
    max_context_chunks: int
    llm_temperature: float
    rerank_min_score: float
    fused_top_n: int
    filenames: list[str] | None


class TraceCandidateResponse(BaseModel):
    """One fused retrieval candidate's journey through rerank and context selection."""

    point_id: str
    filename: str
    page: int
    section: str
    chunk_id: str
    retrieval_score: float
    rerank_score: float | None
    kept: bool
    drop_reason: Literal["below_min_score", "near_duplicate", "top_k_cut", "not_scored"] | None
    selected_for_context: bool
    neighbor_expanded: bool


class TraceSummaryResponse(BaseModel):
    """One row of the trace list — enough to pick a trace without fetching it."""

    trace_id: str
    created_at: float
    question: str
    mode: Literal["sync", "stream"]
    status: Literal["ok", "insufficient_context", "error"]
    llm_provider: str | None
    total_ms: float | None
    candidate_count: int
    kept_count: int


class TraceListResponse(BaseModel):
    """Recent query traces, newest first."""

    traces: list[TraceSummaryResponse]


class TraceDetailResponse(BaseModel):
    """Full per-query debug trace: candidates, the exact prompt, and timings."""

    trace_id: str
    created_at: float
    question: str
    mode: Literal["sync", "stream"]
    status: Literal["ok", "insufficient_context", "error"]
    config: TraceConfigResponse | None
    candidates: list[TraceCandidateResponse]
    prompt_messages: list[PromptMessageResponse] | None
    answer: str | None
    cited_source_numbers: list[int]
    timings: TimingsResponse | None
    error: str | None
    # None when no history was sent, condense was disabled, or the condense call
    # failed/returned empty and the pipeline fell back to the raw question.
    condensed_question: str | None = None
    history_message_count: int = 0
    # Alternative query phrasings used for multi-query retrieval (empty when disabled).
    query_variants: list[str] = Field(default_factory=list)
    # True when a zero-citation answer was retried and the retry supplied citations.
    citation_retry_used: bool = False


class HistoryMessageRequest(BaseModel):
    """One prior turn of client-sent conversation history."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=REQUEST_HISTORY_MESSAGE_MAX_CHARS)


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
    # Restricts retrieval to these documents only, when given. None/omitted searches
    # the whole collection — the pre-existing behavior.
    filenames: list[str] | None = Field(default=None, min_length=1)
    # Prior turns of the conversation, oldest first. None/omitted (the pre-existing
    # behavior) skips the condense step entirely — the server itself stores no
    # conversation state, the client resends what it wants remembered each request.
    history: list[HistoryMessageRequest] | None = Field(
        default=None, max_length=REQUEST_HISTORY_MAX_MESSAGES
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


class TimingsResponse(BaseModel):
    """Per-stage wall-clock latency, in milliseconds, for one answer."""

    embed_ms: float
    search_ms: float
    rerank_ms: float
    generate_ms: float
    total_ms: float
    # 0.0 when no history was sent, or when CONVERSATION_CONDENSE_ENABLED is off.
    condense_ms: float = 0.0
    # 0.0 when query expansion is disabled (the default) or produced no variants.
    expand_ms: float = 0.0


class QuestionResponse(BaseModel):
    """Grounded answer response."""

    answer: str
    sources: list[CitationResponse]
    timings: TimingsResponse | None = None
    trace_id: str | None = None

