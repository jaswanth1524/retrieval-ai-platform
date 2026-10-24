"""FastAPI request and response schemas."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator

from api.citations import ANSWER_STATUSES
from api.documents import MAX_TAGS_PER_DOCUMENT
from api.settings import SUMMARY_DOCUMENTS_LIMIT, AnswerFormat, normalize_answer_language
from api.tracing import TRACE_DROP_REASONS

# Bounds for per-question retrieval/generation overrides accepted by QuestionRequest.
# rrf_k and fused_top_n stay server-only (hybrid retrieval / RRF fusion is spec-pinned
# by CLAUDE.md) — only these fields are ever user-adjustable per request.
REQUEST_RERANK_TOP_K_MAX = 50
REQUEST_MAX_CONTEXT_CHUNKS_MAX = 20
REQUEST_TEMPERATURE_MIN = 0.0
REQUEST_TEMPERATURE_MAX = 2.0
# One question's text (and a search query, and a rated question). The frontend's
# MAX_QUESTION_CHARS (src/utils/limits.ts) must match; tests/test_schemas_limits.py checks.
MAX_QUESTION_CHARS = 4000
# Providers a question may name per request (routes/questions.py's allowlist reads these).
RequestProvider = Literal["ollama", "openai", "openai_compatible", "anthropic"]

# Bounds for the client-sent conversation history on QuestionRequest. The server
# also has its own conversation_max_history_messages setting for further truncation;
# these are the hard request-shape limits enforced regardless of server config.
REQUEST_HISTORY_MAX_MESSAGES = 12
REQUEST_HISTORY_MESSAGE_MAX_CHARS = 4000

# Upper bound on the per-request search scope. Generous relative to any realistic
# self-hosted corpus — this exists to stop an unbounded client-supplied list, not to
# constrain legitimate scoping.
REQUEST_FILENAMES_MAX = 500
# A section filter on POST /search; stored labels are capped well below it
# (parsers.common.MAX_SECTION_CHARS).
SEARCH_SECTION_MAX_CHARS = 200

# Per-item caps on client-sent lists, which were capped only in count: one feedback row
# could carry ~1 MB of filenames, and question scopes are copied into every trace and
# answer-cache key. Generous on purpose: uploads cap names at 255 bytes, but documents
# indexed before that may carry longer ones; stored tags are at most 40 characters, but
# question tags are whitespace-collapsed before matching, so a padded one still works.
REQUEST_FILENAME_MAX_CHARS = 1024
REQUEST_TAG_MAX_CHARS = 128
# Trace ids are 32 hex characters (uuid4().hex).
REQUEST_TRACE_ID_MAX_CHARS = 64
# Feedback limits, shared with a backup restore (api/restore.py) so a restored row is one
# POST /feedback could have stored. A citation number is an answer's [n]: a few dozen at
# most, and an unbounded int overflowed sqlite's INTEGER into a 500.
FEEDBACK_EXCERPT_MAX_CHARS = 2000
FEEDBACK_CITED_FILENAMES_MAX = 50
FEEDBACK_CITATION_NUMBER_MAX = 1000
# Passages a question or search may be narrowed to (picked in the Search panel). Chunk
# ids are 20 hex characters (documents.build_chunk_id).
REQUEST_CHUNK_IDS_MAX = 100
REQUEST_CHUNK_ID_MAX_CHARS = 64
RequestFilename = Annotated[str, StringConstraints(max_length=REQUEST_FILENAME_MAX_CHARS)]
RequestTag = Annotated[str, StringConstraints(max_length=REQUEST_TAG_MAX_CHARS)]
RequestChunkId = Annotated[
    str, StringConstraints(min_length=1, max_length=REQUEST_CHUNK_ID_MAX_CHARS)
]


class PassageNarrowing(BaseModel):
    """Where in the documents to look, shared by POST /search and questions. Every field
    narrows the same hybrid search, fusion and rerank; none replaces them."""

    # Passages starting on a page in this range (either end optional).
    page_min: int | None = Field(default=None, ge=1)
    page_max: int | None = Field(default=None, ge=1)
    # Passages under exactly this section label (as citations show it).
    section: str | None = Field(default=None, min_length=1, max_length=SEARCH_SECTION_MAX_CHARS)
    # Documents to leave out.
    exclude_filenames: list[RequestFilename] | None = Field(
        default=None, min_length=1, max_length=REQUEST_FILENAMES_MAX
    )
    # Only these passages, by chunk id.
    chunk_ids: list[RequestChunkId] | None = Field(
        default=None, min_length=1, max_length=REQUEST_CHUNK_IDS_MAX
    )

    @model_validator(mode="after")
    def _page_range_in_order(self) -> PassageNarrowing:
        if self.page_min is not None and self.page_max is not None:
            if self.page_min > self.page_max:
                raise ValueError("page_min must not be greater than page_max.")
        return self


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    # The running version (pyproject.toml), open like the rest of /health so an operator
    # can tell what is deployed without a key.
    version: str = ""


class ReadinessResponse(BaseModel):
    """Readiness check response — unlike /health, this probes real dependencies."""

    status: Literal["ok", "degraded"]
    qdrant: bool
    generation_provider: bool
    llm_provider: str
    # False when the collection was built for another embedding setup: questions and
    # uploads are refused (409) while listing, export and delete still work. Not part
    # of `status` — the app is up, and the UI is how the operator recovers.
    index_compatible: bool = True
    index_detail: str | None = None
    # Whether uploaded originals are kept (RAW_DOCUMENT_DIR): only then can a backup
    # rebuild the index, so the UI offers that way out of a mismatch only then.
    originals_kept: bool = False
    # Each model the startup warmup loaded: "ok", "failed" (questions will fail until a
    # background retry loads it; makes `status` degraded) or "fallback" (the tokenizer
    # fell back to the heuristic counter; never degraded). Empty when warmup is off.
    models: dict[str, Literal["ok", "failed", "fallback"]] = Field(default_factory=dict)


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
    feedback_enabled: bool
    # RAW_DOCUMENT_DIR is set: originals are stored, so documents can be re-indexed.
    raw_documents_enabled: bool = False
    # OPENAI_COMPATIBLE_BASE_URL and _MODEL are set (the server itself isn't probed).
    openai_compatible_available: bool = False
    openai_compatible_model: str = ""
    # WORKSPACES_ENABLED: requests may name a workspace (X-Workspace).
    workspaces_enabled: bool = False
    # URL_INGEST_ENABLED: POST /documents/from-url is available.
    url_ingest_enabled: bool = False
    # ANTHROPIC_API_KEY is set (the key itself is never checked here).
    anthropic_available: bool = False
    anthropic_model: str = ""
    # Providers a question may choose per request (ALLOWED_REQUEST_PROVIDERS, plus
    # LLM_PROVIDER). Whether each is reachable is the *_available fields above.
    allowed_request_providers: list[str] = Field(
        default_factory=lambda: ["ollama", "openai", "openai_compatible", "anthropic"]
    )
    # File suffixes an upload may have (".pdf", ".epub", ...), so the UI's file picker
    # and its pre-upload check follow the server instead of a hard-coded copy.
    supported_extensions: list[str] = Field(default_factory=list)
    # WATCH_DIR is set: GET /watch and POST /watch/scan are available.
    watch_enabled: bool = False
    # ANSWER_LANGUAGE ("" = the model's choice, "match" = the question's language).
    answer_language: str = ""
    # SUMMARY_MAX_DOCUMENTS: the most documents one brief may cover.
    summary_max_documents: int = 5
    # ANSWER_FORMAT: how answers are laid out unless a question asks otherwise.
    answer_format: str = "default"
    # QUERY_LOG_ENABLED: GET /query-log (the questions the documents didn't answer).
    query_log_enabled: bool = False
    # RAW_DOCUMENT_VERSIONS > 0 with RAW_DOCUMENT_DIR: replaced originals are kept, so
    # GET /documents/{f}/versions and their restore are available.
    document_versions_enabled: bool = False
    # TRACE_STORE_PATH: traces are kept on disk (GET /traces/export has them all).
    traces_persisted: bool = False


class DocumentIngestResponse(BaseModel):
    """Completed document ingestion result (also embedded in a finished job status)."""

    filename: str
    sections_parsed: int
    chunks_ingested: int
    collection_name: str
    # The document was already indexed from these exact bytes, with current chunking
    # settings: nothing was parsed, embedded or written.
    unchanged: bool = False


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
    # True for a job stopped by DELETE /documents/jobs/{job_id} (state "failed").
    cancelled: bool = False
    # When the job was accepted (Unix time).
    created_at: float | None = None
    # True for a job that found the document already indexed from identical bytes and
    # left it as it was (state "done", nothing re-embedded).
    unchanged: bool = False


class DocumentJobListResponse(BaseModel):
    """A workspace's queued and running ingest jobs, oldest first: uploads from any
    client, the watched folder, a bulk re-index or a restore."""

    jobs: list[DocumentJobStatusResponse]


class DocumentListResponse(BaseModel):
    """Filenames currently indexed, for per-document query filtering."""

    filenames: list[str]
    chunk_counts: dict[str, int] = {}
    page_counts: dict[str, int] = {}
    # A filename ingested before byte_size/uploaded_at were stamped at ingest time
    # (api.documents.DocumentChunk) is simply absent from these two — not present with
    # a null value — since neither field exists in any of its points' payloads.
    byte_sizes: dict[str, int] = {}
    uploaded_ats: dict[str, float] = {}
    # Indexed by an older chunker (or before the version was recorded) — re-indexing
    # them applies the current chunking.
    stale_filenames: list[str] = []
    # Have a stored original (RAW_DOCUMENT_DIR), so POST /documents/{f}/reindex works.
    reindexable_filenames: list[str] = []
    # Tagged documents only (PATCH /documents/{f}/tags); an untagged one is absent.
    tags: dict[str, list[str]] = {}
    # Documents fetched by URL (POST /documents/from-url): where from, without its query
    # string (POST /documents/{f}/refresh uses the full one). Uploads are absent.
    source_urls: dict[str, str] = {}


class DocumentVersionResponse(BaseModel):
    version: str
    # When it was replaced (Unix time).
    saved_at: float
    size: int


class DocumentVersionsResponse(BaseModel):
    """Earlier versions of a document's original (RAW_DOCUMENT_VERSIONS), newest first."""

    filename: str
    versions: list[DocumentVersionResponse]


class DocumentRefreshResponse(BaseModel):
    """POST /documents/{f}/refresh when the source said nothing changed (304)."""

    filename: str
    unchanged: bool = True


class DocumentTagsRequest(BaseModel):
    """Replace a document's tags. An empty list removes them all."""

    tags: list[str] = Field(max_length=MAX_TAGS_PER_DOCUMENT)


class DocumentTagsResponse(BaseModel):
    filename: str
    tags: list[str]


class DocumentReindexAllResponse(BaseModel):
    """Result of re-indexing every stale document that has a stored original."""

    jobs: list[DocumentJobAcceptedResponse]
    # Stale and re-indexable, but the ingest backlog was full; retry later.
    deferred: list[str] = []
    # Already being indexed (an earlier re-index, or an upload): not queued twice.
    in_progress: list[str] = []
    # Could not be queued (logged on the server); the others were.
    failed: list[str] = []


class DocumentDeleteResponse(BaseModel):
    """Result of removing an indexed document's chunks."""

    filename: str
    points_deleted: int


class FeedbackRequest(BaseModel):
    """A thumbs up/down rating of an assistant answer, optionally one citation.

    Denormalized rather than trace_id-only: the client already holds the question,
    answer, and cited filenames for the turn being rated, and TraceStore's ring
    buffer can evict trace_id before this ever reaches durable storage.
    """

    trace_id: str | None = Field(default=None, max_length=REQUEST_TRACE_ID_MAX_CHARS)
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    answer_excerpt: str = Field(min_length=1, max_length=FEEDBACK_EXCERPT_MAX_CHARS)
    cited_filenames: list[RequestFilename] = Field(
        default_factory=list, max_length=FEEDBACK_CITED_FILENAMES_MAX
    )
    rating: Literal["up", "down"]
    # Set only when the rating targets one specific citation (its [n] source_number
    # as shown in the answer), rather than the answer as a whole.
    citation_source_number: int | None = Field(default=None, ge=1, le=FEEDBACK_CITATION_NUMBER_MAX)
    # What was rated: an answer to a question (the default, and all a client that
    # predates this sends), a suggested follow-up question, or a summary/comparison.
    # Only "question" ratings become retrieval labels (eval/feedback_dataset.py).
    kind: Literal["question", "follow_up", "summary"] = "question"


class FeedbackResponse(BaseModel):
    """Confirmation that a feedback rating was recorded."""

    id: str


FEEDBACK_LIST_MAX_LIMIT = 500


class FeedbackItemResponse(BaseModel):
    """One recorded rating, as stored — the denormalized row, not a trace_id join."""

    id: str
    trace_id: str | None
    question: str
    answer_excerpt: str
    cited_filenames: list[str]
    rating: Literal["up", "down"]
    citation_source_number: int | None
    created_at: float
    kind: Literal["question", "follow_up", "summary"] = "question"


class FeedbackListResponse(BaseModel):
    """Recently recorded ratings, newest first."""

    feedback: list[FeedbackItemResponse]


class DocumentChunkResponse(BaseModel):
    """One stored chunk of a document, for the source viewer."""

    chunk_id: str
    page: int
    section: str
    text: str
    chunk_ordinal: int | None = None
    # The last page when the passage crosses a page break (it starts on `page`).
    page_end: int | None = None


class DocumentContentResponse(BaseModel):
    """A document reconstructed from its stored chunks, in ordinal order."""

    filename: str
    chunks: list[DocumentChunkResponse]
    # Every chunk the document has; with `around` or `start`/`end`, `chunks` is a page.
    total_chunks: int | None = None
    # With `around`: whether that chunk still exists (a re-index changes chunk ids).
    target_found: bool | None = None


class TimingsResponse(BaseModel):
    """Per-stage wall-clock latency, in milliseconds, for one answer."""

    embed_ms: float
    search_ms: float
    rerank_ms: float
    generate_ms: float
    total_ms: float
    # 0.0 when no history was sent, or when CONVERSATION_CONDENSE_ENABLED is off.
    condense_ms: float = 0.0
    # LLM generation of alternative query phrasings, which runs BEFORE embedding. 0.0
    # when QUERY_EXPANSION_ENABLED is off (the default) or it produced no variants.
    query_expansion_ms: float = 0.0
    # Neighbour/context expansion after rerank (small-to-big). 0.0 when
    # CONTEXT_NEIGHBOR_RADIUS is 0. Previously folded into rerank_ms.
    context_expansion_ms: float = 0.0
    # The zero-citation retry's LLM round trip. 0.0 when it didn't fire, which is the
    # common case. generate_ms now excludes it in both response modes.
    citation_retry_ms: float = 0.0
    # Part of rerank_ms spent queued behind another question's rerank (the model scores
    # one question at a time). 0.0 when nothing was ahead of it.
    rerank_wait_ms: float = 0.0
    # Request start to the first streamed answer token. 0.0 on /questions (nothing
    # streams) and on answers served from the cache.
    first_token_ms: float = 0.0


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
    # The tag scope the question was asked under; None when it wasn't tag-scoped.
    tags: list[str] | None = None
    reranker_model: str | None = None
    answer_format: str = "default"
    answer_language: str = ""
    # The question's passage narrowing (page_min, page_max, section, chunk_ids,
    # exclude_filenames — only the ones given); None when it had none.
    narrowing: dict[str, Any] | None = None
    # The model the provider was asked for (LiteLLM's name).
    llm_model: str | None = None


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
    # Declared extensible: traces gain drop reasons as retrieval gains stages, and a
    # client should show an unknown one as-is (the UI does) rather than reject it.
    drop_reason: str | None = Field(
        json_schema_extra={"x-extensible-enum": list(TRACE_DROP_REASONS)}
    )
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


class QueryLogEntryResponse(BaseModel):
    """One logged question or summary (QUERY_LOG_ENABLED)."""

    created_at: float
    question: str
    # The trace's status: ok, insufficient_context (not answered from the documents),
    # error. Extensible.
    status: str
    # Best passage score (0-1) the question found; None for a summary.
    top_score: float | None = None
    mode: str
    trace_id: str | None = None


class QueryLogResponse(BaseModel):
    queries: list[QueryLogEntryResponse]


class UsageResponse(BaseModel):
    """Tokens the provider reported for an answer's LLM calls: the condense, query
    expansion, the answer, its citation retry, a summary's notes, and (on POST
    /questions) the follow-ups; a stream's `done` comes before its follow-ups."""

    prompt_tokens: int
    completion_tokens: int
    # An estimate at the hosted model's list price (LiteLLM's bundled table); absent for
    # local providers and models it doesn't list.
    cost_usd: float | None = None


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
    # Why condense did not run, or did not use the model's output as returned:
    # "standalone", "disabled", "failed", "empty", "cleaned", "rejected: ...".
    condense_note: str | None = None
    # The kept candidates scored under RERANK_MIN_SCORE (the weak-match fallback).
    weak_matches: bool = False
    history_message_count: int = 0
    # Alternative query phrasings used for multi-query retrieval (empty when disabled).
    query_variants: list[str] = Field(default_factory=list)
    # True when a zero-citation answer was retried and the retry supplied citations.
    citation_retry_used: bool = False
    # True when a zero-citation answer had citations added by matching its sentences to
    # the sources (CITATION_RETRY_MODE), without a second LLM call.
    citation_backfilled: bool = False
    # The X-Request-ID of the request that asked it, to find its log lines.
    request_id: str | None = None
    usage: UsageResponse | None = None


class HistoryMessageRequest(BaseModel):
    """One prior turn of client-sent conversation history."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=REQUEST_HISTORY_MESSAGE_MAX_CHARS)


class QuestionRequest(PassageNarrowing):
    """Question-answering request."""

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    llm_provider: RequestProvider | None = None
    rerank_top_k: int | None = Field(default=None, ge=1, le=REQUEST_RERANK_TOP_K_MAX)
    max_context_chunks: int | None = Field(default=None, ge=1, le=REQUEST_MAX_CONTEXT_CHUNKS_MAX)
    llm_temperature: float | None = Field(
        default=None, ge=REQUEST_TEMPERATURE_MIN, le=REQUEST_TEMPERATURE_MAX
    )
    # Restricts retrieval to these documents only, when given. None/omitted searches
    # the whole collection — the pre-existing behavior. Capped like every other
    # list-valued request field here: it becomes a Qdrant MatchAny filter that gets
    # rebuilt once per query variant when query expansion is on, so an unbounded list
    # is unbounded server-side work driven entirely by the client.
    filenames: list[RequestFilename] | None = Field(
        default=None, min_length=1, max_length=REQUEST_FILENAMES_MAX
    )
    # Prior turns of the conversation, oldest first. None/omitted (the pre-existing
    # behavior) skips the condense step entirely — the server itself stores no
    # conversation state, the client resends what it wants remembered each request.
    history: list[HistoryMessageRequest] | None = Field(
        default=None, max_length=REQUEST_HISTORY_MAX_MESSAGES
    )
    # Restricts retrieval to documents carrying any of these tags (and, with
    # `filenames`, also to those files).
    tags: list[RequestTag] | None = Field(
        default=None, min_length=1, max_length=MAX_TAGS_PER_DOCUMENT
    )
    # False skips the answer cache's lookup (Regenerate); the new answer is still cached.
    use_cache: bool = True
    # This answer's layout and language, over ANSWER_FORMAT / ANSWER_LANGUAGE.
    answer_format: AnswerFormat | None = None
    answer_language: str | None = Field(default=None, max_length=40)

    @field_validator("answer_language", mode="after")
    @classmethod
    def _check_answer_language(cls, value: str | None) -> str | None:
        return None if value is None else normalize_answer_language(value)

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


class CitationSpanResponse(BaseModel):
    """The passage of a cited chunk an answer's claim rests on (see api/citations.py)."""

    # Code-point offsets into the chunk's full text (GET /documents/{f}/content).
    start: int
    end: int
    # The passage itself: a client indexing strings differently finds it by search.
    quote: str


class CitationResponse(BaseModel):
    """Citation metadata returned with an answer."""

    source_number: int
    filename: str
    page: int
    section: str
    chunk_id: str
    text: str
    # Empty in the `sources` frame (sent before the answer exists) and when no passage
    # matched the answer closely enough to point at.
    spans: list[CitationSpanResponse] = []
    # The last page when the passage crosses a page break (it starts on `page`).
    page_end: int | None = None


class QuestionResponse(BaseModel):
    """Grounded answer response."""

    answer: str
    sources: list[CitationResponse]
    timings: TimingsResponse | None = None
    trace_id: str | None = None
    # Served from the answer cache: no retrieval or generation ran for this request.
    cached: bool = False
    # "no_context": retrieval found nothing to use (nothing cleared the rerank bar, with
    # RERANK_FALLBACK_COUNT=0); "insufficient_context": the model said the documents
    # don't cover the question (citing nothing, or citing only to say so); "uncited": it
    # answered without citing a source. Extensible: show an unknown value as "answered".
    answer_status: str = Field(
        default="answered", json_schema_extra={"x-extensible-enum": list(ANSWER_STATUSES)}
    )
    # Best cross-encoder score (0-1) among the candidates, whether or not it was kept.
    top_rerank_score: float | None = None
    # No passage cleared RERANK_MIN_SCORE: the answer comes from the closest ones
    # (RERANK_FALLBACK_COUNT), which the model was told match the question only weakly.
    weak_matches: bool = False
    # The provider stopped at the token limit (LLM_MAX_TOKENS): the answer is cut off.
    truncated: bool = False
    # The model cited nothing; its citations were added by matching its sentences to the
    # sources (CITATION_RETRY_MODE), so check them ("auto-cited" in the UI).
    citation_backfilled: bool = False
    # Suggested next questions (FOLLOWUPS_ENABLED); empty when off or none were found.
    followups: list[str] = []
    # None when the provider reported no usage, and on a cache hit.
    usage: UsageResponse | None = None


class WorkspaceCreateRequest(BaseModel):
    """Create a workspace (POST /workspaces); see api/workspaces.py for the name rule."""

    name: str = Field(min_length=1, max_length=64)


class WorkspaceResponse(BaseModel):
    name: str


class WorkspaceListResponse(BaseModel):
    """The workspaces besides the default one (which has no name and is always there)."""

    enabled: bool
    workspaces: list[str]


class UrlIngestRequest(BaseModel):
    """Index a document fetched from a URL (POST /documents/from-url)."""

    url: str = Field(min_length=1, max_length=2048)
    # The stored name; else the server's Content-Disposition, else the URL's last path
    # segment (with an extension from the content type when it names no supported one).
    filename: RequestFilename | None = None


class SummaryRequest(BaseModel):
    """Summarize one whole document, compare two, or write one brief across several
    (POST /questions/summary)."""

    mode: Literal["summarize", "compare", "brief"]
    # One document to summarize, two (different ones) to compare, two up to
    # SUMMARY_MAX_DOCUMENTS (different ones) for a brief.
    filenames: list[RequestFilename] = Field(min_length=1, max_length=SUMMARY_DOCUMENTS_LIMIT)
    # What to concentrate on ("deadlines", "risks"); omitted for a general summary.
    instruction: str | None = Field(default=None, max_length=MAX_QUESTION_CHARS)
    llm_provider: RequestProvider | None = None
    llm_temperature: float | None = Field(
        default=None, ge=REQUEST_TEMPERATURE_MIN, le=REQUEST_TEMPERATURE_MAX
    )
    # False skips the answer cache's lookup (Regenerate); the new summary is still cached.
    use_cache: bool = True

    @model_validator(mode="after")
    def _validate_document_count(self) -> SummaryRequest:
        distinct = len(set(self.filenames))
        if distinct != len(self.filenames):
            raise ValueError("each document may be named once")
        if self.mode == "compare" and distinct != 2:
            raise ValueError("compare takes two different documents")
        if self.mode == "summarize" and distinct != 1:
            raise ValueError("summarize takes one document")
        if self.mode == "brief" and distinct < 2:
            raise ValueError("a brief takes two documents or more")
        return self


class SummaryResponse(QuestionResponse):
    """A summary or comparison: an answer, plus how much of the documents it read."""

    # Passages read, of the passages the documents hold. Fewer when a long document was
    # read at evenly spaced passages (SUMMARY_MAX_BATCHES parts per request).
    chunks_read: int = 0
    chunks_total: int = 0


class AccessResponse(BaseModel):
    """What the caller's key unlocks: everything ("full"), reading and asking only
    ("read", API_READ_KEY), or everything because no key is configured ("open")."""

    access: Literal["full", "read", "open"]


class SearchRequest(PassageNarrowing):
    """Retrieval without an answer: the passages a question would be answered from."""

    query: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    filenames: list[RequestFilename] | None = Field(
        default=None, min_length=1, max_length=REQUEST_FILENAMES_MAX
    )
    tags: list[RequestTag] | None = Field(
        default=None, min_length=1, max_length=MAX_TAGS_PER_DOCUMENT
    )
    # How many reranked passages to return; capped by the server's fused_top_n (400).
    rerank_top_k: int | None = Field(default=None, ge=1, le=REQUEST_RERANK_TOP_K_MAX)


class SearchResultResponse(BaseModel):
    """One passage, best first, with the scores that ranked it."""

    rank: int
    filename: str
    page: int
    section: str
    chunk_id: str
    chunk_ordinal: int | None
    text: str
    # RRF-fused hybrid score, then the cross-encoder's score normalized to [0, 1].
    retrieval_score: float
    rerank_score: float
    # The last page when the passage crosses a page break (it starts on `page`).
    page_end: int | None = None


class SearchTimingsResponse(BaseModel):
    embed_ms: float
    search_ms: float
    rerank_ms: float


class SearchResponse(BaseModel):
    results: list[SearchResultResponse]
    timings: SearchTimingsResponse
    # No passage cleared RERANK_MIN_SCORE (the bar an answer's sources must clear); the
    # results are the closest ones, best first. Typical of a one- or two-word search,
    # which the cross-encoder scores low even against a passage containing the words.
    weak_matches: bool = False


class ImportResponse(BaseModel):
    """What POST /import did with each document in the backup."""

    # Indexing jobs started, one per restored document (poll like an upload's).
    jobs: list[DocumentJobAcceptedResponse]
    # Not started because the indexing queue was full; import the same file again later.
    deferred: list[str]
    # Already indexed here, so left as they are.
    skipped: list[str]
    # Listed in the manifest without a stored original (RAW_DOCUMENT_DIR was off).
    missing_originals: list[str]
    # Unusable entries: an invalid name, or an original over MAX_UPLOAD_BYTES.
    rejected: list[str]
    feedback_imported: int
    feedback_skipped: int


class WatchRetry(BaseModel):
    """A watched file waiting to be tried again after a failure."""

    filename: str
    attempts: int
    next_attempt_at: float
    error: str


class WatchStatusResponse(BaseModel):
    """The watched folder (WATCH_DIR) as of the watcher's last pass."""

    folder: str
    # Unix time of the last pass; None before the first.
    last_scan_at: float | None = None
    files: int = 0
    problems: list[str] = []
    deferred: list[str] = []
    retrying: list[WatchRetry] = []
    # Refused for good, or out of retries: tried again when the file changes.
    gave_up: list[str] = []
