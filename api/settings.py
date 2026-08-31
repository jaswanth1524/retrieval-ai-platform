"""Application settings loaded from environment variables."""

from typing import Literal

from pydantic import Field, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_VALID_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})
# OpenAI reasoning models (gpt-5 family, o-series) accept a reasoning_effort knob;
# "" disables sending it (uses the provider default).
_VALID_REASONING_EFFORTS = frozenset({"", "minimal", "low", "medium", "high"})


class AppSettings(BaseSettings):
    """Runtime settings shared by backend components."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docrag_documents"
    qdrant_dense_vector_name: str = "dense"
    qdrant_sparse_vector_name: str = "sparse"
    qdrant_dense_vector_size: PositiveInt = 384
    # gRPC avoids per-call JSON (de)serialization overhead versus Qdrant's REST API —
    # meaningful at query volume, free to enable since the client already exposes both
    # transports over the same connection details.
    qdrant_prefer_grpc: bool = False

    dense_embedding_model: str = "BAAI/bge-small-en-v1.5"
    sparse_embedding_model: str = "Qdrant/BM25"
    # Prepended to the dense query text only (never the corpus side) before embedding.
    # bge-small's model card recommends this instruction for short-query→passage
    # retrieval, and fastembed does not apply it for us. Empty string disables it.
    # Query-side only, so changing it needs no re-ingest.
    dense_query_instruction: str = "Represent this sentence for searching relevant passages: "
    reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual"
    embedding_model_tag: str = "fastembed:BAAI/bge-small-en-v1.5"

    rrf_k: PositiveInt = 60
    dense_retrieval_limit: PositiveInt = 50
    sparse_retrieval_limit: PositiveInt = 50
    fused_top_n: PositiveInt = 50
    # How many of the fused top-N candidates are actually sent through the
    # cross-encoder — the reranker is the single most expensive retrieval-side stage,
    # so this lets an operator trade candidate coverage for latency without touching
    # fused_top_n (which stays "the fusion spec's own top-N" per CLAUDE.md). Clamped to
    # fused_top_n at the call site — it can never rerank more than fusion produced.
    rerank_candidates: PositiveInt = 50
    rerank_top_k: PositiveInt = 8
    # Caps how many query-document pairs the cross-encoder scores per forward pass, which
    # is what bounds its peak memory: attention cost grows with batch x sequence^2, and at
    # CHUNK_SIZE_TOKENS=448 the sequences are long.
    #
    # Peak memory is a function of THIS VALUE ALONE, not of its ratio to anything else.
    # Measured over 50 candidates x ~448 tokens: batch 8 peaked at 2.66 GB (exactly the
    # loaded-model baseline — scoring adds nothing on top), batch 16 at 3.53 GB, batch 50
    # at 6.33 GB, which OOM-killed (exit 137) the container on a default ~6 GB Docker
    # Desktop the first time anyone asked a whole-corpus question. Latency was flat across
    # the whole range (~45-49s), so a larger batch buys no speed and no answer quality —
    # it only changes how the same work is chunked, and what it peaks at while doing so.
    # Raise it only against known memory headroom; api/main.py logs a warning at startup
    # past the measured-safe point.
    #
    # Keeping it <= rerank_candidates is a SEPARATE and weaker rule: above that, batching
    # is simply a no-op because there aren't enough pairs to fill a batch. That relation
    # is not the memory guard, and satisfying it is not sufficient — batch 50 against 50
    # candidates satisfies it and still peaks at 6.33 GB. The historical incident (a
    # default of 64 against 50 candidates) happened to violate both rules at once, which
    # is what made the relation look like the cause.
    reranker_batch_size: PositiveInt = 8
    max_context_chunks: PositiveInt = 6
    # Cross-encoder logits are unbounded and model-specific; sigmoid-normalizing to
    # [0, 1] before comparing against this threshold makes it comparable across
    # reranker models. sigmoid(0)=0.5 is the "indifferent" point, but in practice
    # jina-reranker-v2's scores for genuinely relevant pairs run well below that (a
    # clean direct-hit query scored ~0.6, a paraphrase-only match ~0.2) — 0.30 was
    # too strict and returned "insufficient context" even when loosely relevant
    # content existed. 0.15 still blocks clearly off-topic chunks (scored <0.05 in
    # the same real-world trace) while letting paraphrase-level matches through.
    rerank_min_score: float = Field(default=0.15, ge=0.0, le=1.0)

    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: PositiveInt = 512
    llm_request_timeout_seconds: float = Field(default=60.0, gt=0.0)
    llm_num_retries: int = Field(default=0, ge=0)
    ollama_base_url: str = "http://localhost:11434"
    # Ollama unloads a model from memory 5 minutes after its last use by default,
    # forcing a multi-second reload on the next question. Keeping it resident trades
    # idle RAM for eliminating that reload on every request after the first.
    ollama_keep_alive: str = "30m"
    openai_api_key: str | None = Field(default=None)
    openai_model: str = "gpt-4o-mini"
    # Reasoning models (gpt-5 family) spend hidden reasoning tokens out of the same
    # max_completion_tokens budget as the visible answer. "low" keeps that spend modest
    # while still reasoning; "minimal" nearly eliminates it. "" leaves the provider
    # default. Only applied to models LiteLLM reports as reasoning-capable.
    openai_reasoning_effort: str = "low"
    # Extra output-token budget added ON TOP of llm_max_tokens for reasoning models, so
    # hidden reasoning tokens don't starve the visible answer (a 512 budget is entirely
    # consumed by reasoning on non-trivial questions, yielding an empty answer). Ignored
    # for non-reasoning models, which keep the plain llm_max_tokens budget.
    reasoning_token_headroom: PositiveInt = 3072

    max_upload_bytes: PositiveInt = 50 * 1024 * 1024
    # Real dense-model subword tokens (api/chunking.py counts with the model's own
    # tokenizer). bge-small hard-truncates at 512 tokens; the chunker reserves headroom
    # for the contextual-embedding prefix and never exceeds this budget, so the embedded
    # string fits the model. 448 leaves margin under 512 for special tokens and the
    # prefix. Changing this alters chunk contents — re-upload documents for consistency.
    chunk_size_tokens: PositiveInt = 448
    chunk_overlap_tokens: int = Field(default=100, ge=0)
    # Markdown heading sections shorter than this are folded into a neighboring
    # section before chunking, so a one-line subsection doesn't become its own
    # low-information, hard-to-retrieve chunk.
    min_section_words: int = Field(default=40, ge=0)

    # CSV rows per section: a CSV upload is grouped into sections of this many data
    # rows (each row rendered as "col: value; ..."), so citations point at a labeled
    # row range instead of one giant section.
    csv_rows_per_section: int = Field(default=50, ge=1)

    # Chunks are embedded and upserted in batches of this size rather than all at
    # once, so a single huge document reports incremental job progress and never
    # holds one giant embedding call in memory.
    ingest_batch_size: PositiveInt = 64

    # Finished background ingest jobs beyond this count are pruned oldest-first, so
    # a long-running process doesn't accumulate unbounded job history in memory —
    # same tradeoff/pattern as trace_max_retained below.
    ingest_jobs_max_retained: PositiveInt = 50

    # "memory" (default) matches every prior release: jobs are lost on restart, and a
    # client polling a job whose process just restarted gets a 404 even if the ingest
    # actually succeeded in Qdrant. "sqlite" persists job status to job_store_path
    # (stdlib sqlite3, no new dependency) so that poll survives a restart. Traces
    # remain in-memory-only either way — see api/tracing.py's TraceStore docstring.
    job_store_backend: Literal["memory", "sqlite"] = "memory"
    # Only read when job_store_backend="sqlite". The containing directory must exist
    # and be writable by the process; docker-compose.yml doesn't mount one by default,
    # so an operator opting into this backend needs to add a volume for it.
    job_store_path: str = "./data/jobs.db"

    # None (default) matches every prior release: an uploaded document's raw bytes are
    # discarded once ingestion is queued — no "download original", and a future
    # chunking/embedding change needs every document manually re-uploaded. Setting
    # this to a directory keeps a copy of each upload (keyed by its normalized
    # basename filename) and exposes it via GET /documents/{filename}/original. Same
    # writable-directory/volume requirement as job_store_path. Re-indexing from these
    # originals isn't implemented — this only stores them.
    raw_document_dir: str | None = None

    # False (default): no answer-feedback capture, matching every prior release. When
    # true, POST /feedback (thumbs up/down, from ChatMessage.tsx) is enabled and
    # PublicConfigResponse.feedback_enabled tells the frontend to render the buttons.
    # Stores to feedback_store_path — its own sqlite3 file (stdlib, no new
    # dependency), independent of job_store_backend: this feature works regardless of
    # whether the operator has also opted the job store into sqlite persistence.
    feedback_enabled: bool = False
    feedback_store_path: str = "./data/feedback.db"

    # After rerank, each selected chunk's context is expanded with up to this many
    # neighboring chunks (by chunk_ordinal) on each side from the same document —
    # small-to-big retrieval: rerank on tight chunks, generate on richer context.
    # 0 disables expansion entirely.
    context_neighbor_radius: int = Field(default=1, ge=0)

    # Post-rerank diversity/dedup: after scoring, drop a survivor that is a near-
    # duplicate of a higher-ranked kept chunk (same-document neighbor within
    # context_neighbor_radius ordinals, whose expanded contexts would nearly coincide,
    # OR word-shingle Jaccard >= context_diversity_max_similarity). Freed slots refill
    # from lower-ranked survivors, so context still gets up to rerank_top_k *distinct*
    # chunks. Pure selection after scoring — rerank scores/order are untouched.
    context_diversity_enabled: bool = True
    context_diversity_max_similarity: float = Field(default=0.6, gt=0.0, le=1.0)

    # Inert in the primary paths (dev uses the Vite proxy, prod serves the frontend
    # same-origin via StaticFiles) — a fallback for a contributor who points a
    # separately-running frontend directly at this API instead of using the proxy.
    cors_allow_origins: list[str] = Field(default_factory=list)

    # Off by default: the embedding/reranker models are lazy-loaded on first use, so a
    # cold-start warmup at boot moves that (potentially tens-of-seconds) cost earlier,
    # surfacing a misconfigured MODEL name at startup instead of on a user's first
    # question. Left opt-in rather than on-by-default because the warmup path reads
    # this setting directly (not through FastAPI's DI), which sidesteps test fixtures'
    # dependency overrides — an operator-set `true` in a real `.env` would otherwise
    # also trigger real model downloads during local test runs against that same file.
    # Note: the default reranker (jinaai/jina-reranker-v2-base-multilingual) is ~1.1 GB,
    # larger than the previous default — enabling warmup or asking the first question
    # after a config change pays that download cost once.
    warmup_models: bool = False

    # Per-query debug traces (retrieval candidates, rerank keep/drop decisions, the
    # full generation prompt, stage timings) held in an in-memory ring buffer — same
    # persistence tradeoff as IngestJobStore: lost on restart, fine for debugging a
    # running instance. trace_max_retained caps memory use for large prompts/candidate
    # lists; oldest traces are pruned first once the cap is exceeded.
    trace_enabled: bool = True
    trace_max_retained: PositiveInt = 100

    # Conversation memory: the client sends recent prior turns with each question
    # (the server stays stateless — no session store). When history is present, one
    # extra LLM call rewrites the follow-up into a standalone retrieval query before
    # embedding/search/rerank, so "what about the second one?" doesn't get embedded
    # literally. Generation always sees the raw history either way; this only gates
    # the condense step.
    conversation_condense_enabled: bool = True
    # Server-side truncation cap (most recent messages kept) — independent of the
    # request schema's own cap, so an operator can tighten it without a schema change.
    conversation_max_history_messages: int = Field(default=12, ge=0)
    # The condense call only needs to produce one rewritten sentence — capped well
    # below llm_max_tokens so a runaway rewrite can't burn the generation token budget.
    condense_max_tokens: PositiveInt = 128

    # Multi-query expansion (default OFF): when enabled, one LLM call rewrites the
    # query into query_expansion_count alternative phrasings; each is embedded and
    # hybrid-searched, and the result lists are RRF-fused (same pinned rrf_k) before a
    # single rerank. Adds one LLM round-trip per question, so it stays opt-in.
    query_expansion_enabled: bool = False
    query_expansion_count: int = Field(default=3, ge=1, le=8)
    query_expansion_max_tokens: PositiveInt = 192

    # Zero-citation retry (default ON): if a non-insufficient answer comes back with no
    # [n] citation markers though sources were available, retry once with a stricter
    # reminder to cite. Never loops; falls back to the original answer if the retry
    # still lacks citations. See api/generation.py.
    citation_retry_enabled: bool = True

    # Optional single shared API key. Empty (the default) keeps every route fully open,
    # preserving the zero-config self-host story. When set, mutating/query routes
    # require a matching X-API-Key header (see api/dependencies.py's require_api_key);
    # /health and /health/ready stay unauthenticated so container healthchecks and
    # probes keep working either way. Not multi-user auth — one key for the whole API.
    api_key: str = ""

    # Verbosity of the application's own loggers (the "api" and "eval" namespaces).
    # api/logging_config.py consumes this via dictConfig at app construction; without
    # it the root logger's WARNING default silences every logger.info in the codebase
    # (request logs, ingest progress, warmup notices).
    log_level: str = "INFO"

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized not in _VALID_LOG_LEVELS:
                raise ValueError(
                    f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, got {value!r}."
                )
            return normalized
        return value

    @field_validator("openai_reasoning_effort", mode="before")
    @classmethod
    def _normalize_reasoning_effort(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized not in _VALID_REASONING_EFFORTS:
                raise ValueError(
                    "openai_reasoning_effort must be one of "
                    f"{sorted(_VALID_REASONING_EFFORTS)}, got {value!r}."
                )
            return normalized
        return value


def get_settings() -> AppSettings:
    """Return settings from the current process environment."""

    return AppSettings()
