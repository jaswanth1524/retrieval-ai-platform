"""Application settings loaded from environment variables."""

import logging
import re
from typing import Literal

from pydantic import Field, PositiveInt, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The schema's cap on documents per summary request (api/schemas.py imports it).
SUMMARY_DOCUMENTS_LIMIT = 10

_VALID_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})
# Longest input the supported embedding models read (bge, e5, MiniLM: 512 tokens).
_EMBEDDING_MAX_TOKENS = 512
# The characters Qdrant v1.19 refuses in a collection name (checked against the server).
_BAD_COLLECTION_CHARS = re.compile(r'[<>:"/\\|?*\x00]')
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
    # The gRPC port the client dials when qdrant_prefer_grpc is on (the host it uses is
    # qdrant_url's). Compose also maps it on the host, and pins it to 6334 for the API
    # container, which reaches Qdrant inside the compose network.
    qdrant_grpc_port: int = Field(default=6334, ge=1, le=65535)
    # Qdrant's own API key (the server's QDRANT__SERVICE__API_KEY). Empty (default)
    # sends none. Without it anyone who can reach Qdrant's port reads or drops the
    # collection directly, around API_KEY entirely.
    qdrant_api_key: str = ""

    dense_embedding_model: str = "BAAI/bge-small-en-v1.5"
    sparse_embedding_model: str = "Qdrant/BM25"
    # Prepended to the dense query text only (never the corpus side) before embedding.
    # bge-small's model card recommends this instruction for short-query→passage
    # retrieval, and fastembed does not apply it for us. Empty string disables it.
    # Query-side only, so changing it needs no re-ingest.
    dense_query_instruction: str = "Represent this sentence for searching relevant passages: "
    # Prepended to every *corpus* text before dense embedding (never to queries, never
    # to BM25). Empty for bge; "passage: " for the e5 family, which is trained on that
    # pair with DENSE_QUERY_INSTRUCTION="query: ". Recorded with the collection: changing
    # it is a new vector space, like changing the model.
    dense_passage_prefix: str = ""
    # BM25's stemmer language (Qdrant/BM25 only), e.g. "german", "french", "spanish".
    # Recorded with the collection like the models: a change needs a re-index.
    sparse_language: str = "english"
    reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual"
    embedding_model_tag: str = "fastembed:BAAI/bge-small-en-v1.5"

    # Dense vectors kept as int8 in RAM beside the float32 originals ("scalar"), about a
    # quarter of the memory for a large corpus; the top candidates are re-scored against
    # the originals (oversampled by QDRANT_QUANTIZATION_OVERSAMPLING), so ranking stays
    # close. "none" (default): float32 only. Applied to an existing collection in place
    # (no re-embedding); switching back removes it.
    qdrant_quantization: Literal["none", "scalar"] = "none"
    qdrant_quantization_oversampling: float = Field(default=2.0, ge=1.0, le=8.0)
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
    # Which ONNX export of RERANKER_MODEL to load. "auto" (default): jina-v2's int8
    # export, fastembed's registered file for any other model. "" : always fastembed's
    # registered file (jina: fp32 onnx/model.onnx). Anything else: that path in the repo.
    # int8 jina measured 2.5x faster than fp32 with 40% less memory, 81-83% of the same
    # top-6 context and no question losing its context (README "Latency").
    reranker_onnx_file: str = "auto"
    # ONNX Runtime intra-op threads per session; 0 keeps ORT's default (all cores).
    # Lowering these makes a lone question SLOWER — their value is stopping ingest
    # embedding and query-time rerank from oversubscribing the same cores when they
    # overlap. Leave at 0 unless you run ingest and questions concurrently.
    embedding_threads: int = Field(default=0, ge=0)
    reranker_threads: int = Field(default=0, ge=0)
    # Rerank scores memoized per (query, passage text). Regenerate/Retry re-ask the
    # identical query, so a hit skips the most expensive stage entirely. 0 disables.
    rerank_cache_size: int = Field(default=4096, ge=0)
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
    # When no candidate clears rerank_min_score, this many of the best-ranked ones go
    # to the model anyway, marked as weak matches, and it decides whether they answer
    # the question (the answer then carries weak_matches). 0: refuse at once, as
    # before. Each cross-encoder scores on its own scale, so a fixed floor dropped the
    # passage that answered the question: "what are foundational models" scored 0.006
    # with MiniLM and 0.154 with jina against "Foundation models: You have trained ...".
    rerank_fallback_count: int = Field(default=3, ge=0, le=20)
    # Opt-in (0 = off): of the passages that cleared the floor, keep only those scoring
    # within this share of the way from the median candidate's score to the best one
    # ((score - median) / (top - median) >= this), and always the best two. Works on
    # any reranker's scale, and only ever trims the prompt — never turns an answer into
    # a refusal. A low-scoring sixth passage is ~1k tokens of prompt to read.
    rerank_relative_floor: float = Field(default=0.0, ge=0.0, le=1.0)

    # Checked at startup (case-insensitive): a typo used to boot fine and then fail
    # every question with a 400 that blamed the client.
    llm_provider: Literal["ollama", "openai", "openai_compatible", "anthropic"] = "ollama"
    # Providers a question may pick per request (`llm_provider` on /questions*), as a
    # JSON list, e.g. ["ollama"]. Unset: any configured provider. LLM_PROVIDER itself
    # is always allowed. Restricting it stops a caller — anyone with the read key, or
    # anyone at all without API_KEY — from switching to a paid provider at will.
    allowed_request_providers: (
        list[Literal["ollama", "openai", "openai_compatible", "anthropic"]] | None
    ) = None
    llm_model: str = "llama3.1:8b"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: PositiveInt = 512
    llm_request_timeout_seconds: float = Field(default=60.0, gt=0.0)
    # Local providers (Ollama, OpenAI-compatible) send nothing while they read the
    # prompt, and LiteLLM hands its timeout to httpx as the longest silence allowed
    # (it turns an httpx.Timeout back into one number for Ollama), so a 6k-token prompt
    # on an M1 timed out before the first token. Each call's timeout grows by its
    # estimated prompt tokens divided by this rate (or by the slower rate this server
    # has been measured at). 0 turns the scaling off; hosted providers never scale.
    llm_prefill_tokens_per_second: float = Field(default=40.0, ge=0.0)
    llm_num_retries: int = Field(default=0, ge=0)
    ollama_base_url: str = "http://localhost:11434"
    # Ollama unloads a model from memory 5 minutes after its last use by default,
    # forcing a multi-second reload on the next question. Keeping it resident trades
    # idle RAM for eliminating that reload on every request after the first.
    ollama_keep_alive: str = "30m"
    # Context window requested from Ollama; 0 leaves it to Ollama. Not a speed knob —
    # measured on an M1, 8192 vs Ollama's 32768 left prefill/decode speed unchanged
    # but cut the loaded model from 8.2 GB to 5.3 GB. Too small silently truncates the
    # front of the prompt (the grounding rules), so leave headroom for history.
    ollama_num_ctx: int = Field(default=0, ge=0)
    # How many LLM calls this process sends Ollama at once. Ollama answers one request
    # at a time unless OLLAMA_NUM_PARALLEL says otherwise and queues the rest — where
    # each waiting request's timeout runs, so a second question timed out without ever
    # being read. Calls past this wait here instead (the stream says so), and their
    # timeout starts when they are sent. 0 = no limit (match a raised OLLAMA_NUM_PARALLEL).
    ollama_max_parallel: int = Field(default=1, ge=0)
    # The model's context window in tokens, for fitting the prompt into it. 0 (default)
    # uses OLLAMA_NUM_CTX when the provider is Ollama and that is set; for OpenAI and
    # Anthropic models, the window LiteLLM's bundled model table gives; else no fitting.
    # When known, the prompt drops the lowest-ranked context and then the oldest
    # history until it fits with LLM_MAX_TOKENS to spare — an over-long prompt had its
    # *front* (the grounding rules) silently cut by the model server.
    llm_context_window: int = Field(default=0, ge=0)
    # Opt-in (0 = off): the most estimated tokens a question's prompt may take, whatever
    # the window. Over it, neighbour text goes first (lowest-ranked passage first), then
    # whole lowest-ranked passages, then the oldest history. A local model reads a
    # prompt at ~70 tokens/s on an M1, so prompt size is most of the wait.
    prompt_token_budget: int = Field(default=0, ge=0)
    # Questions answered at once; past it a question gets 429 + Retry-After at once
    # instead of holding a request thread for a minute (api/admission.py). 0 = no cap.
    max_concurrent_questions: int = Field(default=4, ge=0)
    # Of those, how many may be summaries (POST /questions/summary*), which run for
    # minutes: without it four summaries held every slot and each question got 429.
    # 0 = no separate cap.
    max_concurrent_summaries: int = Field(default=2, ge=0)
    # The language answers are written in. Empty: the model's choice (usually the
    # question's, often English with English instructions); "match": the question's
    # language; or a language name ("German"). One fixed sentence in the system prompt,
    # so prompt prefix caching still applies.
    answer_language: str = Field(default="", max_length=40)
    # Requests per minute each client may make that change something (POST, PATCH,
    # DELETE on the API: questions, searches, uploads, feedback...) or that present a
    # wrong API key; past it, 429 + Retry-After (api/rate_limit.py). Reads and job polls
    # are never limited. 0 = off.
    rate_limit_per_minute: int = Field(default=0, ge=0)
    # Finished answers kept for repeat questions (api/answer_cache.py); 0 turns the
    # cache off. Invalidated by every ingest, delete and tag change, and expired after
    # the TTL. Questions with conversation history are never cached.
    answer_cache_size: int = Field(default=256, ge=0)
    answer_cache_ttl_seconds: float = Field(default=3600.0, gt=0.0)
    openai_api_key: str | None = Field(default=None)
    # Any server speaking the OpenAI chat-completions API — vLLM, llama.cpp's server,
    # LM Studio, LocalAI — selected with LLM_PROVIDER=openai_compatible (or per request).
    # Self-hostable like Ollama; the base URL is the one ending in /v1. The key is
    # optional: most local servers accept anything.
    openai_compatible_base_url: str = ""
    openai_compatible_model: str = ""
    openai_compatible_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    # Optional Anthropic (Claude) generation, selected with LLM_PROVIDER=anthropic or
    # per question. Like OpenAI it is a hosted add-on: upload, indexing and passage
    # search never call it, and embeddings stay local.
    anthropic_api_key: str | None = Field(default=None)
    anthropic_model: str = "claude-sonnet-5-5"
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
    # POST /import: the largest backup zip (GET /export) accepted. Each original inside
    # is still held to max_upload_bytes.
    max_import_bytes: PositiveInt = 256 * 1024 * 1024
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

    # PDF work limits. A PDF over max_pdf_pages is refused (400) before any page is
    # read; only the first max_ocr_pages text-less pages are OCR'd (200 dpi each, the
    # slowest step by far), the rest skipped with a warning. Without them a small file
    # of thousands of blank pages held both ingest workers for hours.
    max_pdf_pages: int = Field(default=2000, ge=1)
    max_ocr_pages: int = Field(default=100, ge=0)
    # An XLSX workbook with more cells than this (counted across sheets, empty ones
    # included as openpyxl reads them) is refused. Every cell is held as text while the
    # sheet is rendered; this bounds a small file that declares a huge sheet.
    max_spreadsheet_cells: int = Field(default=200_000, ge=1)
    # Each upload is parsed in a child process that is stopped after this many seconds
    # (api/parse_isolation.py): a pathological file used to hold an ingest worker
    # indefinitely, competing with questions for the GIL. 0 parses in the API process
    # with no limit.
    parse_timeout_seconds: float = Field(default=300.0, ge=0)

    # Chunks are embedded and upserted in batches of this size rather than all at
    # once, so a single huge document reports incremental job progress and never
    # holds one giant embedding call in memory.
    ingest_batch_size: PositiveInt = 64
    # Points written per Qdrant upsert, independent of the embedding batch above. Each
    # upsert is a round trip with wait=True under the filename lock, so writing a
    # 1,000-chunk document in 64-point calls held the lock through 16 of them.
    ingest_upsert_batch_size: PositiveInt = 256
    # Upload bytes allowed to sit queued or in progress at once. Every accepted upload
    # holds its full content in memory until its ingest job finishes, and the executor
    # queue itself is unbounded — past this budget uploads get 503 + Retry-After
    # instead of growing memory without limit. An empty backlog always admits one.
    ingest_max_pending_bytes: PositiveInt = 512 * 1024 * 1024

    # Finished background ingest jobs beyond this count are pruned oldest-first, so
    # a long-running process doesn't accumulate unbounded job history in memory —
    # same tradeoff/pattern as trace_max_retained below. It also caps the jobs queued
    # or running at once (IngestBacklog): past it an upload gets 503 + Retry-After
    # even with byte budget left, so the store never evicts a job still in flight.
    ingest_jobs_max_retained: PositiveInt = 50
    # The document listing (GET /documents) is read by scrolling every point, and kept
    # between changes made by this process, which it updates in place. This only bounds
    # how long a change made by *another* process (a second API worker, the eval
    # harness's ingest) can go unseen. It was 30 s, which made the panel rescan a large
    # collection twice a minute for writes a single-process deployment never has.
    document_listing_ttl_seconds: float = Field(default=300.0, gt=0)

    # "memory" (default) matches every prior release: jobs are lost on restart, and a
    # client polling a job whose process just restarted gets a 404 even if the ingest
    # actually succeeded in Qdrant. "sqlite" persists job status to job_store_path
    # (stdlib sqlite3, no new dependency) so that poll survives a restart. Traces
    # remain in-memory-only either way — see api/tracing.py's TraceStore docstring.
    job_store_backend: Literal["memory", "sqlite"] = "memory"
    # Only read when job_store_backend="sqlite". The containing directory (created if
    # missing) must be writable (in Docker, under /app/data: the docrag_data volume);
    # startup fails naming the setting otherwise (api/storage.py).
    job_store_path: str = "./data/jobs.db"

    # None (default) matches every prior release: an uploaded document's raw bytes are
    # discarded once ingestion is queued — no "download original", and a future
    # chunking/embedding change needs every document manually re-uploaded. Setting
    # this to a directory keeps a copy of each upload (keyed by its normalized
    # basename filename) and exposes it via GET /documents/{filename}/original. Same
    # writable-directory/volume requirement as job_store_path. It also enables
    # POST /documents/{filename}/reindex (and the bulk POST /documents/reindex), which
    # re-chunk and re-embed a document from its stored original.
    raw_document_dir: str | None = None

    # False (default): no answer-feedback capture, matching every prior release. When
    # true, POST /feedback (thumbs up/down, from ChatMessage.tsx) is enabled and
    # PublicConfigResponse.feedback_enabled tells the frontend to render the buttons.
    # Stores to feedback_store_path — its own sqlite3 file (stdlib, no new
    # dependency), independent of job_store_backend: this feature works regardless of
    # whether the operator has also opted the job store into sqlite persistence.
    feedback_enabled: bool = False
    feedback_store_path: str = "./data/feedback.db"
    # With RAW_DOCUMENT_DIR: how many earlier versions of each original a re-upload keeps
    # (GET /documents/{f}/versions, POST .../versions/{v}/restore). 0 = none (default).
    raw_document_versions: int = Field(default=0, ge=0, le=50)
    # Oldest ratings are dropped past this many rows (~6 KB each at most).
    feedback_max_rows: PositiveInt = 50_000
    # A persistent log of every question and summary asked (text, workspace, how it went,
    # best passage score), for "what do people ask that the documents don't cover?"
    # (GET /query-log, full key). Off by default: it keeps what users typed. Its own
    # sqlite file; never in a backup. Oldest rows go past QUERY_LOG_MAX_ROWS.
    query_log_enabled: bool = False
    query_log_path: str = "./data/query_log.db"
    query_log_max_rows: PositiveInt = 20_000

    # After rerank, each selected chunk's context is expanded with up to this many
    # neighboring chunks (by chunk_ordinal) on each side from the same document —
    # small-to-big retrieval: rerank on tight chunks, generate on richer context.
    # 0 disables expansion entirely.
    context_neighbor_radius: int = Field(default=1, ge=0)
    # Caps the neighbour text added on each side of a chunk, in characters: the text
    # nearest the chunk is kept, cut at a word boundary. 0 = no cap (the default).
    # Neighbours are most of the prompt, and prefill is most of the wait before the first
    # token. 300 was measured only at CHUNK_SIZE_TOKENS=96 (about 1s less prefill on
    # qwen2.5, same answers on 11 questions); at the default chunk size it would drop
    # most of each neighbour, which hasn't been measured (README "Latency").
    context_neighbor_max_chars: int = Field(default=0, ge=0)

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
    # Note: the default reranker (jina-reranker-v2, int8 export) is ~0.28 GB (~1.1 GB
    # for fp32) — enabling warmup or asking the first question after a config change
    # pays that download cost once.
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
    # Opt-in (0 = off): earlier answers sent to the answering model are cut to this many
    # characters each (the condense step keeps its own 1500). The question already
    # carries its resolved meaning ("Read with the conversation as: ...").
    generation_history_assistant_chars: int = Field(default=0, ge=0)
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
    # How an uncited answer gets its citations. "llm": the retry above (a second full
    # generation — about a minute on a local model). "backfill": each answer sentence
    # a source plainly states (most of its content words, in one or two adjacent
    # sentences of the source) gets that source's [n]; no LLM call. "auto" (default):
    # backfill first for local providers (Ollama, OpenAI-compatible), the LLM retry only
    # when nothing matched; the LLM retry for hosted providers. "off": neither (as
    # CITATION_RETRY_ENABLED=false).
    citation_retry_mode: Literal["auto", "llm", "backfill", "off"] = "auto"

    # Suggested follow-up questions (default OFF: one more LLM call per answer, which on
    # a local model is seconds). When on, an answer that cited its sources gets up to
    # FOLLOWUP_COUNT short questions those sources can answer, sent after the answer
    # (a `followups` frame after `done` on the stream; `followups` on /questions).
    followups_enabled: bool = False
    followup_count: int = Field(default=3, ge=1, le=5)
    followup_max_tokens: PositiveInt = 128

    # Whole-document summaries and comparisons (POST /questions/summary): the document's
    # passages are read in order, in parts of at most SUMMARY_BATCH_CHARS characters
    # (lowered to fit LLM_CONTEXT_WINDOW / OLLAMA_NUM_CTX when known), each part is turned
    # into cited notes, and the notes are combined into one answer. At most
    # SUMMARY_MAX_BATCHES parts per request: a longer document is read at evenly spaced
    # passages, and the answer reports how many it read. Each part is one LLM call, so
    # this bounds the time a summary takes. SUMMARY_MAX_TOKENS is the final answer's
    # budget (summaries run longer than answers); notes keep LLM_MAX_TOKENS.
    # POST /documents/from-url (off by default): the server downloads a document from a
    # URL and indexes it like an upload (MAX_UPLOAD_BYTES applies). Only public
    # addresses are fetched unless URL_INGEST_ALLOW_PRIVATE is set — the request comes
    # from inside the deployment, so a private one could reach Qdrant or cloud metadata.
    # Separate corpora on one server (api/workspaces.py), off by default: each is its own
    # Qdrant collection (QDRANT_COLLECTION__ws_<name>) and originals directory, chosen
    # per request with the X-Workspace header and created with POST /workspaces. Not
    # access control — any key that can use the API can use every workspace.
    workspaces_enabled: bool = False

    # Keep a workspace's index in step with a folder (api/watch.py): every
    # WATCH_INTERVAL_SECONDS, new and changed supported files are indexed like uploads.
    # Empty (default) = off. WATCH_DELETE_MISSING also deletes the document of a file
    # that disappears — only ones the watcher saw in the folder itself. WATCH_WORKSPACE
    # names the workspace it indexes into (needs WORKSPACES_ENABLED; created at startup
    # if it doesn't exist yet); empty = the default workspace.
    watch_dir: str = ""
    watch_interval_seconds: float = Field(default=30.0, ge=2.0, le=86_400.0)
    watch_delete_missing: bool = False
    watch_workspace: str = ""
    # A watched file whose read fails, or whose indexing job fails, is tried again after
    # WATCH_INTERVAL_SECONDS x 2^n (at most WATCH_RETRY_MAX_SECONDS), up to
    # WATCH_MAX_RETRIES times; then it waits until the file changes. 0 = never retried.
    watch_max_retries: int = Field(default=3, ge=0, le=100)
    watch_retry_max_seconds: float = Field(default=3600.0, ge=2.0, le=604_800.0)

    url_ingest_enabled: bool = False
    url_ingest_allow_private: bool = False
    url_ingest_timeout_seconds: float = Field(default=30.0, gt=0.0, le=600.0)

    summary_batch_chars: int = Field(default=9000, ge=1000, le=200_000)
    summary_max_batches: int = Field(default=8, ge=1, le=64)
    summary_max_tokens: PositiveInt = 1024
    # Parts of a long document turned into notes at once, with a hosted provider (OpenAI,
    # Anthropic, an OpenAI-compatible server). Ollama always takes them one at a time:
    # it runs one generation at a time anyway, and parallel requests only queue there.
    summary_parallel_parts: int = Field(default=4, ge=1, le=16)
    # Most documents one brief (POST /questions/summary, "mode": "brief") may cover.
    summary_max_documents: int = Field(default=5, ge=2, le=SUMMARY_DOCUMENTS_LIMIT)

    # Optional single shared API key. Empty (the default) keeps every route fully open,
    # preserving the zero-config self-host story. When set, every route except
    # /health, /health/ready and /config — reads included — requires a matching
    # X-API-Key header (api/dependencies.py's require_api_key, and
    # api/request_guard.py, which checks it before a request body is read). Not multi-user auth —
    # one key for the whole API.
    api_key: str = ""
    # Optional second, read-only key (needs API_KEY set too): it can ask questions and
    # read documents, traces and jobs, but not upload, delete, re-index, tag, export,
    # or read feedback/metrics — those need API_KEY (403 for the read key). For sharing
    # the UI with people who should use the corpus but not change it.
    api_read_key: str = ""

    # Verbosity of the application's own loggers (the "api" and "eval" namespaces).
    # api/logging_config.py consumes this via dictConfig at app construction; without
    # it the root logger's WARNING default silences every logger.info in the codebase
    # (request logs, ingest progress, warmup notices).
    log_level: str = "INFO"
    # "json": one JSON object per log line (for a log shipper); "text" otherwise.
    log_format: Literal["text", "json"] = "text"

    @model_validator(mode="after")
    def _overlap_below_chunk_size(self) -> "AppSettings":
        # Refused at startup: otherwise every upload was accepted (202) and every job
        # then failed in the background with the same message.
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS.")
        return self

    @model_validator(mode="after")
    def _brief_fits_the_part_cap(self) -> "AppSettings":
        # Every document in a summary gets at least one part, so a brief over more
        # documents than SUMMARY_MAX_BATCHES would make more LLM calls than that cap
        # allows. Lowered rather than refused: compose loads .env.example, so
        # SUMMARY_MAX_DOCUMENTS always counts as set there, and a refusal would stop
        # every install that lowered SUMMARY_MAX_BATCHES. Never below 2 (a comparison
        # reads two documents, one part each, whatever the cap).
        limit = max(2, self.summary_max_batches)
        if self.summary_max_documents > limit:
            if "summary_max_documents" in self.model_fields_set:
                logging.getLogger(__name__).warning(
                    "SUMMARY_MAX_DOCUMENTS=%d is above SUMMARY_MAX_BATCHES=%d; briefs "
                    "cover at most %d documents.",
                    self.summary_max_documents,
                    self.summary_max_batches,
                    limit,
                )
            self.summary_max_documents = limit
        return self

    @model_validator(mode="after")
    def _warn_on_an_untuned_rerank_floor(self) -> "AppSettings":
        # RERANK_MIN_SCORE is a sigmoid of the cross-encoder's logit, and each model's
        # logits sit on their own scale: 0.15 was measured for the default reranker.
        # Another one at the same floor dropped passages that answer the question
        # ("not enough information"), so say so at startup. Compared by value, not by
        # whether it was set: compose loads .env.example, which sets it.
        fields = type(self).model_fields
        if (
            self.reranker_model != fields["reranker_model"].default
            and self.rerank_min_score == fields["rerank_min_score"].default
        ):
            logging.getLogger(__name__).warning(
                "RERANKER_MODEL=%s with RERANK_MIN_SCORE=%s: that floor was measured for "
                "%s. Measure it for this model (eval/harness.py) or answers may say "
                "the documents don't cover what they do.",
                self.reranker_model,
                self.rerank_min_score,
                fields["reranker_model"].default,
            )
        return self

    @model_validator(mode="after")
    def _warn_on_settings_that_do_nothing(self) -> "AppSettings":
        log = logging.getLogger(__name__)
        if self.chunk_size_tokens > _EMBEDDING_MAX_TOKENS:
            # The embedder cuts every longer input to its limit without a word, so the
            # end of each chunk would be indexed for BM25 but never embedded.
            log.warning(
                "CHUNK_SIZE_TOKENS=%d is past the %d tokens embedding models read; the "
                "end of every chunk would be left out of its vector.",
                self.chunk_size_tokens,
                _EMBEDDING_MAX_TOKENS,
            )
        if self.raw_document_versions and not self.raw_document_dir:
            log.warning(
                "RAW_DOCUMENT_VERSIONS=%d has no effect without RAW_DOCUMENT_DIR.",
                self.raw_document_versions,
            )
        return self

    @model_validator(mode="after")
    def _read_key_needs_a_full_key(self) -> "AppSettings":
        # With only a read key, the "admin" routes would have no key at all: anyone
        # could change the corpus that readers need a key to look at.
        if self.api_read_key and not self.api_key:
            raise ValueError("API_READ_KEY needs API_KEY set as well.")
        if self.api_read_key and self.api_read_key == self.api_key:
            raise ValueError("API_READ_KEY must differ from API_KEY.")
        return self

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalize_llm_provider(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("answer_language", mode="after")
    @classmethod
    def _check_answer_language(cls, value: str) -> str:
        value = " ".join(value.split())
        if value.lower() == "match":
            return "match"
        if value and not re.fullmatch(r"[^\W\d_]+(?: [^\W\d_]+)*", value):
            raise ValueError(
                'ANSWER_LANGUAGE must be empty, "match", or a language name ("German").'
            )
        return value

    @field_validator("sparse_language", mode="before")
    @classmethod
    def _normalize_sparse_language(cls, value: object) -> object:
        # One spelling per language: it is recorded with the collection and compared
        # on every check, so "English", " english" or "" (fastembed's default, which is
        # English) must not read as a different index needing a re-ingest.
        if isinstance(value, str):
            return value.strip().lower() or "english"
        return value

    @field_validator("qdrant_collection")
    @classmethod
    def _collection_without_workspace_separator(cls, value: str) -> str:
        # A workspace's collection is "<QDRANT_COLLECTION>__ws_<name>"
        # (api/workspaces.py), and the separator is how a collection is told to be a
        # workspace's: a base name containing it would read as a workspace itself.
        if "__ws_" in value:
            raise ValueError("QDRANT_COLLECTION must not contain '__ws_' (used for workspaces).")
        # Qdrant's own rules (only what the server refuses), checked at startup rather
        # than as an error on the first request.
        if not value or len(value) > 255 or _BAD_COLLECTION_CHARS.search(value):
            raise ValueError(
                'QDRANT_COLLECTION must be 1-255 characters, without any of < > : " / \\ | ? *.'
            )
        return value

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
