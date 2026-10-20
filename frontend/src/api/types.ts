export interface HealthResponse {
  status: string;
  /** The server's DocRAG version; absent on older servers. */
  version?: string;
}

/** GET /health/ready (open; 503 with the same body when degraded). The index check is
 *  of the default workspace's collection. */
export interface ReadinessResponse {
  status: 'ok' | 'degraded';
  qdrant: boolean;
  generation_provider: boolean;
  llm_provider: string;
  /** False when the index was built for another embedding setup (questions get 409). */
  index_compatible?: boolean;
  index_detail?: string | null;
  /** Uploaded originals are kept (RAW_DOCUMENT_DIR): a backup can rebuild the index. */
  originals_kept?: boolean;
}

/** Tokens the provider reported for an answer's LLM calls. */
export interface UsageResponse {
  prompt_tokens: number;
  completion_tokens: number;
}

// 'unauthorized' is distinct from 'error' on purpose: the server is reachable and
// answering, it just rejected us for want of an API key. Collapsing the two would put
// the app behind an "unreachable" wall that hides the very settings needed to fix it.
export type ApiStatus = 'checking' | 'ok' | 'error' | 'unauthorized';

export interface PublicConfigResponse {
  qdrant_collection: string;
  dense_embedding_model: string;
  sparse_embedding_model: string;
  reranker_model: string;
  embedding_model_tag: string;
  llm_provider: string;
  llm_model: string;
  llm_temperature: number;
  openai_model: string;
  openai_available: boolean;
  ollama_available: boolean;
  rrf_k: number;
  dense_retrieval_limit: number;
  sparse_retrieval_limit: number;
  fused_top_n: number;
  rerank_top_k: number;
  max_context_chunks: number;
  rerank_min_score: number;
  max_upload_bytes: number;
  rerank_top_k_limit: number;
  max_context_chunks_limit: number;
  llm_temperature_max: number;
  feedback_enabled: boolean;
  /** RAW_DOCUMENT_DIR is set: originals are kept, so documents can be re-indexed. */
  raw_documents_enabled?: boolean;
  /** An OpenAI-compatible server (vLLM, llama.cpp, LM Studio) is configured. */
  openai_compatible_available?: boolean;
  openai_compatible_model?: string;
  /** WORKSPACES_ENABLED: separate corpora chosen per request; absent on older servers. */
  workspaces_enabled?: boolean;
  /** URL_INGEST_ENABLED: documents can be added from a URL; absent on older servers. */
  url_ingest_enabled?: boolean;
  /** ANTHROPIC_API_KEY is set; absent on older servers. */
  anthropic_available?: boolean;
  anthropic_model?: string;
  /** Providers a question may pick per request (ALLOWED_REQUEST_PROVIDERS). */
  allowed_request_providers?: string[];
  // Upload suffixes the server accepts (".pdf", ".epub", ...); absent on older servers.
  supported_extensions?: string[];
  /** WATCH_DIR is set: GET /watch and POST /watch/scan exist. */
  watch_enabled?: boolean;
  /** ANSWER_LANGUAGE: "" = the model's choice, "match" = the question's language. */
  answer_language?: string;
  /** SUMMARY_MAX_DOCUMENTS: the most documents one brief covers; absent on older servers. */
  summary_max_documents?: number;
}

/** A watched file waiting to be tried again after a failure. */
export interface WatchRetry {
  filename: string;
  attempts: number;
  next_attempt_at: number;
  error: string;
}

/** GET /watch: the watched folder as of the watcher's last pass. */
export interface WatchStatusResponse {
  folder: string;
  last_scan_at: number | null;
  files: number;
  problems: string[];
  deferred: string[];
  retrying: WatchRetry[];
  gave_up: string[];
}

/** What the caller's API key unlocks (GET /access). */
export type AccessLevel = 'full' | 'read' | 'open';

export interface AccessResponse {
  access: AccessLevel;
}

export interface ImportResponse {
  jobs: DocumentJobAcceptedResponse[];
  deferred: string[];
  skipped: string[];
  missing_originals: string[];
  rejected: string[];
  feedback_imported: number;
  feedback_skipped: number;
}

/** POST /search: the passages a question would be answered from, with no LLM involved. */
export interface SearchRequest {
  query: string;
  filenames?: string[];
  tags?: string[];
  rerank_top_k?: number;
}

export interface SearchResultResponse {
  rank: number;
  filename: string;
  page: number;
  /** The last page when the passage crosses a page break (it starts on `page`). */
  page_end?: number | null;
  section: string;
  chunk_id: string;
  chunk_ordinal: number | null;
  text: string;
  /** RRF-fused hybrid score. */
  retrieval_score: number;
  /** Cross-encoder score, normalized to [0, 1]. */
  rerank_score: number;
}

export interface SearchResponse {
  results: SearchResultResponse[];
  timings: { embed_ms: number; search_ms: number; rerank_ms: number };
  /** Nothing cleared the relevance bar an answer's sources must clear; these are the
   *  closest passages (typical of a one- or two-word search). Absent from older servers. */
  weak_matches?: boolean;
}

export interface FeedbackItemResponse {
  id: string;
  trace_id: string | null;
  question: string;
  answer_excerpt: string;
  cited_filenames: string[];
  rating: FeedbackRating;
  citation_source_number: number | null;
  created_at: number;
  /** Absent from rows recorded before the kind was (read as "question"). */
  kind?: FeedbackKind;
}

export interface FeedbackListResponse {
  feedback: FeedbackItemResponse[];
}

export interface QuestionOverrides {
  rerankTopK: number | null;
  maxContextChunks: number | null;
  llmTemperature: number | null;
}

export interface DocumentIngestResponse {
  filename: string;
  sections_parsed: number;
  chunks_ingested: number;
  collection_name: string;
  /** Already indexed from identical bytes: left as it was. */
  unchanged?: boolean;
}

export type IngestJobState = 'queued' | 'parsing' | 'embedding' | 'done' | 'failed';

export interface DocumentJobAcceptedResponse {
  job_id: string;
  filename: string;
  state: 'queued';
}

export interface DocumentReindexAllResponse {
  jobs: DocumentJobAcceptedResponse[];
  /** Stale and re-indexable, but the ingest backlog was full — retry later. */
  deferred: string[];
}

export interface DocumentJobStatusResponse {
  job_id: string;
  filename: string;
  state: IngestJobState;
  chunks_total: number;
  chunks_done: number;
  error: string | null;
  result: DocumentIngestResponse | null;
  /** Stopped by DELETE /documents/jobs/{id} (state "failed"). */
  cancelled?: boolean;
  /** Unix time the job was accepted. */
  created_at?: number | null;
  /** Found the document already indexed from identical bytes; nothing re-embedded. */
  unchanged?: boolean;
}

/** GET /documents/jobs: queued and running jobs, whoever started them. */
export interface DocumentJobListResponse {
  jobs: DocumentJobStatusResponse[];
}

/** POST /documents/{f}/refresh when the source said nothing changed. */
export interface DocumentRefreshResponse {
  filename: string;
  unchanged: boolean;
}

export interface DocumentListResponse {
  filenames: string[];
  chunk_counts?: Record<string, number>;
  page_counts?: Record<string, number>;
  // A filename is absent from these two (not present with a null value) when it was
  // ingested before byte_size/uploaded_at were stamped at ingest time.
  byte_sizes?: Record<string, number>;
  uploaded_ats?: Record<string, number>;
  /** Indexed by an older chunker; re-indexing applies the current one. */
  stale_filenames?: string[];
  /** Have a stored original, so POST /documents/{f}/reindex works. */
  reindexable_filenames?: string[];
  /** Tagged documents only; an untagged one is absent. */
  tags?: Record<string, string[]>;
  /** Documents fetched by URL: where from (POST /documents/{f}/refresh fetches again). */
  source_urls?: Record<string, string>;
}

export interface DocumentTagsResponse {
  filename: string;
  tags: string[];
}

export interface DocumentDeleteResponse {
  filename: string;
  points_deleted: number;
}

export type FeedbackRating = 'up' | 'down';

/** What was rated: a standalone question, one asked with the conversation as history,
 *  or a summary/comparison. The server defaults to "question". */
export type FeedbackKind = 'question' | 'follow_up' | 'summary';

export interface FeedbackRequest {
  trace_id: string | null;
  question: string;
  answer_excerpt: string;
  cited_filenames: string[];
  rating: FeedbackRating;
  citation_source_number: number | null;
  kind?: FeedbackKind;
}

export interface FeedbackResponse {
  id: string;
}

export interface DocumentChunkResponse {
  chunk_id: string;
  page: number;
  /** The last page when the passage crosses a page break (it starts on `page`). */
  page_end?: number | null;
  section: string;
  text: string;
  chunk_ordinal: number | null;
}

export interface DocumentContentResponse {
  filename: string;
  chunks: DocumentChunkResponse[];
  /** Every chunk the document has, when `chunks` is one page of it. */
  total_chunks?: number | null;
  /** With `around`: whether that chunk still exists (a re-index changes chunk ids). */
  target_found?: boolean | null;
}

/** The passage of a cited chunk an answer's claim rests on. */
export interface CitationSpan {
  /** Code-point offsets into the chunk's text: a hint only, since JS indexes UTF-16. */
  start: number;
  end: number;
  quote: string;
}

export interface CitationResponse {
  source_number: number;
  filename: string;
  page: number;
  /** The last page when the passage crosses a page break (it starts on `page`). */
  page_end?: number | null;
  section: string;
  chunk_id: string;
  text: string;
  /** Only on the final answer's sources; absent from older servers and saved turns. */
  spans?: CitationSpan[];
}

export interface TimingsResponse {
  embed_ms: number;
  search_ms: number;
  // Cross-encoder scoring plus the diversity filter. Neighbour expansion is timed
  // separately as context_expansion_ms rather than hidden in here.
  rerank_ms: number;
  generate_ms: number;
  total_ms: number;
  condense_ms: number;
  // Query-variant generation, which runs BEFORE embedding — not context expansion.
  query_expansion_ms?: number;
  // Neighbour/context expansion after rerank (small-to-big retrieval).
  context_expansion_ms?: number;
  // The zero-citation retry's LLM round trip, after generation; 0 when it didn't fire.
  // generate_ms excludes it.
  citation_retry_ms?: number;
  // Part of rerank_ms spent queued behind another question's rerank.
  rerank_wait_ms?: number;
  // Request start to the first streamed token; 0 on /questions and cached answers.
  first_token_ms?: number;
}

/** How well an answer is grounded. Extensible: treat an unknown value as 'answered'. */
export type AnswerStatus =
  | 'answered'
  | 'insufficient_context'
  | 'no_context'
  | 'uncited'
  | (string & {});

export interface HistoryMessage {
  role: 'user' | 'assistant';
  content: string;
}

export type LlmProvider = 'ollama' | 'openai' | 'openai_compatible' | 'anthropic';

/** Per-question options beyond the retrieval overrides. */
/** A whole-document task (POST /questions/summary): summarize one document, or
 *  compare two. Kept on the turn that asked it, so Retry and Regenerate repeat it. */
export interface DocumentTask {
  /** One document summarized, two compared, or several in one brief. */
  mode: 'summarize' | 'compare' | 'brief';
  filenames: string[];
  /** What to concentrate on; a general summary when absent. */
  instruction?: string;
}

/** GET /workspaces: the workspaces besides the default one. */
export interface WorkspaceListResponse {
  enabled: boolean;
  workspaces: string[];
}

/** How much of the documents a summary read (fewer than all when sampled). */
export interface SummaryCoverage {
  read: number;
  total: number;
}

export interface QuestionOptions {
  /** Ask this whole-document task instead of a retrieval question. */
  task?: DocumentTask;
  /** Only documents carrying one of these tags. */
  tags?: string[];
  /** Skip the server's answer cache (Regenerate). */
  bypassCache?: boolean;
  /** Client-only, never sent: the model asked, recorded on the answer it produces. */
  modelLabel?: string;
}

export type TraceStatus = 'ok' | 'insufficient_context' | 'error';
export type TraceMode = 'sync' | 'stream';
export type TraceDropReason =
  | 'below_min_score'
  | 'near_duplicate'
  | 'top_k_cut'
  | 'not_scored'
  | 'context_budget'
  // Kept by the reranker, but past max_context_chunks.
  | 'context_limit'
  // The server declares drop_reason extensible (x-extensible-enum): a reason added there
  // must still type-check here, and is shown with its underscores replaced.
  | (string & {});

export interface PromptMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface TraceConfigResponse {
  llm_provider: string;
  rerank_top_k: number;
  max_context_chunks: number;
  llm_temperature: number;
  rerank_min_score: number;
  fused_top_n: number;
  filenames: string[] | null;
  tags?: string[] | null;
}

export interface TraceCandidateResponse {
  point_id: string;
  filename: string;
  page: number;
  section: string;
  chunk_id: string;
  retrieval_score: number;
  rerank_score: number | null;
  kept: boolean;
  drop_reason: TraceDropReason | null;
  selected_for_context: boolean;
  neighbor_expanded: boolean;
}

export interface TraceSummaryResponse {
  trace_id: string;
  created_at: number;
  question: string;
  mode: TraceMode;
  status: TraceStatus;
  llm_provider: string | null;
  total_ms: number | null;
  candidate_count: number;
  kept_count: number;
}

export interface TraceListResponse {
  traces: TraceSummaryResponse[];
}

export interface TraceDetailResponse {
  trace_id: string;
  created_at: number;
  question: string;
  mode: TraceMode;
  status: TraceStatus;
  config: TraceConfigResponse | null;
  candidates: TraceCandidateResponse[];
  prompt_messages: PromptMessage[] | null;
  answer: string | null;
  cited_source_numbers: number[];
  timings: TimingsResponse | null;
  error: string | null;
  condensed_question?: string | null;
  history_message_count?: number;
  query_variants?: string[];
  citation_retry_used?: boolean;
  /** The X-Request-ID of the request that produced it, for finding its log lines. */
  request_id?: string | null;
  usage?: UsageResponse | null;
}
