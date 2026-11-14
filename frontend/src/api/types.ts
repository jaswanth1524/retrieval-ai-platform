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
  /** Each model the startup warmup loaded: "failed" makes the server degraded until a
   *  background retry loads it; "fallback" is the tokenizer's heuristic. Empty when
   *  warmup is off; absent on older servers. */
  models?: Record<string, 'ok' | 'failed' | 'fallback'>;
  /** With BACKUP_DIR: the last scheduled backup (null until one has run). */
  last_backup?: BackupStatusResponse | null;
}

/** The last scheduled backup written or attempted. */
export interface BackupStatusResponse {
  /** Unix seconds. */
  at: number;
  workspace: string;
  ok: boolean;
  file?: string | null;
  error?: string | null;
  /** Indexed documents the backup holds no original of. */
  originals_missing?: number;
}

/** Tokens the provider reported for an answer's LLM calls. */
export interface UsageResponse {
  prompt_tokens: number;
  completion_tokens: number;
  /** An estimate at the hosted model's list price; absent for local models. */
  cost_usd?: number | null;
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
  /** ANSWER_FORMAT: how answers are laid out unless a question asks otherwise. */
  answer_format?: string;
  /** QUERY_LOG_ENABLED: GET /query-log lists what the documents didn't answer. */
  query_log_enabled?: boolean;
  /** Replaced originals are kept (RAW_DOCUMENT_VERSIONS): versions can be restored. */
  document_versions_enabled?: boolean;
  /** TRACE_STORE_PATH: traces survive a restart and all of them can be exported. */
  traces_persisted?: boolean;
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
  /** Earlier versions put back (a backup with versions; absent from older servers). */
  versions_restored?: number;
}

/** POST /search: the passages a question would be answered from, with no LLM involved. */
export interface SearchRequest {
  /** The query; may be empty when `like_chunk_id` is given. */
  query: string;
  /** Search from this passage ("more like this") instead of the query. */
  like_chunk_id?: string;
  filenames?: string[];
  tags?: string[];
  rerank_top_k?: number;
  page_min?: number;
  page_max?: number;
  section?: string;
  exclude_filenames?: string[];
  chunk_ids?: string[];
  exclude_chunk_ids?: string[];
  uploaded_after?: number;
  uploaded_before?: number;
  /** Each result's rank on the dense and the BM25 search, and where an expected
   *  passage was lost (`explanation`). */
  explain?: boolean;
  expect_chunk_id?: string;
  expect_filename?: string;
}

/** Where an expected passage was lost on its way through a search. */
export interface SearchExplanationResponse {
  target: string;
  /** returned / not_retrieved / not_scored / below_min_score / near_duplicate /
   *  past_top_k (extensible). */
  outcome: string;
  detail: string;
  chunk_id?: string | null;
  dense_rank?: number | null;
  sparse_rank?: number | null;
  fused_rank?: number | null;
  rerank_rank?: number | null;
  rerank_score?: number | null;
  returned_rank?: number | null;
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
  /** The sentences sharing the most words with the query, for highlighting; absent
   *  from older servers. */
  spans?: CitationSpan[];
  /** With `explain`: its rank on the dense and the BM25 search (null: not returned). */
  dense_rank?: number | null;
  sparse_rank?: number | null;
}

export interface SearchResponse {
  results: SearchResultResponse[];
  timings: { embed_ms: number; search_ms: number; rerank_ms: number };
  /** Nothing cleared the relevance bar an answer's sources must clear; these are the
   *  closest passages (typical of a one- or two-word search). Absent from older servers. */
  weak_matches?: boolean;
  explanation?: SearchExplanationResponse | null;
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
  /** How answers are laid out; null = the server's ANSWER_FORMAT. */
  answerFormat?: AnswerFormat | null;
  /** A language name or "match"; null = the server's ANSWER_LANGUAGE. */
  answerLanguage?: string | null;
  /** Answer from the matched passages alone, without their neighbouring text. */
  passagesOnly?: boolean;
  /** Say "not found" rather than answer from weak matches. */
  strongMatchesOnly?: boolean;
}

export interface DocumentIngestResponse {
  filename: string;
  sections_parsed: number;
  chunks_ingested: number;
  collection_name: string;
  /** Already indexed from identical bytes: left as it was. */
  unchanged?: boolean;
  /** Passages whose stored vectors were kept, and passages of the previous version removed. */
  chunks_reused?: number;
  chunks_removed?: number;
  /** Another indexed document with exactly these bytes (both are kept). */
  duplicate_of?: string | null;
  /** What indexing left out or couldn't read (a cap reached, a page OCR failed on). */
  warnings?: string[];
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
  /** Already being indexed: not queued a second time. */
  in_progress?: string[];
  /** Could not be queued; the others were. */
  failed?: string[];
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
  /** What started it: upload, url, reindex, restore, import, watch. */
  kind?: string;
  /** Failed, and POST /documents/jobs/{id}/retry can run it again. */
  retryable?: boolean;
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
  /** Points from two ingests, or fewer than were written: re-indexing repairs them. */
  incomplete_filenames?: string[];
  /** Have a stored original, so POST /documents/{f}/reindex works. */
  reindexable_filenames?: string[];
  /** Tagged documents only; an untagged one is absent. */
  tags?: Record<string, string[]>;
  /** Documents fetched by URL: where from (POST /documents/{f}/refresh fetches again). */
  source_urls?: Record<string, string>;
  /** The scheduled refresh of each document fetched by URL. */
  url_refresh?: Record<string, UrlRefreshStatusResponse>;
  /** Documents holding exactly the same bytes under different names. */
  duplicate_groups?: string[][];
}

export interface UrlRefreshStatusResponse {
  /** Its own interval in minutes (0 = never); null = the server's. */
  refresh_minutes?: number | null;
  last_checked?: number | null;
  /** unchanged, queued or failed. */
  last_outcome?: string | null;
  failures?: number;
  /** Unix seconds; null when it is never checked. */
  next_due?: number | null;
}

/** PATCH /documents/{f}/refresh-settings */
export interface DocumentRefreshSettingsResponse {
  filename: string;
  refresh_minutes: number | null;
}

/** POST /documents/preview: how a file would be indexed. */
export interface PreviewChunkResponse {
  ordinal: number;
  page: number;
  page_end?: number | null;
  section: string;
  text: string;
}

export interface DocumentPreviewResponse {
  filename: string;
  section_count: number;
  chunk_count: number;
  page_count: number;
  section_labels: string[];
  first_chunks: PreviewChunkResponse[];
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
  /** The cross-encoder's score in [0, 1]; null for a summary's passages (read in order,
   *  never ranked) and absent from older servers. */
  rerank_score?: number | null;
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
  /** A summary of part of one document: its pages from..to. */
  pageMin?: number;
  pageMax?: number;
}

/** GET /workspaces: the workspaces besides the default one. */
export interface WorkspaceListResponse {
  enabled: boolean;
  workspaces: string[];
}

/** How much of the documents a summary read (fewer than all when sampled). */
export interface SummaryCoverage {
  /** Parts left out after their call failed twice (the summary may miss them). */
  skipped?: number;
  /** Parts whose notes stopped at the token limit. */
  cutOff?: number;
  read: number;
  total: number;
}

/** How an answer is laid out (ANSWER_FORMAT, or per question). */
export type AnswerFormat = 'default' | 'bullets' | 'table' | 'brief';

/** Where in the documents a question looks: the same narrowing POST /search takes. */
export interface PassageNarrowing {
  page_min?: number;
  page_max?: number;
  /** Exactly this section label. */
  section?: string;
  exclude_filenames?: string[];
  /** Only these passages (picked in the Search panel). */
  chunk_ids?: string[];
  /** Leave these passages out ("ask again without this source"). */
  exclude_chunk_ids?: string[];
  /** Documents uploaded in this range (Unix seconds). */
  uploaded_after?: number;
  uploaded_before?: number;
}

export interface QuestionOptions {
  /** Ask this whole-document task instead of a retrieval question. */
  task?: DocumentTask;
  /** Narrow where the question looks. */
  narrowing?: PassageNarrowing;
  answerFormat?: AnswerFormat;
  /** A language name, or "match" the question's. */
  answerLanguage?: string;
  /** Only documents carrying one of these tags. */
  tags?: string[];
  /** Skip the server's answer cache (Regenerate). */
  bypassCache?: boolean;
  /** Client-only, never sent: the model asked, recorded on the answer it produces. */
  modelLabel?: string;
  /** Client-only (the X-Workspace header, not the body): the workspace to ask in, when
   *  it must not follow a later switch ("" = the default). Else the current one. */
  workspace?: string;
  /** Client-only: the answer (turn id) this one replaces once it is done — Regenerate
   *  and Reformat keep it as an earlier version of the new one. */
  replacesAnswer?: string;
  /** The trace of an answer cut off at the token limit: its rest is asked for
   *  (POST /questions/continue) instead of a new answer. */
  continueFrom?: string;
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
  // Over the floor, but far below the best passage (RERANK_RELATIVE_FLOOR).
  | 'relative_cut'
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
  /** The cross-encoder that scored the candidates. */
  reranker_model?: string | null;
  answer_format?: string;
  answer_language?: string;
  /** The question's narrowing (only the parts it had). */
  narrowing?: Record<string, unknown> | null;
  /** The model the provider was asked for. */
  llm_model?: string | null;
  /** Neighbouring passages added per side (0 = passages only). */
  context_neighbor_radius?: number | null;
  /** Closest passages used when none cleared the bar (0 = none). */
  rerank_fallback_count?: number | null;
}

/** GET /query-log: one logged question (QUERY_LOG_ENABLED). */
export interface QueryLogEntryResponse {
  created_at: number;
  question: string;
  status: string;
  top_score: number | null;
  mode: string;
  trace_id: string | null;
  /** The documents the answer was drawn from (empty on rows logged before this). */
  filenames?: string[];
  kind?: string;
  /** Served from the answer cache (or by joining the same question in flight). */
  cached?: boolean;
}

export interface QueryLogResponse {
  queries: QueryLogEntryResponse[];
}

/** GET /documents/{filename}/versions: replaced originals kept. */
export interface DocumentVersionResponse {
  version: string;
  saved_at: number;
  size: number;
}

export interface DocumentVersionsResponse {
  filename: string;
  versions: DocumentVersionResponse[];
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
  /** Its [n] in the prompt (null: not in it), and whether the answer cited it. */
  source_number?: number | null;
  cited?: boolean;
  /** Its rank on the dense and BM25 legs before fusion (null: not returned / not kept). */
  dense_rank?: number | null;
  sparse_rank?: number | null;
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
  /** answered, uncited, insufficient_context, no_context (extensible). */
  answer_status?: string | null;
  top_rerank_score?: number | null;
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
  /** Why a follow-up was not condensed, or its rewrite not used as returned. */
  condense_note?: string | null;
  /** The kept passages scored under the server's bar (weak-match fallback). */
  weak_matches?: boolean;
  history_message_count?: number;
  query_variants?: string[];
  citation_retry_used?: boolean;
  /** The model cited nothing; the citations were added by matching its sentences to the
   *  sources (CITATION_RETRY_MODE). Absent from older servers. */
  citation_backfilled?: boolean;
  /** The X-Request-ID of the request that produced it, for finding its log lines. */
  request_id?: string | null;
  usage?: UsageResponse | null;
  /** answered / uncited / insufficient_context / no_context (extensible); absent from
   *  older servers and on errors. */
  answer_status?: string | null;
  /** The answer stopped at the token limit. */
  truncated?: boolean;
  /** The provider read far less of the prompt than was sent. */
  prompt_cut?: boolean;
  /** The LLM reply came from the prompt cache (PROMPT_CACHE_SIZE). */
  prompt_cached?: boolean;
  /** Citing sentences no passage of their cited sources supports. */
  unsupported_claims?: number;
  /** History messages left out of the prompt to fit the context window. */
  dropped_history_messages?: number;
  /** "question", or "summary" (whose question is its task line). */
  kind?: string;
}

/** GET /traces/score-report: best rerank scores of answered against unanswered questions. */
export interface TraceRerunRequest {
  rerank_min_score?: number;
  rerank_top_k?: number;
  rerank_relative_floor?: number;
  recency_boost_weight?: number;
}

/** POST /traces/what-if: settings to try on the latest traced questions. */
export interface TraceWhatIfRequest {
  rerank_min_score?: number;
  rerank_relative_floor?: number;
  rerank_fallback_count?: number;
  limit?: number;
  only_unanswered?: boolean;
}

export interface TraceWhatIfItemResponse {
  trace_id: string;
  question: string;
  answered_before: boolean;
  would_answer: boolean;
  weak_matches?: boolean;
  top_score?: number | null;
  error?: string | null;
}

/** How many of the latest traced questions the settings tried would answer. */
export interface TraceWhatIfResponse {
  settings: Record<string, number>;
  checked: number;
  would_answer: number;
  newly_answered: number;
  newly_unanswered: number;
  items: TraceWhatIfItemResponse[];
  stopped_early?: boolean;
}

/** POST /traces/{id}/rerun: the question searched again under other ranking settings. */
export interface TraceRerunResponse {
  query: string;
  settings: Record<string, number>;
  would_answer: boolean;
  weak_matches: boolean;
  diff: TraceDiffResponse;
  candidates: TraceCandidateResponse[];
}

/** A workspace's own relevance settings (null: the server's). */
export interface WorkspaceSettingsRequest {
  rerank_min_score?: number | null;
  recency_boost_weight?: number | null;
  answer_language?: string | null;
  answer_format?: AnswerFormat | null;
  rerank_fallback_count?: number | null;
  rerank_relative_floor?: number | null;
  max_context_chunks?: number | null;
  followups_enabled?: boolean | null;
}

export interface WorkspaceSettingsResponse {
  workspace: string;
  settings: WorkspaceSettingsRequest;
  effective: WorkspaceSettingsRequest;
}

export interface CachedQuestionResponse {
  question: string;
  age_seconds: number;
}

/** GET /answer-cache: this workspace's cached answers. */
export interface AnswerCacheResponse {
  enabled: boolean;
  entries: CachedQuestionResponse[];
  total_entries: number;
  max_entries: number;
  ttl_seconds: number;
  in_flight: number;
}

export interface BackupFileResponse {
  name: string;
  size: number;
  created_at: number;
  /** Read back cleanly when written (null: not checked in this run). */
  verified?: boolean | null;
  problems?: string[];
}

/** GET /backups: this workspace's scheduled backups, newest first. */
export interface BackupListResponse {
  backups: BackupFileResponse[];
  interval_hours: number;
  keep: number;
}

export interface BackupCheckResponse {
  ok: boolean;
  documents: number;
  originals: number;
  originals_missing: number;
  versions: number;
  feedback: boolean;
  problems: string[];
  /** By restore's own rules: documents it would queue, and those it would refuse. */
  would_restore?: number;
  rejected?: string[];
}

/** POST /admin/consistency/repair: which findings to put right. */
export interface ConsistencyRepairRequest {
  delete_orphan_originals?: boolean;
  delete_orphan_versions?: boolean;
  reindex_incomplete?: boolean;
}

export interface ConsistencyRepairResponse {
  originals_deleted: string[];
  versions_deleted: string[];
  reindex_jobs: DocumentJobAcceptedResponse[];
  left: string[];
}

/** GET /admin/consistency: how the index and the stored originals agree. */
export interface ConsistencyResponse {
  documents: number;
  originals_kept: boolean;
  problems: number;
  originals_without_document: string[];
  documents_without_original: string[];
  versions_without_document: string[];
  incomplete: string[];
  stale: string[];
  duplicate_groups: string[][];
  in_flight: string[];
}

export interface ArchiveSkipResponse {
  name: string;
  reason: string;
}

/** POST /documents/archive: a zip's documents, each queued as its own upload. */
export interface ArchiveUploadResponse {
  jobs: DocumentJobAcceptedResponse[];
  skipped: ArchiveSkipResponse[];
  deferred: string[];
}

/** POST /documents/from-sitemap: the pages being fetched in the background. */
export interface SitemapIngestResponse {
  urls: string[];
  left_out: number;
  child_sitemaps: number;
  /** Follow it at GET /documents/sitemap-imports/{import_id}. */
  import_id?: string | null;
}

/** A sitemap import as it goes. */
export interface SitemapImportResponse {
  id: string;
  url: string;
  state: string;
  total: number;
  done: number;
  queued: number;
  failed: number;
  skipped: number;
  unchanged: number;
  cancelled_pages: number;
  started_at: number;
  finished_at?: number | null;
  problems?: string[];
}

export interface SitemapImportListResponse {
  imports: SitemapImportResponse[];
}

/** GET /admin/doctor: risky configurations of the server. A report only. */
export interface DoctorFindingResponse {
  /** "warning" or "note" (extensible). */
  level: string;
  subject: string;
  message: string;
}

export interface DoctorResponse {
  findings: DoctorFindingResponse[];
}

/** A zip or backup being imported in the background (GET /documents/imports/{id}). */
export interface BackgroundImportResponse {
  id: string;
  /** "archive" or "restore" (extensible). */
  kind: string;
  name: string;
  /** running / cancelling / done / cancelled. */
  state: string;
  started_at: number;
  finished_at?: number | null;
  queued?: number;
  skipped?: number;
  failed?: number;
  problems?: string[];
}

export interface BackgroundImportListResponse {
  imports: BackgroundImportResponse[];
}

/** GET /documents/{filename}/outline: sections and page starts, each with its first
 *  passage. */
export interface OutlineEntryResponse {
  section: string;
  page: number;
  chunk_id: string;
}

export interface DocumentOutlineResponse {
  filename: string;
  total_chunks: number;
  sections: OutlineEntryResponse[];
  pages: OutlineEntryResponse[];
  truncated?: boolean;
}

export interface ScoreReportResponse {
  current_min_score: number;
  /** answered[i] counts scores in [bin_edges[i], bin_edges[i + 1]). */
  bin_edges: number[];
  answered: number[];
  unanswered: number[];
  answered_count: number;
  unanswered_count: number;
  suggested_min_score?: number | null;
  note: string;
  would_change?: number;
  /** What decided each side: "model" (the answer's status) or "feedback" (ratings). */
  labels?: string;
}

export interface TraceSettingChangeResponse {
  name: string;
  a?: unknown;
  b?: unknown;
}

export interface TraceCandidateChangeResponse {
  point_id: string;
  filename: string;
  page: number;
  section?: string;
  chunk_id: string;
  rank_a?: number | null;
  rank_b?: number | null;
  score_a?: number | null;
  score_b?: number | null;
  selected_a?: boolean;
  selected_b?: boolean;
  drop_reason_a?: string | null;
  drop_reason_b?: string | null;
}

/** GET /traces/{a}/diff/{b} */
export interface TraceDiffResponse {
  question_a: string;
  question_b: string;
  same_answer: boolean;
  settings: TraceSettingChangeResponse[];
  candidates: TraceCandidateChangeResponse[];
}

export interface QueryRecheckItemResponse {
  question: string;
  status: string;
  covered_now: boolean;
  top_score?: number | null;
  top_filename?: string | null;
  top_chunk_id?: string | null;
}

/** POST /query-log/recheck */
export interface QueryRecheckResponse {
  checked: number;
  covered_now: number;
  items: QueryRecheckItemResponse[];
}
