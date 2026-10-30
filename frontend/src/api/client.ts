import type {
  AccessResponse,
  AnswerStatus,
  CitationResponse,
  DocumentContentResponse,
  DocumentDeleteResponse,
  DocumentJobAcceptedResponse,
  DocumentJobListResponse,
  DocumentJobStatusResponse,
  DocumentPreviewResponse,
  DocumentRefreshResponse,
  DocumentRefreshSettingsResponse,
  DocumentListResponse,
  DocumentReindexAllResponse,
  DocumentTagsResponse,
  DocumentVersionsResponse,
  FeedbackListResponse,
  FeedbackRequest,
  FeedbackResponse,
  SearchRequest,
  SearchResponse,
  HealthResponse,
  HistoryMessage,
  ImportResponse,
  LlmProvider,
  PassageNarrowing,
  PublicConfigResponse,
  QueryLogResponse,
  ReadinessResponse,
  UsageResponse,
  WatchStatusResponse,
  DocumentTask,
  QuestionOptions,
  QuestionOverrides,
  SummaryCoverage,
  WorkspaceListResponse,
  TimingsResponse,
  TraceDetailResponse,
  TraceListResponse,
} from './types';
import { readStored, writeStored } from '../utils/safeStorage';

import { ApiClientError } from './errors';
import { readSseStream } from './sse';

export { ApiClientError };

/** The API's own message for a failed call (already client-safe), else ``fallback``
 *  for anything that isn't an API error (a bug, a network stack's raw text). */
export function apiErrorMessage(err: unknown, fallback: string): string {
  return err instanceof ApiClientError ? err.message : fallback;
}

function retryAfterSeconds(response: Response): number | undefined {
  const value = Number(response.headers.get('Retry-After'));
  return Number.isFinite(value) && value > 0 ? value : undefined;
}

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '';

// Only relevant when the server has API_KEY set (see api/settings.py); with no key
// configured server-side, the header is harmless to send and ignored.
const API_KEY_STORAGE_KEY = 'docrag-api-key';

export function getApiKey(): string {
  return readStored(API_KEY_STORAGE_KEY) ?? '';
}

export function setApiKey(key: string): void {
  writeStored(API_KEY_STORAGE_KEY, key || null);
}

// The workspace this browser works in (WORKSPACES_ENABLED); '' is the default one,
// which needs no header — so a server without workspaces never sees it.
const WORKSPACE_STORAGE_KEY = 'docrag-workspace';
// Held per tab: storage only seeds the next tab. Read from storage on every request,
// switching in one tab silently sent the other tab's questions to the new workspace.
let currentWorkspace: string | null = null;

export function getWorkspace(): string {
  currentWorkspace ??= readStored(WORKSPACE_STORAGE_KEY) ?? '';
  return currentWorkspace;
}

export function setWorkspace(name: string): void {
  currentWorkspace = name;
  writeStored(WORKSPACE_STORAGE_KEY, name || null);
}

// Told when a request sent with a workspace's header comes back 404 "No workspace named
// …": that workspace was deleted elsewhere (another tab, the CLI). Given the workspace
// the request was sent with, which needn't be the one in use by then.
const WORKSPACE_MISSING_PREFIX = 'No workspace named';
let workspaceMissingListener: ((workspace: string) => void) | null = null;

export function onWorkspaceMissing(listener: ((workspace: string) => void) | null): void {
  workspaceMissingListener = listener;
}

function noteWorkspaceMissing(sentWorkspace: string, status: number, detail: string): void {
  if (status === 404 && sentWorkspace && detail.startsWith(WORKSPACE_MISSING_PREFIX)) {
    workspaceMissingListener?.(sentWorkspace);
  }
}

/** `workspace`: the one to send instead of the one in use — an operation started in a
 *  workspace keeps going to it after a switch ('' = the default, no header). */
function authHeaders(workspace?: string): Record<string, string> {
  const key = getApiKey();
  workspace ??= getWorkspace();
  return {
    ...(key ? { 'X-API-Key': key } : {}),
    ...(workspace ? { 'X-Workspace': workspace } : {}),
  };
}

// Without a timeout, a hung backend (a stalled Ollama call, a network blip) leaves
// the UI stuck on its loading state for the browser's own default (~300s) with no
// way to cancel. Question-answering routinely takes 70+ seconds, so it gets a much
// longer budget than health/config.
const DEFAULT_TIMEOUT_MS = 15_000;

interface RequestOptions extends Omit<RequestInit, 'headers'> {
  timeoutMs?: number;
  headers?: Record<string, string>;
  /** Sent instead of the workspace in use (see authHeaders). */
  workspace?: string;
}

/** An AbortSignal that fires ``timeoutMs`` after the last ``reset`` (or creation), or
 *  when ``callerSignal`` does. ``dispose`` once the request — its body included — is
 *  over. Shared by every request so each has a timeout and honours its caller's abort:
 *  one without either (the readiness check) hung when the server stopped answering. */
function deadline(
  timeoutMs: number,
  callerSignal?: AbortSignal | null,
): { signal: AbortSignal; reset: () => void; dispose: () => void } {
  const controller = new AbortController();
  let timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  const abortFromCaller = () => controller.abort();
  callerSignal?.addEventListener('abort', abortFromCaller);
  // A signal aborted before the call never fires 'abort' again, so the request ran.
  if (callerSignal?.aborted) controller.abort();
  return {
    signal: controller.signal,
    reset: () => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    },
    dispose: () => {
      clearTimeout(timeoutId);
      callerSignal?.removeEventListener('abort', abortFromCaller);
    },
  };
}

async function request<T>(path: string, options?: RequestOptions): Promise<T> {
  const {
    timeoutMs = DEFAULT_TIMEOUT_MS,
    signal: callerSignal,
    headers,
    workspace = getWorkspace(),
    ...init
  } = options ?? {};
  // The caller's own signal (e.g. aborted on unmount) cancels the request too, without
  // losing the timeout's independent ability to cancel it.
  const limit = deadline(timeoutMs, callerSignal);

  // The timeout and the caller's signal stay armed until the body is read: cleared once
  // the headers arrived, a server that then stalled mid-body hung the promise for good,
  // and aborting on unmount did nothing.
  try {
    let response: Response;
    try {
      response = await fetch(`${BASE_URL}${path}`, {
        ...init,
        headers: { ...authHeaders(workspace), ...headers },
        signal: limit.signal,
      });
    } catch (err) {
      if (limit.signal.aborted) {
        throw new ApiClientError('Request timed out or was cancelled.');
      }
      throw new ApiClientError(`API request failed: ${(err as Error).message}`);
    }
    if (!response.ok) {
      const detail = await extractErrorDetail(response);
      noteWorkspaceMissing(workspace, response.status, detail);
      throw new ApiClientError(detail, response.status, retryAfterSeconds(response));
    }
    try {
      return (await response.json()) as T;
    } catch {
      if (limit.signal.aborted) {
        throw new ApiClientError('Request timed out or was cancelled.');
      }
      throw new ApiClientError('API response was not valid JSON.');
    }
  } finally {
    limit.dispose();
  }
}

export async function extractErrorDetail(response: Response): Promise<string> {
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    return `API returned HTTP ${response.status}.`;
  }

  if (payload && typeof payload === 'object' && 'detail' in payload) {
    const detail = (payload as { detail: unknown }).detail;

    if (typeof detail === 'string' && detail) {
      return detail;
    }

    // FastAPI's own request-validation errors (422) shape `detail` as an array
    // of {loc, msg, type} objects, distinct from this app's custom handlers,
    // which always return `detail` as a plain string.
    if (Array.isArray(detail) && detail.length > 0) {
      return detail
        .map((item) =>
          item && typeof item === 'object' && 'msg' in item
            ? String((item as { msg: unknown }).msg)
            : JSON.stringify(item),
        )
        .join('; ');
    }
  }

  return `API returned HTTP ${response.status}.`;
}

// Question-answering and upload both run a full ingest/generation pipeline that
// routinely takes 70+ seconds; health/config use the short DEFAULT_TIMEOUT_MS.
const LONG_RUNNING_TIMEOUT_MS = 120_000;

// An upload's budget grows with its size: a fixed 120 s cut off a 50 MB file on any link
// slower than ~3.5 Mbit/s, and Retry failed the same way. 256 KiB/s (~2 Mbit/s) on top
// of the base, capped at 30 minutes.
const UPLOAD_MIN_BYTES_PER_SECOND = 256 * 1024;
const UPLOAD_MAX_TIMEOUT_MS = 30 * 60_000;

// A file downloaded whole (an original, a corpus backup). Without a limit a stalled
// export left "Download a backup" disabled until the page was reloaded.
const DOWNLOAD_TIMEOUT_MS = 10 * 60_000;

/** GET ``path`` as a Blob within DOWNLOAD_TIMEOUT_MS, the body read included. The
 *  caller's ``signal`` cancels it too (closing the viewer stops a 50 MB PDF download). */
async function downloadBlob(
  path: string,
  signal?: AbortSignal,
): Promise<{ response: Response; blob: Blob }> {
  const workspace = getWorkspace();
  const limit = deadline(DOWNLOAD_TIMEOUT_MS, signal);
  try {
    let response: Response;
    try {
      response = await fetch(`${BASE_URL}${path}`, {
        headers: authHeaders(workspace),
        signal: limit.signal,
      });
    } catch (err) {
      if (limit.signal.aborted) throw new ApiClientError('Request timed out or was cancelled.');
      throw new ApiClientError(`API request failed: ${(err as Error).message}`);
    }
    if (!response.ok) {
      const detail = await extractErrorDetail(response);
      noteWorkspaceMissing(workspace, response.status, detail);
      throw new ApiClientError(detail, response.status);
    }
    try {
      return { response, blob: await response.blob() };
    } catch {
      throw new ApiClientError('The download was interrupted or timed out.');
    }
  } finally {
    limit.dispose();
  }
}

export function uploadTimeoutMs(sizeBytes: number): number {
  const transferMs = (sizeBytes / UPLOAD_MIN_BYTES_PER_SECOND) * 1000;
  return Math.min(LONG_RUNNING_TIMEOUT_MS + Math.ceil(transferMs), UPLOAD_MAX_TIMEOUT_MS);
}

// A streamed answer can stay silent for a long stretch legitimately — rerank before
// the first frame (measured 46-74s on CPU), prefill before the first token, the
// unstreamed citation retry — so this bounds *silence*, re-armed on every chunk, not
// the whole answer. A fixed cap on the whole stream cut off default-config answers
// that measured 102-122s end to end while they were still streaming.
const STREAM_IDLE_TIMEOUT_MS = 180_000;

// Background ingest job polling: starts at 1s, backs off to 5s. With 2 ingest workers
// a large drop queues most jobs, and a flat 1s interval polled each of them every
// second for the whole wait.
const JOB_POLL_INITIAL_MS = 1_000;
const JOB_POLL_MAX_MS = 5_000;
// Consecutive transient failures (network blip, 5xx) tolerated before the upload is
// reported failed — the job keeps running server-side regardless, so one dropped poll
// used to show "failed" for a file that then appeared indexed.
const JOB_POLL_MAX_TRANSIENT_FAILURES = 3;

function isTransientPollError(err: unknown): boolean {
  if (!(err instanceof ApiClientError)) return false;
  return err.statusCode === undefined || err.statusCode >= 500;
}

export function questionRequestBody(
  question: string,
  llmProvider?: LlmProvider,
  overrides?: QuestionOverrides,
  filenames?: string[],
  history?: HistoryMessage[],
  options?: QuestionOptions,
): Record<string, unknown> {
  // Only include a field when it's set, so an unset value exercises the backend's
  // `| None` default (base provider/settings) rather than pinning a value.
  const body: Record<string, unknown> = { question };
  if (llmProvider) body.llm_provider = llmProvider;
  if (overrides?.rerankTopK != null) body.rerank_top_k = overrides.rerankTopK;
  if (overrides?.maxContextChunks != null) body.max_context_chunks = overrides.maxContextChunks;
  if (overrides?.llmTemperature != null) body.llm_temperature = overrides.llmTemperature;
  if (filenames && filenames.length > 0) body.filenames = filenames;
  if (history && history.length > 0) body.history = history;
  if (options?.tags && options.tags.length > 0) body.tags = options.tags;
  if (options?.bypassCache) body.use_cache = false;
  Object.assign(body, narrowingBody(options?.narrowing));
  const answerFormat = options?.answerFormat ?? overrides?.answerFormat;
  if (answerFormat) body.answer_format = answerFormat;
  const answerLanguage = options?.answerLanguage ?? overrides?.answerLanguage;
  if (answerLanguage) body.answer_language = answerLanguage;
  if (overrides?.passagesOnly) body.passages_only = true;
  if (overrides?.strongMatchesOnly) body.strong_matches_only = true;
  return body;
}

/** The narrowing fields that are set, as the server names them (questions and search). */
export function narrowingBody(narrowing: PassageNarrowing | undefined): Record<string, unknown> {
  const body: Record<string, unknown> = {};
  if (!narrowing) return body;
  if (narrowing.page_min != null) body.page_min = narrowing.page_min;
  if (narrowing.page_max != null) body.page_max = narrowing.page_max;
  if (narrowing.section) body.section = narrowing.section;
  if (narrowing.exclude_filenames && narrowing.exclude_filenames.length > 0) {
    body.exclude_filenames = narrowing.exclude_filenames;
  }
  if (narrowing.chunk_ids && narrowing.chunk_ids.length > 0) body.chunk_ids = narrowing.chunk_ids;
  if (narrowing.exclude_chunk_ids && narrowing.exclude_chunk_ids.length > 0) {
    body.exclude_chunk_ids = narrowing.exclude_chunk_ids;
  }
  if (narrowing.uploaded_after != null) body.uploaded_after = narrowing.uploaded_after;
  if (narrowing.uploaded_before != null) body.uploaded_before = narrowing.uploaded_before;
  return body;
}

function summaryRequestBody(
  task: DocumentTask,
  llmProvider?: LlmProvider,
  overrides?: QuestionOverrides,
  options?: QuestionOptions,
): Record<string, unknown> {
  // Only what a summary uses: retrieval overrides, scope and history don't apply to it.
  const body: Record<string, unknown> = { mode: task.mode, filenames: task.filenames };
  if (task.instruction) body.instruction = task.instruction;
  if (llmProvider) body.llm_provider = llmProvider;
  if (overrides?.llmTemperature != null) body.llm_temperature = overrides.llmTemperature;
  if (options?.bypassCache) body.use_cache = false;
  return body;
}

/** A page of GET /documents/{filename}/content. */
export interface ContentPage {
  around?: string;
  radius?: number;
  start?: number;
  end?: number;
}

export interface QuestionStreamHandlers {
  // Which pipeline stage the server just started, before any token exists. Retrieval
  // can run for tens of seconds on a whole-corpus question, so this is the difference
  // between a progress indicator and a frozen one.
  // `progress`: step of steps through a multi-step stage (a summary's parts).
  // `reading`: the prompt the model is about to read — its size, and how long the
  // local model should take to read it (etaMs, once the server has measured it).
  onStage?: (
    stage: string,
    progress?: { step: number; steps: number },
    reading?: { promptTokens?: number; etaMs?: number },
  ) => void;
  onSources?: (sources: CitationResponse[], traceId: string | null) => void;
  onDelta?: (text: string) => void;
  // Suggested next questions, after `done` (FOLLOWUPS_ENABLED on the server).
  onFollowups?: (questions: string[]) => void;
  onDone?: (
    answer: string,
    sources: CitationResponse[],
    timings: TimingsResponse | null,
    traceId: string | null,
    cached?: boolean,
    grounding?: {
      answerStatus?: AnswerStatus;
      topRerankScore?: number | null;
      /** No passage cleared the server's bar: answered from the closest ones. */
      weakMatches?: boolean;
      truncated?: boolean;
      /** The model cited nothing; the citations were matched to its sentences. */
      autoCited?: boolean;
      /** Summaries only: passages read, of the passages the documents hold. */
      coverage?: SummaryCoverage;
      /** Tokens the provider reported for this answer; absent when it reported none. */
      usage?: UsageResponse | null;
    },
  ) => void;
}

export const api = {
  health: (signal?: AbortSignal) => request<HealthResponse>('/health', { signal }),

  // Open, and 503 with the same body when degraded: that body is the answer here (an
  // index built for another embedding setup), so it is read whatever the status.
  readiness: async (signal?: AbortSignal): Promise<ReadinessResponse> => {
    // Its own fetch (a 503 carries the body to read), with the same timeout as every
    // other request: without one, a server that accepted the connection and never
    // answered left the readiness check, and the notices it drives, pending for good.
    const limit = deadline(DEFAULT_TIMEOUT_MS, signal);
    try {
      let response: Response;
      try {
        response = await fetch(`${BASE_URL}/health/ready`, { signal: limit.signal });
      } catch (err) {
        if (limit.signal.aborted) {
          throw new ApiClientError('Request timed out or was cancelled.');
        }
        throw new ApiClientError(`API request failed: ${(err as Error).message}`);
      }
      if (!response.ok && response.status !== 503) {
        throw new ApiClientError(await extractErrorDetail(response), response.status);
      }
      try {
        return (await response.json()) as ReadinessResponse;
      } catch {
        throw new ApiClientError('API response was not valid JSON.');
      }
    } finally {
      limit.dispose();
    }
  },

  // Neither depends on a workspace, so neither names one: a remembered workspace deleted
  // since would 404 /access (it resolves the request's workspace), which read as an
  // older server and granted full access.
  config: (signal?: AbortSignal) =>
    request<PublicConfigResponse>('/config', { signal, workspace: '' }),

  // Which key this browser holds: "read" hides what the server would refuse (403).
  access: (signal?: AbortSignal) => request<AccessResponse>('/access', { signal, workspace: '' }),

  // Recorded answer ratings, newest first (full key; 404 when feedback is off).
  listFeedback: (limit = 200, signal?: AbortSignal) =>
    request<FeedbackListResponse>(`/feedback?limit=${limit}`, { signal }),

  // Restores a backup zip from exportCorpus: each document is re-indexed as a job.
  importBackup: (file: File, signal?: AbortSignal, workspace?: string) => {
    const form = new FormData();
    form.append('file', file);
    return request<ImportResponse>('/import', {
      method: 'POST',
      body: form,
      // Scaled to the zip like an upload: a backup is often far larger than one
      // document, and a fixed two minutes failed big restores on a slow link.
      timeoutMs: uploadTimeoutMs(file.size),
      signal,
      workspace,
    });
  },

  // The exact uploaded bytes (RAW_DOCUMENT_DIR on). Fetched, not linked: a plain link
  // can't carry the X-API-Key header.
  getDocumentOriginal: async (filename: string, signal?: AbortSignal): Promise<Blob> =>
    (await downloadBlob(`/documents/${encodeURIComponent(filename)}/original`, signal)).blob,

  // `workspace` on the corpus calls below: the workspace the operation was started in
  // (see authHeaders); the one in use when absent.
  listDocuments: (signal?: AbortSignal, workspace?: string) =>
    request<DocumentListResponse>('/documents', { signal, workspace }),

  // `tags` (optional) replace the document's tags; giving any re-indexes the same bytes.
  uploadDocument: (file: File, signal?: AbortSignal, workspace?: string, tags?: string[]) => {
    const form = new FormData();
    // Do NOT set Content-Type manually — the browser sets the multipart boundary.
    form.append('file', file);
    for (const tag of tags ?? []) form.append('tags', tag);
    return request<DocumentJobAcceptedResponse>('/documents', {
      method: 'POST',
      body: form,
      timeoutMs: uploadTimeoutMs(file.size),
      signal,
      workspace,
    });
  },

  // How a file would be indexed (its sections and first passages), indexing nothing.
  previewDocument: (file: File, signal?: AbortSignal, workspace?: string) => {
    const form = new FormData();
    form.append('file', file);
    return request<DocumentPreviewResponse>('/documents/preview', {
      method: 'POST',
      body: form,
      timeoutMs: uploadTimeoutMs(file.size),
      signal,
      workspace,
    });
  },

  // Workspaces besides the default (GET answers {enabled: false} when they are off).
  listWorkspaces: (signal?: AbortSignal) =>
    request<WorkspaceListResponse>('/workspaces', { signal }),

  createWorkspace: (name: string) =>
    request<{ name: string }>('/workspaces', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }),

  // Only an empty one (409 while it holds documents).
  deleteWorkspace: (name: string) =>
    request<{ name: string }>(`/workspaces/${encodeURIComponent(name)}`, { method: 'DELETE' }),

  // The server fetches the URL's document and indexes it like an upload (URL_INGEST_ENABLED);
  // poll the job the same way. 400 names why a URL can't be fetched.
  // The server answers once it has fetched the document (up to URL_INGEST_TIMEOUT_SECONDS,
  // 30 s by default): the 15 s default reported a fetch still running as failed.
  uploadFromUrl: (url: string, signal?: AbortSignal, workspace?: string, tags?: string[]) =>
    request<DocumentJobAcceptedResponse>('/documents/from-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(tags && tags.length > 0 ? { url, tags } : { url }),
      timeoutMs: LONG_RUNNING_TIMEOUT_MS,
      signal,
      workspace,
    }),

  // Re-chunks and re-embeds a document from its stored original; poll the job like an
  // upload's. 409 when the server keeps no originals, 404 when this one has none.
  reindexDocument: (filename: string, signal?: AbortSignal, workspace?: string) =>
    request<DocumentJobAcceptedResponse>(`/documents/${encodeURIComponent(filename)}/reindex`, {
      method: 'POST',
      signal,
      workspace,
    }),

  // Stops an ingest job that is still queued or hasn't written to the index yet (409 once
  // it has: point ids are deterministic, so a half-written document can't be rolled
  // back). The job then ends `failed` with `cancelled: true`; keep polling to see it.
  cancelDocumentJob: (jobId: string, signal?: AbortSignal, workspace?: string) =>
    request<DocumentJobStatusResponse>(`/documents/jobs/${encodeURIComponent(jobId)}`, {
      method: 'DELETE',
      signal,
      workspace,
    }),

  // Runs a failed job again as it was asked for (a URL fetched again, a re-index);
  // 409 for an upload, whose bytes weren't kept.
  retryDocumentJob: (jobId: string, signal?: AbortSignal, workspace?: string) =>
    request<DocumentJobAcceptedResponse>(`/documents/jobs/${encodeURIComponent(jobId)}/retry`, {
      method: 'POST',
      timeoutMs: LONG_RUNNING_TIMEOUT_MS,
      signal,
      workspace,
    }),

  // How often a fetched document is checked again: minutes, 0 = never, null = the
  // server's interval.
  setRefreshInterval: (
    filename: string,
    refreshMinutes: number | null,
    signal?: AbortSignal,
    workspace?: string,
  ) =>
    request<DocumentRefreshSettingsResponse>(
      `/documents/${encodeURIComponent(filename)}/refresh-settings`,
      {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_minutes: refreshMinutes }),
        signal,
        workspace,
      },
    ),

  // Re-indexes every document chunked by an older chunker that has a stored original.
  reindexStaleDocuments: (signal?: AbortSignal, workspace?: string) =>
    request<DocumentReindexAllResponse>('/documents/reindex', { method: 'POST', signal, workspace }),

  // Replaces a document's tags (an empty list clears them). Payload-only on the server.
  setDocumentTags: (filename: string, tags: string[], signal?: AbortSignal, workspace?: string) =>
    request<DocumentTagsResponse>(`/documents/${encodeURIComponent(filename)}/tags`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tags }),
      signal,
      workspace,
    }),

  // The whole-corpus backup zip (manifest, stored originals, feedback). Needs the full
  // API key when one is set. Returns the file and the name the server suggested.
  exportCorpus: async (signal?: AbortSignal): Promise<{ blob: Blob; filename: string }> => {
    const { response, blob } = await downloadBlob('/export', signal);
    const disposition = response.headers.get('Content-Disposition') ?? '';
    const filename = /filename="([^"]+)"/.exec(disposition)?.[1] ?? 'docrag-export.zip';
    return { blob, filename };
  },

  deleteDocument: (filename: string, signal?: AbortSignal, workspace?: string) =>
    request<DocumentDeleteResponse>(`/documents/${encodeURIComponent(filename)}`, {
      method: 'DELETE',
      signal,
      workspace,
    }),

  // Reconstructs a document from its stored chunks (ordinal order) for the source
  // viewer (getDocumentOriginal serves the uploaded bytes themselves).
  // With no page, the whole document; `around` + `radius` for the chunks either side
  // of a cited one, or `start`/`end` (1-based ordinals, inclusive) for a range.
  getDocumentContent: (filename: string, signal?: AbortSignal, page: ContentPage = {}) => {
    const query = new URLSearchParams();
    if (page.around !== undefined) query.set('around', page.around);
    if (page.radius !== undefined) query.set('radius', String(page.radius));
    if (page.start !== undefined) query.set('start', String(page.start));
    if (page.end !== undefined) query.set('end', String(page.end));
    const suffix = query.size > 0 ? `?${query.toString()}` : '';
    return request<DocumentContentResponse>(
      `/documents/${encodeURIComponent(filename)}/content${suffix}`,
      { signal },
    );
  },

  // Queued and running indexing jobs in this workspace, whoever started them (the
  // watched folder, a restore, another tab).
  listActiveJobs: (signal?: AbortSignal, workspace?: string) =>
    request<DocumentJobListResponse>('/documents/jobs', { signal, workspace }),

  // Finished jobs the server still keeps, newest first: everything, or only failures.
  listFinishedJobs: (failedOnly: boolean, signal?: AbortSignal, workspace?: string) =>
    request<DocumentJobListResponse>(
      `/documents/jobs?state=${failedOnly ? 'failed' : 'recent'}&limit=50`,
      { signal, workspace },
    ),

  // The watched folder (WATCH_DIR) as of its last pass; 404 when none is watched.
  watchStatus: (signal?: AbortSignal) =>
    request<WatchStatusResponse>('/watch', { signal, workspace: '' }),

  // Runs the watcher's next pass now (full key).
  watchScan: () =>
    request<WatchStatusResponse>('/watch/scan', { method: 'POST', workspace: '' }),

  // Fetches a URL-ingested document again: a job to poll, or `unchanged` (304).
  refreshDocument: (filename: string, signal?: AbortSignal, workspace?: string) =>
    request<DocumentJobAcceptedResponse | DocumentRefreshResponse>(
      `/documents/${encodeURIComponent(filename)}/refresh`,
      { method: 'POST', timeoutMs: LONG_RUNNING_TIMEOUT_MS, signal, workspace },
    ),

  // Ranked passages without an answer: works with no LLM reachable. 429 when the server
  // is answering its maximum number of questions (it shares their slots).
  searchPassages: (body: SearchRequest, signal?: AbortSignal) =>
    request<SearchResponse>('/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      timeoutMs: LONG_RUNNING_TIMEOUT_MS,
      signal,
    }),

  submitFeedback: (body: FeedbackRequest, signal?: AbortSignal) =>
    request<FeedbackResponse>('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    }),

  // Trace detail is fetched lazily — only when a debug drawer is actually opened —
  // since the full prompt + candidate list can be tens of KB per question and most
  // turns are never inspected.
  getTrace: (traceId: string, signal?: AbortSignal) =>
    request<TraceDetailResponse>(`/traces/${encodeURIComponent(traceId)}`, { signal }),

  // Lists recent query traces (newest first) for the trace history browser.
  listTraces: (signal?: AbortSignal, answerStatuses: readonly string[] = []) =>
    request<TraceListResponse>(
      answerStatuses.length > 0
        ? `/traces?${answerStatuses.map((status) => `answer_status=${encodeURIComponent(status)}`).join('&')}`
        : '/traces',
      { signal },
    ),

  // Every trace of this workspace as JSON Lines (full key): all archived ones with
  // TRACE_STORE_PATH, else the ones the server still holds in memory.
  exportTraces: async (signal?: AbortSignal): Promise<{ blob: Blob; filename: string }> => {
    const { response, blob } = await downloadBlob('/traces/export', signal);
    const disposition = response.headers.get('Content-Disposition') ?? '';
    const filename = /filename="([^"]+)"/.exec(disposition)?.[1] ?? 'docrag-traces.jsonl';
    return { blob, filename };
  },

  // Every logged question and summary, newest first (QUERY_LOG_ENABLED; full key; 404
  // when the log is off). The Asked questions panel picks the unanswered ones out.
  listQueryLog: (limit = 500, signal?: AbortSignal, uncovered = false) =>
    request<QueryLogResponse>(`/query-log?limit=${limit}${uncovered ? '&uncovered=true' : ''}`, {
      signal,
    }),

  // Earlier originals of a document kept when it was replaced (RAW_DOCUMENT_VERSIONS).
  listDocumentVersions: (filename: string, signal?: AbortSignal, workspace?: string) =>
    request<DocumentVersionsResponse>(`/documents/${encodeURIComponent(filename)}/versions`, {
      signal,
      workspace,
    }),

  // Puts an earlier version back: indexed like an upload of those bytes; poll the job.
  restoreDocumentVersion: (filename: string, version: string, signal?: AbortSignal, workspace?: string) =>
    request<DocumentJobAcceptedResponse>(
      `/documents/${encodeURIComponent(filename)}/versions/${encodeURIComponent(version)}/restore`,
      { method: 'POST', signal, workspace },
    ),

  // Uploads and ingests run in a background job (see api.uploadDocument) — this
  // polls the job status endpoint until it reaches a terminal state, so the caller
  // never has to hold one long-lived request open for a large document's ingest.
  pollDocumentJob: async (
    jobId: string,
    onProgress?: (status: DocumentJobStatusResponse) => void,
    signal?: AbortSignal,
    workspace?: string,
  ): Promise<DocumentJobStatusResponse> => {
    let intervalMs = JOB_POLL_INITIAL_MS;
    let lastProgress = '';
    let transientFailures = 0;
    while (true) {
      let status: DocumentJobStatusResponse | null = null;
      try {
        status = await request<DocumentJobStatusResponse>(`/documents/jobs/${encodeURIComponent(jobId)}`, {
          signal,
          workspace,
        });
        transientFailures = 0;
      } catch (err) {
        if (signal?.aborted || !isTransientPollError(err)) throw err;
        transientFailures += 1;
        if (transientFailures > JOB_POLL_MAX_TRANSIENT_FAILURES) throw err;
      }
      if (status) {
        onProgress?.(status);
        if (status.state === 'done' || status.state === 'failed') {
          return status;
        }
        // Back to the fastest interval whenever the job moves: the backoff is for a job
        // waiting in the queue, and a finished one was reported up to 5 s late.
        const progress = `${status.state}:${status.chunks_done}`;
        if (progress !== lastProgress) {
          lastProgress = progress;
          intervalMs = JOB_POLL_INITIAL_MS;
        }
      }
      const waitMs = intervalMs;
      intervalMs = Math.min(Math.round(intervalMs * 1.5), JOB_POLL_MAX_MS);
      await new Promise((resolve, reject) => {
        // `{ once: true }` only removes the listener if it FIRES. When the timeout wins
        // — which is every tick of a normal ingest — the listener stays attached, so a
        // long upload accumulated one dead listener per second on the caller's signal.
        const onAbort = () => {
          clearTimeout(timeoutId);
          reject(new ApiClientError('Upload cancelled.'));
        };
        const timeoutId = setTimeout(() => {
          signal?.removeEventListener('abort', onAbort);
          resolve(undefined);
        }, waitMs);
        signal?.addEventListener('abort', onAbort, { once: true });
      });
    }
  },

  // Streams sources -> answer deltas -> a final done event over SSE, so the UI can
  // render citations and incremental text instead of waiting the full ~70s+ for a
  // complete answer. Errors (including a mid-stream `error` event from the backend)
  // reject the returned promise.
  askQuestionStream: async (
    question: string,
    llmProvider: LlmProvider | undefined,
    overrides: QuestionOverrides | undefined,
    filenames: string[] | undefined,
    history: HistoryMessage[] | undefined,
    handlers: QuestionStreamHandlers,
    signal?: AbortSignal,
    options?: QuestionOptions,
  ): Promise<void> => {
    // Bounds silence, not the whole answer: reset on every chunk (readSseStream).
    const limit = deadline(STREAM_IDLE_TIMEOUT_MS, signal);
    const resetIdleTimeout = limit.reset;
    let sawDone = false;
    const workspace = getWorkspace();

    try {
      let response: Response;
      try {
        const task = options?.task;
        response = await fetch(`${BASE_URL}${task ? '/questions/summary/stream' : '/questions/stream'}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...authHeaders(workspace) },
          body: JSON.stringify(
            task
              ? summaryRequestBody(task, llmProvider, overrides, options)
              : questionRequestBody(question, llmProvider, overrides, filenames, history, options),
          ),
          signal: limit.signal,
        });
      } catch (err) {
        if (limit.signal.aborted) {
          throw new ApiClientError('Request timed out or was cancelled.');
        }
        throw new ApiClientError(`API request failed: ${(err as Error).message}`);
      }

      if (!response.ok) {
        const detail = await extractErrorDetail(response);
        noteWorkspaceMissing(workspace, response.status, detail);
        throw new ApiClientError(detail, response.status, retryAfterSeconds(response));
      }

      // Body reading needs the same error translation the fetch above already has.
      // Once headers arrive, an abort surfaces as a raw DOMException out of
      // reader.read() — outside the try/catch above — so it used to escape unwrapped
      // and every caller saw an unrecognised error instead of a cancellation.
      try {
        await readSseStream(
          response,
          (event) => {
            if (event.type === 'stage') {
              const reading =
                typeof event.prompt_tokens === 'number' || typeof event.eta_ms === 'number'
                  ? {
                      ...(typeof event.prompt_tokens === 'number' ? { promptTokens: event.prompt_tokens } : {}),
                      ...(typeof event.eta_ms === 'number' ? { etaMs: event.eta_ms } : {}),
                    }
                  : undefined;
              const progress =
                typeof event.step === 'number' && typeof event.steps === 'number'
                  ? { step: event.step, steps: event.steps }
                  : undefined;
              if (reading) handlers.onStage?.(event.stage, progress, reading);
              else handlers.onStage?.(event.stage, progress);
            } else if (event.type === 'sources') handlers.onSources?.(event.sources, event.trace_id);
            else if (event.type === 'delta') handlers.onDelta?.(event.text);
            else if (event.type === 'done') {
              sawDone = true;
              handlers.onDone?.(
                event.answer,
                event.sources,
                event.timings,
                event.trace_id,
                event.cached ?? false,
                {
                  answerStatus: event.answer_status,
                  topRerankScore: event.top_rerank_score,
                  ...(event.weak_matches === true ? { weakMatches: true } : {}),
                  ...(event.citation_backfilled === true ? { autoCited: true } : {}),
                  truncated: event.truncated,
                  ...(event.usage ? { usage: event.usage } : {}),
                  ...(typeof event.chunks_read === 'number' && typeof event.chunks_total === 'number'
                    ? { coverage: { read: event.chunks_read, total: event.chunks_total } }
                    : {}),
                },
              );
            } else if (event.type === 'followups') {
              if (Array.isArray(event.questions)) {
                handlers.onFollowups?.(event.questions.filter((item) => typeof item === 'string'));
              }
            } else if (event.type === 'error') throw new ApiClientError(event.detail);
          },
          resetIdleTimeout,
        );
      } catch (err) {
        // The answer is already complete and delivered: a connection reset while the
        // server closes the stream must not stack an error turn under it.
        if (sawDone) return;
        // A backend `error` event already threw the right thing — don't re-wrap it.
        if (err instanceof ApiClientError) throw err;
        if (limit.signal.aborted) {
          throw new ApiClientError('Request timed out or was cancelled.');
        }
        throw new ApiClientError(`API request failed: ${(err as Error).message}`);
      }
      // A connection that closes cleanly without `done` (a proxy timeout, a server
      // restart) used to resolve as success — an answer with no error and no retry.
      if (!sawDone) {
        throw new ApiClientError('The answer stream ended before the server finished.');
      }
    } finally {
      limit.dispose();
    }
  },
};
