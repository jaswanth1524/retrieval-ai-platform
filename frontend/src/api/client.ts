import type {
  AccessResponse,
  CitationResponse,
  DocumentContentResponse,
  DocumentDeleteResponse,
  DocumentJobAcceptedResponse,
  DocumentJobStatusResponse,
  DocumentListResponse,
  DocumentReindexAllResponse,
  DocumentTagsResponse,
  FeedbackListResponse,
  FeedbackRequest,
  FeedbackResponse,
  HealthResponse,
  HistoryMessage,
  ImportResponse,
  LlmProvider,
  PublicConfigResponse,
  QuestionOptions,
  QuestionOverrides,
  QuestionResponse,
  TimingsResponse,
  TraceDetailResponse,
  TraceListResponse,
} from './types';
import { readStored, writeStored } from '../utils/safeStorage';

export class ApiClientError extends Error {
  statusCode?: number;
  /** Seconds from the response's Retry-After header (a 503 the server expects to clear). */
  retryAfterSeconds?: number;

  constructor(message: string, statusCode?: number, retryAfterSeconds?: number) {
    super(message);
    this.name = 'ApiClientError';
    this.statusCode = statusCode;
    this.retryAfterSeconds = retryAfterSeconds;
  }
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

function authHeaders(): Record<string, string> {
  const key = getApiKey();
  return key ? { 'X-API-Key': key } : {};
}

// Without a timeout, a hung backend (a stalled Ollama call, a network blip) leaves
// the UI stuck on its loading state for the browser's own default (~300s) with no
// way to cancel. Question-answering routinely takes 70+ seconds, so it gets a much
// longer budget than health/config.
const DEFAULT_TIMEOUT_MS = 15_000;

interface RequestOptions extends Omit<RequestInit, 'headers'> {
  timeoutMs?: number;
  headers?: Record<string, string>;
}

async function request<T>(path: string, options?: RequestOptions): Promise<T> {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, signal: callerSignal, headers, ...init } = options ?? {};
  const timeoutController = new AbortController();
  const timeoutId = setTimeout(() => timeoutController.abort(), timeoutMs);
  // Let the caller's own signal (e.g. aborted on component unmount) cancel the
  // request too, without losing the timeout's independent ability to cancel it.
  const abortFromCaller = () => timeoutController.abort();
  callerSignal?.addEventListener('abort', abortFromCaller);
  // A signal aborted before the call never fires 'abort' again, so the request ran.
  if (callerSignal?.aborted) timeoutController.abort();

  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers: { ...authHeaders(), ...headers },
      signal: timeoutController.signal,
    });
  } catch (err) {
    if (timeoutController.signal.aborted) {
      throw new ApiClientError('Request timed out or was cancelled.');
    }
    throw new ApiClientError(`API request failed: ${(err as Error).message}`);
  } finally {
    clearTimeout(timeoutId);
    callerSignal?.removeEventListener('abort', abortFromCaller);
  }
  if (!response.ok) {
    throw new ApiClientError(
      await extractErrorDetail(response),
      response.status,
      retryAfterSeconds(response),
    );
  }
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiClientError('API response was not valid JSON.');
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

function questionRequestBody(
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
  onStage?: (stage: string) => void;
  onSources?: (sources: CitationResponse[], traceId: string | null) => void;
  onDelta?: (text: string) => void;
  onDone?: (
    answer: string,
    sources: CitationResponse[],
    timings: TimingsResponse | null,
    traceId: string | null,
    cached?: boolean,
  ) => void;
}

type QuestionStreamEvent =
  | { type: 'stage'; stage: string }
  | { type: 'sources'; sources: CitationResponse[]; trace_id: string | null }
  | { type: 'delta'; text: string }
  | {
      type: 'done';
      answer: string;
      sources: CitationResponse[];
      timings: TimingsResponse;
      trace_id: string | null;
      cached?: boolean;
    }
  | { type: 'error'; detail: string };

async function readSseStream(
  response: Response,
  onEvent: (event: QuestionStreamEvent) => void,
  onChunk?: () => void,
): Promise<void> {
  if (!response.body) {
    throw new ApiClientError('Streaming response had no body.');
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      onChunk?.();
      buffer += decoder.decode(value, { stream: true });
      let separatorIndex = buffer.indexOf('\n\n');
      while (separatorIndex !== -1) {
        const rawEvent = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);
        const dataLine = rawEvent.split('\n').find((line) => line.startsWith('data: '));
        if (dataLine) {
          onEvent(JSON.parse(dataLine.slice('data: '.length)) as QuestionStreamEvent);
        }
        separatorIndex = buffer.indexOf('\n\n');
      }
    }
  } finally {
    // Leaving early — a backend `error` event, a malformed frame, an abort — used to
    // leave the body locked to this reader until GC. cancel() releases the lock and
    // tells the browser to stop pulling bytes we will never read.
    await reader.cancel().catch(() => {
      // Already errored or closed; nothing left to release.
    });
  }
}

export const api = {
  health: (signal?: AbortSignal) => request<HealthResponse>('/health', { signal }),

  config: (signal?: AbortSignal) => request<PublicConfigResponse>('/config', { signal }),

  // Which key this browser holds: "read" hides what the server would refuse (403).
  access: (signal?: AbortSignal) => request<AccessResponse>('/access', { signal }),

  // Recorded answer ratings, newest first (full key; 404 when feedback is off).
  listFeedback: (limit = 200, signal?: AbortSignal) =>
    request<FeedbackListResponse>(`/feedback?limit=${limit}`, { signal }),

  // Restores a backup zip from exportCorpus: each document is re-indexed as a job.
  importBackup: (file: File, signal?: AbortSignal) => {
    const form = new FormData();
    form.append('file', file);
    return request<ImportResponse>('/import', {
      method: 'POST',
      body: form,
      timeoutMs: LONG_RUNNING_TIMEOUT_MS,
      signal,
    });
  },

  // The exact uploaded bytes (RAW_DOCUMENT_DIR on). Fetched, not linked: a plain link
  // can't carry the X-API-Key header.
  getDocumentOriginal: async (filename: string): Promise<Blob> => {
    let response: Response;
    try {
      response = await fetch(`${BASE_URL}/documents/${encodeURIComponent(filename)}/original`, {
        headers: authHeaders(),
      });
    } catch (err) {
      throw new ApiClientError(`API request failed: ${(err as Error).message}`);
    }
    if (!response.ok) {
      throw new ApiClientError(await extractErrorDetail(response), response.status);
    }
    return response.blob();
  },

  listDocuments: (signal?: AbortSignal) =>
    request<DocumentListResponse>('/documents', { signal }),

  uploadDocument: (file: File, signal?: AbortSignal) => {
    const form = new FormData();
    // Do NOT set Content-Type manually — the browser sets the multipart boundary.
    form.append('file', file);
    return request<DocumentJobAcceptedResponse>('/documents', {
      method: 'POST',
      body: form,
      timeoutMs: LONG_RUNNING_TIMEOUT_MS,
      signal,
    });
  },

  // Re-chunks and re-embeds a document from its stored original; poll the job like an
  // upload's. 409 when the server keeps no originals, 404 when this one has none.
  reindexDocument: (filename: string, signal?: AbortSignal) =>
    request<DocumentJobAcceptedResponse>(`/documents/${encodeURIComponent(filename)}/reindex`, {
      method: 'POST',
      signal,
    }),

  // Re-indexes every document chunked by an older chunker that has a stored original.
  reindexStaleDocuments: (signal?: AbortSignal) =>
    request<DocumentReindexAllResponse>('/documents/reindex', { method: 'POST', signal }),

  // Replaces a document's tags (an empty list clears them). Payload-only on the server.
  setDocumentTags: (filename: string, tags: string[], signal?: AbortSignal) =>
    request<DocumentTagsResponse>(`/documents/${encodeURIComponent(filename)}/tags`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tags }),
      signal,
    }),

  // The whole-corpus backup zip (manifest, stored originals, feedback). Needs the full
  // API key when one is set. Returns the file and the name the server suggested.
  exportCorpus: async (): Promise<{ blob: Blob; filename: string }> => {
    let response: Response;
    try {
      response = await fetch(`${BASE_URL}/export`, { headers: authHeaders() });
    } catch (err) {
      throw new ApiClientError(`API request failed: ${(err as Error).message}`);
    }
    if (!response.ok) {
      throw new ApiClientError(await extractErrorDetail(response), response.status);
    }
    const disposition = response.headers.get('Content-Disposition') ?? '';
    const filename = /filename="([^"]+)"/.exec(disposition)?.[1] ?? 'docrag-export.zip';
    return { blob: await response.blob(), filename };
  },

  deleteDocument: (filename: string, signal?: AbortSignal) =>
    request<DocumentDeleteResponse>(`/documents/${encodeURIComponent(filename)}`, {
      method: 'DELETE',
      signal,
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
  listTraces: (signal?: AbortSignal) =>
    request<TraceListResponse>('/traces', { signal }),

  // Uploads and ingests run in a background job (see api.uploadDocument) — this
  // polls the job status endpoint until it reaches a terminal state, so the caller
  // never has to hold one long-lived request open for a large document's ingest.
  pollDocumentJob: async (
    jobId: string,
    onProgress?: (status: DocumentJobStatusResponse) => void,
    signal?: AbortSignal,
  ): Promise<DocumentJobStatusResponse> => {
    let intervalMs = JOB_POLL_INITIAL_MS;
    let transientFailures = 0;
    while (true) {
      let status: DocumentJobStatusResponse | null = null;
      try {
        status = await request<DocumentJobStatusResponse>(`/documents/jobs/${jobId}`, {
          signal,
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

  askQuestion: (
    question: string,
    llmProvider?: LlmProvider,
    overrides?: QuestionOverrides,
    signal?: AbortSignal,
    filenames?: string[],
    history?: HistoryMessage[],
    options?: QuestionOptions,
  ) =>
    request<QuestionResponse>('/questions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(
        questionRequestBody(question, llmProvider, overrides, filenames, history, options),
      ),
      timeoutMs: LONG_RUNNING_TIMEOUT_MS,
      signal,
    }),

  // Streams sources -> answer deltas -> a final done event over SSE, so the UI can
  // render citations and incremental text instead of waiting the full ~70s+ for a
  // complete answer. Falls back to nothing special on failure — errors (including a
  // mid-stream `error` event from the backend) reject the returned promise the same
  // way askQuestion's non-streaming failures do.
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
    const timeoutController = new AbortController();
    let timeoutId = setTimeout(() => timeoutController.abort(), STREAM_IDLE_TIMEOUT_MS);
    const resetIdleTimeout = () => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => timeoutController.abort(), STREAM_IDLE_TIMEOUT_MS);
    };
    const abortFromCaller = () => timeoutController.abort();
    signal?.addEventListener('abort', abortFromCaller);
    let sawDone = false;

    try {
      let response: Response;
      try {
        response = await fetch(`${BASE_URL}/questions/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...authHeaders() },
          body: JSON.stringify(
            questionRequestBody(question, llmProvider, overrides, filenames, history, options),
          ),
          signal: timeoutController.signal,
        });
      } catch (err) {
        if (timeoutController.signal.aborted) {
          throw new ApiClientError('Request timed out or was cancelled.');
        }
        throw new ApiClientError(`API request failed: ${(err as Error).message}`);
      }

      if (!response.ok) {
        throw new ApiClientError(
      await extractErrorDetail(response),
      response.status,
      retryAfterSeconds(response),
    );
      }

      // Body reading needs the same error translation the fetch above already has.
      // Once headers arrive, an abort surfaces as a raw DOMException out of
      // reader.read() — outside the try/catch above — so it used to escape unwrapped
      // and every caller saw an unrecognised error instead of a cancellation.
      try {
        await readSseStream(
          response,
          (event) => {
            if (event.type === 'stage') handlers.onStage?.(event.stage);
            else if (event.type === 'sources') handlers.onSources?.(event.sources, event.trace_id);
            else if (event.type === 'delta') handlers.onDelta?.(event.text);
            else if (event.type === 'done') {
              sawDone = true;
              handlers.onDone?.(
                event.answer,
                event.sources,
                event.timings,
                event.trace_id,
                event.cached ?? false,
              );
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
        if (timeoutController.signal.aborted) {
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
      clearTimeout(timeoutId);
      signal?.removeEventListener('abort', abortFromCaller);
    }
  },
};

export type {
  CitationResponse,
  DocumentContentResponse,
  DocumentDeleteResponse,
  DocumentJobAcceptedResponse,
  DocumentJobStatusResponse,
  DocumentListResponse,
  HealthResponse,
  PublicConfigResponse,
  QuestionResponse,
  TimingsResponse,
  TraceDetailResponse,
  TraceListResponse,
};
