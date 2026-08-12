import type {
  CitationResponse,
  DocumentContentResponse,
  DocumentDeleteResponse,
  DocumentJobAcceptedResponse,
  DocumentJobStatusResponse,
  DocumentListResponse,
  HealthResponse,
  HistoryMessage,
  LlmProvider,
  PublicConfigResponse,
  QuestionOverrides,
  QuestionResponse,
  TimingsResponse,
  TraceDetailResponse,
  TraceListResponse,
} from './types';

export class ApiClientError extends Error {
  statusCode?: number;

  constructor(message: string, statusCode?: number) {
    super(message);
    this.name = 'ApiClientError';
    this.statusCode = statusCode;
  }
}

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '';

// Only relevant when the server has API_KEY set (see api/settings.py); with no key
// configured server-side, the header is harmless to send and ignored.
const API_KEY_STORAGE_KEY = 'docrag-api-key';

export function getApiKey(): string {
  try {
    return localStorage.getItem(API_KEY_STORAGE_KEY) ?? '';
  } catch {
    return '';
  }
}

export function setApiKey(key: string): void {
  try {
    if (key) localStorage.setItem(API_KEY_STORAGE_KEY, key);
    else localStorage.removeItem(API_KEY_STORAGE_KEY);
  } catch {
    // Best-effort; the key still applies for the rest of this session via the
    // in-memory value read on the next request even if persistence failed.
  }
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
    throw new ApiClientError(await extractErrorDetail(response), response.status);
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

// How often to poll a background ingest job's status.
const JOB_POLL_INTERVAL_MS = 1_000;

function questionRequestBody(
  question: string,
  llmProvider?: LlmProvider,
  overrides?: QuestionOverrides,
  filenames?: string[],
  history?: HistoryMessage[],
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
  return body;
}

export interface QuestionStreamHandlers {
  onSources?: (sources: CitationResponse[], traceId: string | null) => void;
  onDelta?: (text: string) => void;
  onDone?: (
    answer: string,
    sources: CitationResponse[],
    timings: TimingsResponse | null,
    traceId: string | null,
  ) => void;
}

type QuestionStreamEvent =
  | { type: 'sources'; sources: CitationResponse[]; trace_id: string | null }
  | { type: 'delta'; text: string }
  | {
      type: 'done';
      answer: string;
      sources: CitationResponse[];
      timings: TimingsResponse;
      trace_id: string | null;
    }
  | { type: 'error'; detail: string };

async function readSseStream(
  response: Response,
  onEvent: (event: QuestionStreamEvent) => void,
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

  getDocumentJob: (jobId: string, signal?: AbortSignal) =>
    request<DocumentJobStatusResponse>(`/documents/jobs/${jobId}`, { signal }),

  deleteDocument: (filename: string, signal?: AbortSignal) =>
    request<DocumentDeleteResponse>(`/documents/${encodeURIComponent(filename)}`, {
      method: 'DELETE',
      signal,
    }),

  // Reconstructs a document from its stored chunks (ordinal order) for the source
  // viewer — no original file is retained, so this is the only content route.
  getDocumentContent: (filename: string, signal?: AbortSignal) =>
    request<DocumentContentResponse>(
      `/documents/${encodeURIComponent(filename)}/content`,
      { signal },
    ),

  // Trace detail is fetched lazily — only when a debug drawer is actually opened —
  // since the full prompt + candidate list can be tens of KB per question and most
  // turns are never inspected.
  getTrace: (traceId: string, signal?: AbortSignal) =>
    request<TraceDetailResponse>(`/traces/${traceId}`, { signal }),

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
    while (true) {
      const status = await request<DocumentJobStatusResponse>(`/documents/jobs/${jobId}`, {
        signal,
      });
      onProgress?.(status);
      if (status.state === 'done' || status.state === 'failed') {
        return status;
      }
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
        }, JOB_POLL_INTERVAL_MS);
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
  ) =>
    request<QuestionResponse>('/questions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(
        questionRequestBody(question, llmProvider, overrides, filenames, history),
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
  ): Promise<void> => {
    const timeoutController = new AbortController();
    const timeoutId = setTimeout(() => timeoutController.abort(), LONG_RUNNING_TIMEOUT_MS);
    const abortFromCaller = () => timeoutController.abort();
    signal?.addEventListener('abort', abortFromCaller);

    try {
      let response: Response;
      try {
        response = await fetch(`${BASE_URL}/questions/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...authHeaders() },
          body: JSON.stringify(
            questionRequestBody(question, llmProvider, overrides, filenames, history),
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
        throw new ApiClientError(await extractErrorDetail(response), response.status);
      }

      // Body reading needs the same error translation the fetch above already has.
      // Once headers arrive, an abort surfaces as a raw DOMException out of
      // reader.read() — outside the try/catch above — so it used to escape unwrapped
      // and every caller saw an unrecognised error instead of a cancellation.
      try {
        await readSseStream(response, (event) => {
          if (event.type === 'sources') handlers.onSources?.(event.sources, event.trace_id);
          else if (event.type === 'delta') handlers.onDelta?.(event.text);
          else if (event.type === 'done')
            handlers.onDone?.(event.answer, event.sources, event.timings, event.trace_id);
          else if (event.type === 'error') throw new ApiClientError(event.detail);
        });
      } catch (err) {
        // A backend `error` event already threw the right thing — don't re-wrap it.
        if (err instanceof ApiClientError) throw err;
        if (timeoutController.signal.aborted) {
          throw new ApiClientError('Request timed out or was cancelled.');
        }
        throw new ApiClientError(`API request failed: ${(err as Error).message}`);
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
