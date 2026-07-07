import type {
  CitationResponse,
  DocumentIngestResponse,
  HealthResponse,
  LlmProvider,
  PublicConfigResponse,
  QuestionResponse,
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

// Without a timeout, a hung backend (a stalled Ollama call, a network blip) leaves
// the UI stuck on its loading state for the browser's own default (~300s) with no
// way to cancel. Question-answering routinely takes 70+ seconds, so it gets a much
// longer budget than health/config.
const DEFAULT_TIMEOUT_MS = 15_000;

interface RequestOptions extends RequestInit {
  timeoutMs?: number;
}

async function request<T>(path: string, options?: RequestOptions): Promise<T> {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, signal: callerSignal, ...init } = options ?? {};
  const timeoutController = new AbortController();
  const timeoutId = setTimeout(() => timeoutController.abort(), timeoutMs);
  // Let the caller's own signal (e.g. aborted on component unmount) cancel the
  // request too, without losing the timeout's independent ability to cancel it.
  const abortFromCaller = () => timeoutController.abort();
  callerSignal?.addEventListener('abort', abortFromCaller);

  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, { ...init, signal: timeoutController.signal });
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

export const api = {
  health: (signal?: AbortSignal) => request<HealthResponse>('/health', { signal }),

  config: (signal?: AbortSignal) => request<PublicConfigResponse>('/config', { signal }),

  uploadDocument: (file: File, signal?: AbortSignal) => {
    const form = new FormData();
    // Do NOT set Content-Type manually — the browser sets the multipart boundary.
    form.append('file', file);
    return request<DocumentIngestResponse>('/documents', {
      method: 'POST',
      body: form,
      timeoutMs: LONG_RUNNING_TIMEOUT_MS,
      signal,
    });
  },

  askQuestion: (question: string, llmProvider?: LlmProvider, signal?: AbortSignal) =>
    request<QuestionResponse>('/questions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      // Only include llm_provider when chosen, so an unset selection exercises the
      // backend's `| None` default (base provider) rather than pinning a value.
      body: JSON.stringify(llmProvider ? { question, llm_provider: llmProvider } : { question }),
      timeoutMs: LONG_RUNNING_TIMEOUT_MS,
      signal,
    }),
};

export type { CitationResponse, DocumentIngestResponse, HealthResponse, PublicConfigResponse, QuestionResponse };
