import type {
  CitationResponse,
  DocumentIngestResponse,
  HealthResponse,
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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, init);
  } catch (err) {
    throw new ApiClientError(`API request failed: ${(err as Error).message}`);
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

export const api = {
  health: () => request<HealthResponse>('/health'),

  config: () => request<PublicConfigResponse>('/config'),

  uploadDocument: (file: File) => {
    const form = new FormData();
    // Do NOT set Content-Type manually — the browser sets the multipart boundary.
    form.append('file', file);
    return request<DocumentIngestResponse>('/documents', { method: 'POST', body: form });
  },

  askQuestion: (question: string) =>
    request<QuestionResponse>('/questions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    }),
};

export type { CitationResponse, DocumentIngestResponse, HealthResponse, PublicConfigResponse, QuestionResponse };
