import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiClientError, api, extractErrorDetail } from '../../src/api/client';

afterEach(() => {
  vi.unstubAllGlobals();
});

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('extractErrorDetail', () => {
  it('returns a plain string detail (this app\'s custom exception handlers)', async () => {
    const response = jsonResponse(409, { detail: 'Re-ingest documents before querying.' });

    expect(await extractErrorDetail(response)).toBe('Re-ingest documents before querying.');
  });

  it('joins a FastAPI validation-error array detail (422)', async () => {
    const response = jsonResponse(422, {
      detail: [
        { loc: ['body', 'question'], msg: 'String should have at least 1 character', type: 'string_too_short' },
      ],
    });

    expect(await extractErrorDetail(response)).toBe('String should have at least 1 character');
  });

  it('falls back to a generic message for a non-JSON body', async () => {
    const response = new Response('not json', { status: 500 });

    expect(await extractErrorDetail(response)).toBe('API returned HTTP 500.');
  });
});

describe('api client', () => {
  it('throws ApiClientError with the extracted detail on a non-2xx response', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(409, { detail: 'Re-ingest documents before querying.' })),
    );

    await expect(api.askQuestion('hi')).rejects.toMatchObject({
      name: 'ApiClientError',
      message: 'Re-ingest documents before querying.',
      statusCode: 409,
    });
  });

  it('throws ApiClientError on a network failure', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValue(new TypeError('Failed to fetch')),
    );

    await expect(api.health()).rejects.toBeInstanceOf(ApiClientError);
  });

  it('includes llm_provider in the body when a provider is passed', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { answer: 'x', sources: [] }));
    vi.stubGlobal('fetch', fetchMock);

    await api.askQuestion('hi', 'openai');

    const body = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    expect(body).toEqual({ question: 'hi', llm_provider: 'openai' });
  });

  it('omits llm_provider from the body when no provider is passed', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { answer: 'x', sources: [] }));
    vi.stubGlobal('fetch', fetchMock);

    await api.askQuestion('hi');

    const body = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    expect(body).toEqual({ question: 'hi' });
    expect('llm_provider' in body).toBe(false);
  });

  it('includes only the set override fields in the body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { answer: 'x', sources: [] }));
    vi.stubGlobal('fetch', fetchMock);

    await api.askQuestion('hi', undefined, {
      rerankTopK: 3,
      maxContextChunks: null,
      llmTemperature: 1.5,
    });

    const body = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    expect(body).toEqual({ question: 'hi', rerank_top_k: 3, llm_temperature: 1.5 });
    expect('max_context_chunks' in body).toBe(false);
  });

  it('omits all override fields when overrides is undefined or all null', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { answer: 'x', sources: [] }));
    vi.stubGlobal('fetch', fetchMock);

    await api.askQuestion('hi', undefined, {
      rerankTopK: null,
      maxContextChunks: null,
      llmTemperature: null,
    });

    const body = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    expect(body).toEqual({ question: 'hi' });
  });

  it('aborts and reports a timeout when a request runs past its budget', async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn((_url: string, init?: RequestInit) => {
      return new Promise((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => {
          reject(new DOMException('The operation was aborted.', 'AbortError'));
        });
      });
    });
    vi.stubGlobal('fetch', fetchMock);

    const promise = api.health();
    const assertion = expect(promise).rejects.toMatchObject({
      name: 'ApiClientError',
      message: 'Request timed out or was cancelled.',
    });
    await vi.advanceTimersByTimeAsync(15_000);
    await assertion;

    vi.useRealTimers();
  });

  it('aborts the underlying fetch when a caller-provided signal is aborted', async () => {
    const fetchMock = vi.fn((_url: string, init?: RequestInit) => {
      return new Promise((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => {
          reject(new DOMException('The operation was aborted.', 'AbortError'));
        });
      });
    });
    vi.stubGlobal('fetch', fetchMock);
    const controller = new AbortController();

    const promise = api.health(controller.signal);
    controller.abort();

    await expect(promise).rejects.toMatchObject({ name: 'ApiClientError' });
  });
});
