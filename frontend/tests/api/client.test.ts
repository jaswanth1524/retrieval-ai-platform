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
});
