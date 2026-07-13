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

describe('uploadDocument', () => {
  it('posts multipart form data and returns the accepted job envelope', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse(202, { job_id: 'job-1', filename: 'guide.txt', state: 'queued' }));
    vi.stubGlobal('fetch', fetchMock);
    const file = new File(['content'], 'guide.txt', { type: 'text/plain' });

    const result = await api.uploadDocument(file);

    expect(result).toEqual({ job_id: 'job-1', filename: 'guide.txt', state: 'queued' });
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.method).toBe('POST');
    expect(init.body).toBeInstanceOf(FormData);
  });
});

describe('pollDocumentJob', () => {
  it('polls until a terminal state and calls onProgress for each poll', async () => {
    vi.useFakeTimers();
    const responses = [
      { job_id: 'j1', filename: 'a.txt', state: 'embedding', chunks_total: 4, chunks_done: 2, error: null, result: null },
      { job_id: 'j1', filename: 'a.txt', state: 'done', chunks_total: 4, chunks_done: 4, error: null, result: { filename: 'a.txt', sections_parsed: 1, chunks_ingested: 4, collection_name: 'c' } },
    ];
    let call = 0;
    const fetchMock = vi.fn(async () => jsonResponse(200, responses[call++]));
    vi.stubGlobal('fetch', fetchMock);
    const onProgress = vi.fn();

    const promise = api.pollDocumentJob('j1', onProgress);
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(1000);
    const result = await promise;

    expect(result.state).toBe('done');
    expect(onProgress).toHaveBeenCalledTimes(2);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    vi.useRealTimers();
  });
});

describe('askQuestionStream', () => {
  function sseResponse(frames: string[]): Response {
    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const frame of frames) controller.enqueue(encoder.encode(frame));
        controller.close();
      },
    });
    return new Response(stream, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    });
  }

  it('parses sources, delta, and done SSE events in order', async () => {
    const frames = [
      `data: ${JSON.stringify({ type: 'sources', sources: [], trace_id: 'trace-1' })}\n\n`,
      `data: ${JSON.stringify({ type: 'delta', text: 'Hel' })}\n\n`,
      `data: ${JSON.stringify({ type: 'delta', text: 'lo' })}\n\n`,
      `data: ${JSON.stringify({
        type: 'done',
        answer: 'Hello',
        sources: [],
        timings: { embed_ms: 1, search_ms: 1, rerank_ms: 1, generate_ms: 1, total_ms: 4 },
        trace_id: 'trace-1',
      })}\n\n`,
    ];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(sseResponse(frames)));

    const onSources = vi.fn();
    const onDelta = vi.fn();
    const onDone = vi.fn();

    await api.askQuestionStream('hi', undefined, undefined, undefined, {
      onSources,
      onDelta,
      onDone,
    });

    expect(onSources).toHaveBeenCalledWith([], 'trace-1');
    expect(onDelta).toHaveBeenNthCalledWith(1, 'Hel');
    expect(onDelta).toHaveBeenNthCalledWith(2, 'lo');
    expect(onDone).toHaveBeenCalledWith(
      'Hello',
      [],
      { embed_ms: 1, search_ms: 1, rerank_ms: 1, generate_ms: 1, total_ms: 4 },
      'trace-1',
    );
  });

  it('throws ApiClientError on a non-2xx response before reading the stream', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(400, { detail: 'Query text is required.' })),
    );

    await expect(
      api.askQuestionStream('', undefined, undefined, undefined, {}),
    ).rejects.toMatchObject({ name: 'ApiClientError', message: 'Query text is required.' });
  });

  it('includes filenames in the request body only when non-empty', async () => {
    const fetchMock = vi.fn().mockResolvedValue(sseResponse([]));
    vi.stubGlobal('fetch', fetchMock);

    await api.askQuestionStream('hi', undefined, undefined, ['a.txt'], {});

    const body = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    expect(body).toEqual({ question: 'hi', filenames: ['a.txt'] });
  });
});

describe('getTrace', () => {
  it('fetches the trace detail endpoint by id', async () => {
    const detail = {
      trace_id: 'trace-1',
      created_at: 0,
      question: 'alpha',
      mode: 'sync',
      status: 'ok',
      config: null,
      candidates: [],
      prompt_messages: null,
      answer: 'Alpha [1].',
      cited_source_numbers: [1],
      timings: null,
      error: null,
    };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, detail));
    vi.stubGlobal('fetch', fetchMock);

    const result = await api.getTrace('trace-1');

    expect(result).toEqual(detail);
    expect(String(fetchMock.mock.calls[0][0])).toContain('/traces/trace-1');
  });
});
