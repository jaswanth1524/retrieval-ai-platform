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

  it('sends tags and use_cache=false only when asked to', async () => {
    // A fresh Response per call: a body can only be read once.
    const fetchMock = vi.fn(async () => jsonResponse(200, { answer: 'x', sources: [] }));
    vi.stubGlobal('fetch', fetchMock);

    await api.askQuestion('hi', undefined, undefined, undefined, undefined, undefined, {
      tags: ['legal'],
      bypassCache: true,
    });
    await api.askQuestion('hi', undefined, undefined, undefined, undefined, undefined, { tags: [] });

    expect(JSON.parse(String(fetchMock.mock.calls[0][1].body))).toEqual({
      question: 'hi',
      tags: ['legal'],
      use_cache: false,
    });
    expect(JSON.parse(String(fetchMock.mock.calls[1][1].body))).toEqual({ question: 'hi' });
  });

  it('replaces a document\'s tags with a PATCH', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { filename: 'a b.pdf', tags: ['x'] }));
    vi.stubGlobal('fetch', fetchMock);

    const result = await api.setDocumentTags('a b.pdf', ['x']);

    expect(result.tags).toEqual(['x']);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain('/documents/a%20b.pdf/tags');
    expect(init.method).toBe('PATCH');
    expect(JSON.parse(String(init.body))).toEqual({ tags: ['x'] });
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

describe('deleteDocument', () => {
  it('sends a DELETE request to the encoded filename path', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse(200, { filename: 'a b.pdf', points_deleted: 3 }));
    vi.stubGlobal('fetch', fetchMock);

    const result = await api.deleteDocument('a b.pdf');

    expect(result).toEqual({ filename: 'a b.pdf', points_deleted: 3 });
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toBe('/documents/a%20b.pdf');
    expect((init as RequestInit).method).toBe('DELETE');
  });
});

describe('reindexDocument', () => {
  it('POSTs to the encoded per-document reindex route', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse(202, { job_id: 'j1', filename: 'a b.md', state: 'queued' }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const accepted = await api.reindexDocument('a b.md');

    expect(accepted.job_id).toBe('j1');
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toMatch(/\/documents\/a%20b\.md\/reindex$/);
    expect(init.method).toBe('POST');
  });
});

describe('reindexStaleDocuments', () => {
  it('POSTs to the bulk reindex route and returns jobs plus deferrals', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse(202, { jobs: [{ job_id: 'j1', filename: 'a.md', state: 'queued' }], deferred: ['b.md'] }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const result = await api.reindexStaleDocuments();

    expect(result.deferred).toEqual(['b.md']);
    expect(String(fetchMock.mock.calls[0][0])).toMatch(/\/documents\/reindex$/);
    expect(fetchMock.mock.calls[0][1].method).toBe('POST');
  });
});

describe('getDocumentContent', () => {
  it('requests the encoded content path', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse(200, { filename: 'a b.md', chunks: [] }));
    vi.stubGlobal('fetch', fetchMock);

    const result = await api.getDocumentContent('a b.md');

    expect(result).toEqual({ filename: 'a b.md', chunks: [] });
    expect(String(fetchMock.mock.calls[0][0])).toBe('/documents/a%20b.md/content');
  });
});

describe('listTraces', () => {
  it('requests the traces list endpoint', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { traces: [] }));
    vi.stubGlobal('fetch', fetchMock);

    const result = await api.listTraces();

    expect(result).toEqual({ traces: [] });
    expect(String(fetchMock.mock.calls[0][0])).toBe('/traces');
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

  const embedding = { job_id: 'j1', filename: 'a.txt', state: 'embedding', chunks_total: 4, chunks_done: 2, error: null, result: null };
  const done = { ...embedding, state: 'done', chunks_done: 4 };

  it('backs off between polls up to a 5s ceiling', async () => {
    vi.useFakeTimers();
    try {
      const fetchMock = vi.fn(async () => jsonResponse(200, embedding));
      vi.stubGlobal('fetch', fetchMock);
      const controller = new AbortController();
      const promise = api.pollDocumentJob('j1', undefined, controller.signal).catch(() => undefined);

      await vi.advanceTimersByTimeAsync(0);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      await vi.advanceTimersByTimeAsync(1_000); // 1s
      expect(fetchMock).toHaveBeenCalledTimes(2);
      await vi.advanceTimersByTimeAsync(1_499);
      expect(fetchMock).toHaveBeenCalledTimes(2); // second wait is 1.5s
      await vi.advanceTimersByTimeAsync(1);
      expect(fetchMock).toHaveBeenCalledTimes(3);
      await vi.advanceTimersByTimeAsync(60_000);
      // 2.25s, 3.375s, then capped at 5s: one poll per 5s from here on.
      const calls = fetchMock.mock.calls.length;
      await vi.advanceTimersByTimeAsync(50_000);
      expect(fetchMock.mock.calls.length - calls).toBe(10);
      controller.abort();
      await promise;
    } finally {
      vi.useRealTimers();
    }
  });

  it('rides out a few transient poll failures but not a 4xx', async () => {
    vi.useFakeTimers();
    try {
      const sequence = [
        () => Promise.reject(new TypeError('network down')),
        async () => jsonResponse(503, { detail: 'busy' }),
        async () => jsonResponse(200, done),
      ];
      let call = 0;
      vi.stubGlobal('fetch', vi.fn(() => sequence[call++]()));
      const promise = api.pollDocumentJob('j1');
      await vi.advanceTimersByTimeAsync(10_000);
      await expect(promise).resolves.toMatchObject({ state: 'done' });

      vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(404, { detail: 'Unknown job.' })));
      await expect(api.pollDocumentJob('gone')).rejects.toMatchObject({ statusCode: 404 });
    } finally {
      vi.useRealTimers();
    }
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

    await api.askQuestionStream('hi', undefined, undefined, undefined, undefined, {
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
      false,
    );
  });

  it('reports a mid-stream abort as ApiClientError, not a raw DOMException', async () => {
    // The response headers arrive, deltas flow, and only then does the caller abort —
    // so the rejection comes out of reader.read(), not out of fetch(). That path used
    // to escape unwrapped, and every caller saw an unrecognised error rather than a
    // cancellation.
    const encoder = new TextEncoder();
    const controller = new AbortController();
    let streamController!: ReadableStreamDefaultController<Uint8Array>;

    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        streamController = c;
        c.enqueue(
          encoder.encode(`data: ${JSON.stringify({ type: 'delta', text: 'partial' })}\n\n`),
        );
      },
    });
    controller.signal.addEventListener('abort', () => {
      streamController.error(new DOMException('The operation was aborted.', 'AbortError'));
    });

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } }),
      ),
    );

    const onDelta = vi.fn(() => {
      controller.abort();
    });

    await expect(
      api.askQuestionStream(
        'hi',
        undefined,
        undefined,
        undefined,
        undefined,
        { onDelta },
        controller.signal,
      ),
    ).rejects.toMatchObject({
      name: 'ApiClientError',
      message: 'Request timed out or was cancelled.',
    });
    expect(onDelta).toHaveBeenCalledWith('partial');
  });

  it('does not re-wrap the ApiClientError a backend error event raises', async () => {
    const frames = [
      `data: ${JSON.stringify({ type: 'error', detail: 'Reranker model failed: boom' })}\n\n`,
    ];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(sseResponse(frames)));

    await expect(
      api.askQuestionStream('hi', undefined, undefined, undefined, undefined, {}),
    ).rejects.toMatchObject({
      name: 'ApiClientError',
      message: 'Reranker model failed: boom',
    });
  });

  it('throws ApiClientError on a non-2xx response before reading the stream', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(400, { detail: 'Query text is required.' })),
    );

    await expect(
      api.askQuestionStream('', undefined, undefined, undefined, undefined, {}),
    ).rejects.toMatchObject({ name: 'ApiClientError', message: 'Query text is required.' });
  });

  const DONE_FRAME = `data: ${JSON.stringify({
    type: 'done',
    answer: '',
    sources: [],
    timings: null,
    trace_id: null,
  })}\n\n`;

  it('includes filenames in the request body only when non-empty', async () => {
    const fetchMock = vi.fn().mockResolvedValue(sseResponse([DONE_FRAME]));
    vi.stubGlobal('fetch', fetchMock);

    await api.askQuestionStream('hi', undefined, undefined, ['a.txt'], undefined, {});

    const body = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    expect(body).toEqual({ question: 'hi', filenames: ['a.txt'] });
  });

  it('includes history in the request body only when non-empty', async () => {
    const fetchMock = vi.fn(async () => sseResponse([DONE_FRAME]));
    vi.stubGlobal('fetch', fetchMock);

    await api.askQuestionStream('hi', undefined, undefined, undefined, [], {});
    let body = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    expect(body).toEqual({ question: 'hi' });

    await api.askQuestionStream(
      'hi',
      undefined,
      undefined,
      undefined,
      [{ role: 'user', content: 'earlier turn' }],
      {},
    );
    body = JSON.parse(String(fetchMock.mock.calls[1][1].body));
    expect(body).toEqual({
      question: 'hi',
      history: [{ role: 'user', content: 'earlier turn' }],
    });
  });
});

describe('askQuestionStream termination', () => {
  function frames(...events: object[]): Response {
    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const event of events)
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
        controller.close();
      },
    });
    return new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
  }

  it('rejects when the stream closes without a done event', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(frames({ type: 'delta', text: 'half an ans' })),
    );
    const onDelta = vi.fn();

    await expect(
      api.askQuestionStream('hi', undefined, undefined, undefined, undefined, { onDelta }),
    ).rejects.toMatchObject({
      name: 'ApiClientError',
      message: 'The answer stream ended before the server finished.',
    });
    expect(onDelta).toHaveBeenCalledWith('half an ans');
  });

  it('ignores a connection reset that lands after the done event', async () => {
    const encoder = new TextEncoder();
    const done = { type: 'done', answer: 'Full answer.', sources: [], timings: null, trace_id: 't1' };
    const stream = new ReadableStream<Uint8Array>({
      // error() discards queued chunks, so fail on the *next* pull — after the
      // reader has consumed the done frame, as a real reset would.
      start(controller) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(done)}\n\n`));
      },
      pull(controller) {
        if (controller.desiredSize !== null && controller.desiredSize > 0) {
          controller.error(new TypeError('network error'));
        }
      },
    });
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } }),
      ),
    );
    const onDone = vi.fn();

    await expect(
      api.askQuestionStream('hi', undefined, undefined, undefined, undefined, { onDone }),
    ).resolves.toBeUndefined();
    expect(onDone).toHaveBeenCalledWith('Full answer.', [], null, 't1', false);
  });

  it('still rejects a connection reset before the done event', async () => {
    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'delta', text: 'pa' })}\n\n`));
      },
      pull(controller) {
        if (controller.desiredSize !== null && controller.desiredSize > 0) {
          controller.error(new TypeError('network error'));
        }
      },
    });
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } }),
      ),
    );

    await expect(
      api.askQuestionStream('hi', undefined, undefined, undefined, undefined, {}),
    ).rejects.toMatchObject({ name: 'ApiClientError' });
  });

  it('bounds silence, not total duration: steady chunks keep a long stream alive', async () => {
    vi.useFakeTimers();
    try {
      const encoder = new TextEncoder();
      let push!: (text: string) => void;
      let close!: () => void;
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          push = (text) => controller.enqueue(encoder.encode(text));
          close = () => controller.close();
        },
      });
      vi.stubGlobal(
        'fetch',
        vi.fn().mockResolvedValue(
          new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } }),
        ),
      );
      const onDelta = vi.fn();
      const pending = api.askQuestionStream('hi', undefined, undefined, undefined, undefined, {
        onDelta,
      });
      // Four minutes of wall clock, never silent for more than a minute at a time.
      for (let minute = 0; minute < 4; minute += 1) {
        await vi.advanceTimersByTimeAsync(60_000);
        push(`data: ${JSON.stringify({ type: 'delta', text: `m${minute} ` })}\n\n`);
      }
      push(
        `data: ${JSON.stringify({ type: 'done', answer: 'ok', sources: [], timings: null, trace_id: null })}\n\n`,
      );
      close();
      await expect(pending).resolves.toBeUndefined();
      expect(onDelta).toHaveBeenCalledTimes(4);
    } finally {
      vi.useRealTimers();
    }
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

describe('Retry-After', () => {
  it('carries a 503 Retry-After on the error, so a full queue can be waited out', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify({ detail: 'Queue full.' }), {
          status: 503,
          headers: { 'Content-Type': 'application/json', 'Retry-After': '15' },
        }),
      ),
    );

    const error = await api.listDocuments().catch((err: unknown) => err);

    expect(error).toBeInstanceOf(ApiClientError);
    expect((error as ApiClientError).statusCode).toBe(503);
    expect((error as ApiClientError).retryAfterSeconds).toBe(15);
  });
});

describe('exportCorpus', () => {
  it('returns the zip and the filename the server suggested', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response('PK', {
          status: 200,
          headers: {
            'Content-Type': 'application/zip',
            'Content-Disposition': 'attachment; filename="docrag-export-20260929.zip"',
          },
        }),
      ),
    );

    const result = await api.exportCorpus();

    expect(result.filename).toBe('docrag-export-20260929.zip');
    expect(await result.blob.text()).toBe('PK');
  });

  it('reports a read-only key refusal', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(403, { detail: 'This API key is read-only.' })));

    await expect(api.exportCorpus()).rejects.toMatchObject({ statusCode: 403 });
  });
});
