import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from '../src/App';

afterEach(() => {
  vi.unstubAllGlobals();
  // useChat persists chat history to localStorage — without clearing it, a test
  // whose App mount happens to boot after another test's successful ask-and-answer
  // rehydrates that turn history into its own fresh render, silently doubling turn
  // counts for any test that asks a question. Found via the feedback-button test
  // below, which is the first test sensitive to an exact element count post-answer.
  localStorage.clear();
});

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });
}

function makeConfigPayload(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    qdrant_collection: 'docrag_documents',
    dense_embedding_model: 'BAAI/bge-small-en-v1.5',
    sparse_embedding_model: 'Qdrant/BM25',
    reranker_model: 'jinaai/jina-reranker-v2-base-multilingual',
    embedding_model_tag: 'fastembed:BAAI/bge-small-en-v1.5',
    llm_provider: 'ollama',
    llm_model: 'llama3.1:8b',
    llm_temperature: 0,
    openai_model: 'gpt-4o-mini',
    openai_available: false,
    ollama_available: true,
    rrf_k: 60,
    dense_retrieval_limit: 50,
    sparse_retrieval_limit: 50,
    fused_top_n: 50,
    rerank_top_k: 8,
    max_context_chunks: 6,
    rerank_min_score: 0.3,
    max_upload_bytes: 52428800,
    rerank_top_k_limit: 50,
    max_context_chunks_limit: 20,
    llm_temperature_max: 2.0,
    ...overrides,
  };
}

async function openCorpusPanel(): Promise<void> {
  await userEvent.click(screen.getByTestId('rail-corpus'));
}

describe('App', () => {
  it('renders the shell and reaches "api ok" when health+config succeed', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);

    expect(await screen.findByText('api ok')).toBeInTheDocument();
    expect(screen.getByLabelText('DocRAG')).toBeInTheDocument();
    expect(screen.getByTestId('question-textarea')).toBeInTheDocument();
  });

  it('reports that an API key is required when the corpus list 401s', async () => {
    // /health and /config are unauthenticated, so a keyed server answers both and the
    // app looks healthy. Only /documents 401s — which used to be swallowed and shown as
    // an empty corpus, disabling the composer under "Upload a document to start."
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
        if (url.endsWith('/documents')) {
          return new Response(JSON.stringify({ detail: 'Missing or invalid API key.' }), {
            status: 401,
            headers: { 'Content-Type': 'application/json' },
          });
        }
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);

    expect(await screen.findByText('api key required')).toBeInTheDocument();
    expect(await screen.findByTestId('question-hint')).toHaveTextContent(
      'This server requires an API key',
    );
    // Crucially NOT the misleading advice, which would itself have 401'd.
    expect(screen.queryByText('Upload a document to start.')).not.toBeInTheDocument();

    // The whole point of the 401 state is that the fix is reachable: the shell stays
    // rendered, and Settings — which owns the Access field — actually opens.
    await userEvent.click(screen.getByTestId('open-settings'));
    expect(await screen.findByTestId('settings-modal')).toBeInTheDocument();
    expect(screen.getByTestId('settings-api-key')).toBeInTheDocument();
  });

  it('keeps the empty-corpus hint when the corpus list simply has no documents', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
        if (url.endsWith('/documents')) return jsonResponse({ filenames: [], chunk_counts: {} });
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);

    expect(await screen.findByText('api ok')).toBeInTheDocument();
    expect(screen.getByTestId('question-hint')).toHaveTextContent('Upload a document to start.');
  });

  it('drops a saved retrieval override the server no longer allows', async () => {
    localStorage.setItem(
      'docrag-advanced-options',
      JSON.stringify({ rerankTopK: 30, maxContextChunks: 4, llmTemperature: null }),
    );
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload({ rerank_top_k_limit: 20 }));
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);
    await screen.findByText('api ok');

    await waitFor(() =>
      expect(JSON.parse(localStorage.getItem('docrag-advanced-options') ?? '{}')).toEqual({
        rerankTopK: null,
        maxContextChunks: 4,
        llmTemperature: null,
      }),
    );
  });

  it('lowers saved context chunks to the saved rerank top K, which the server requires', async () => {
    localStorage.setItem(
      'docrag-advanced-options',
      JSON.stringify({ rerankTopK: 3, maxContextChunks: 6, llmTemperature: null }),
    );
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);
    await screen.findByText('api ok');

    await waitFor(() =>
      expect(JSON.parse(localStorage.getItem('docrag-advanced-options') ?? '{}')).toEqual({
        rerankTopK: 3,
        maxContextChunks: 3,
        llmTemperature: null,
      }),
    );
  });

  it('toggles the command palette with the keyboard shortcut', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
        if (url.endsWith('/documents')) return jsonResponse({ filenames: ['g.md'], chunk_counts: { 'g.md': 4 } });
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);
    await screen.findByText('api ok');

    await userEvent.keyboard('{Control>}k{/Control}');
    // Lazy-loaded: the first open resolves its chunk before rendering.
    expect(await screen.findByTestId('command-palette')).toBeInTheDocument();

    // Same chord closes it again.
    await userEvent.keyboard('{Control>}k{/Control}');
    expect(screen.queryByTestId('command-palette')).not.toBeInTheDocument();
  });

  it('runs a palette command — opening settings', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
        if (url.endsWith('/documents')) return jsonResponse({ filenames: ['g.md'], chunk_counts: { 'g.md': 4 } });
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);
    await screen.findByText('api ok');

    await userEvent.keyboard('{Control>}k{/Control}');
    await userEvent.type(screen.getByTestId('palette-input'), 'settings');
    await userEvent.keyboard('{Enter}');

    expect(await screen.findByTestId('settings-modal')).toBeInTheDocument();
    expect(screen.queryByTestId('command-palette')).not.toBeInTheDocument();
  });

  it('keeps Settings closed until the config has loaded, without blocking shortcuts', async () => {
    let answerConfig: (() => void) | undefined;
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/config')) {
          await new Promise<void>((resolve) => {
            answerConfig = resolve;
          });
          return jsonResponse(makeConfigPayload());
        }
        if (url.endsWith('/documents')) return jsonResponse({ filenames: ['g.md'], chunk_counts: { 'g.md': 4 } });
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);
    const button = screen.getByTestId('open-settings');
    expect(button).toBeDisabled();

    // The shortcut can't half-open it either, so ⌘K still works afterwards (a half-open
    // Settings made every shortcut think a dialog was up).
    await userEvent.keyboard('{Control>},{/Control}');
    await userEvent.keyboard('{Control>}k{/Control}');
    expect(await screen.findByTestId('command-palette')).toBeInTheDocument();
    await userEvent.keyboard('{Escape}');

    await waitFor(() => expect(answerConfig).toBeDefined());
    await act(async () => answerConfig?.());
    await screen.findByText('api ok');
    // Nothing pops up by itself when the config lands.
    expect(screen.queryByTestId('settings-modal')).not.toBeInTheDocument();
    expect(screen.getByTestId('open-settings')).not.toBeDisabled();
  });

  it('saving settings shows a toast instead of a banner', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
        if (url.endsWith('/documents')) return jsonResponse({ filenames: ['g.md'], chunk_counts: { 'g.md': 4 } });
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);
    await screen.findByText('api ok');

    await userEvent.click(screen.getByTestId('open-settings'));
    await userEvent.click(screen.getByTestId('settings-save'));

    expect(await screen.findByTestId('toast')).toHaveTextContent('Settings saved');
  });

  it('opens the inspector from the header toggle and closes it again', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
        if (url.endsWith('/documents')) return jsonResponse({ filenames: ['g.md'], chunk_counts: { 'g.md': 4 } });
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);
    await screen.findByText('api ok');

    expect(screen.queryByTestId('inspector')).not.toBeInTheDocument();
    await userEvent.click(screen.getByTestId('toggle-inspector'));
    expect(await screen.findByTestId('inspector')).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('inspector-close'));
    expect(screen.queryByTestId('inspector')).not.toBeInTheDocument();
  });

  it('closes an inspector opened on a wide window when the window narrows', async () => {
    // Controllable media queries: start wide (neither breakpoint), then go narrow.
    const listeners = new Map<string, (event: MediaQueryListEvent) => void>();
    let narrow = false;
    vi.stubGlobal('matchMedia', (query: string) => ({
      matches: query.includes('max-width: 1119px') ? narrow : false,
      media: query,
      addEventListener: (_type: string, listener: (event: MediaQueryListEvent) => void) =>
        listeners.set(query, listener),
      removeEventListener: () => listeners.delete(query),
    }));
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
        if (url.endsWith('/documents')) return jsonResponse({ filenames: ['g.md'], chunk_counts: { 'g.md': 4 } });
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);
    await screen.findByText('api ok');
    await userEvent.click(screen.getByTestId('toggle-inspector'));
    expect(await screen.findByTestId('inspector')).toBeInTheDocument();

    narrow = true;
    act(() => {
      listeners.get('(max-width: 1119px)')?.({ matches: true } as MediaQueryListEvent);
    });

    // Not turned into a modal drawer that grabs focus: closed instead.
    await waitFor(() => expect(screen.queryByTestId('inspector')).not.toBeInTheDocument());
    // Opened again on the narrow window, it is the drawer the user asked for.
    await userEvent.click(screen.getByTestId('toggle-inspector'));
    expect(await screen.findByTestId('inspector')).toBeInTheDocument();
  });

  it('a trace-row click opens the Trace tab and forces engineer mode', async () => {
    // A trace row is a debugging artifact — in reader mode the Trace tab does not
    // exist, so without forcing the mode the click would appear to do nothing.
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
        if (url.endsWith('/documents')) return jsonResponse({ filenames: ['g.md'], chunk_counts: { 'g.md': 4 } });
        if (url.endsWith('/traces')) {
          return jsonResponse({
            traces: [
              {
                trace_id: 'trace-42',
                created_at: Date.now() / 1000,
                question: 'What is in the guide?',
                mode: 'stream',
                status: 'ok',
                llm_provider: 'ollama',
                total_ms: 2310,
                candidate_count: 12,
                kept_count: 6,
              },
            ],
          });
        }
        if (url.includes('/traces/trace-42')) {
          return jsonResponse({
            trace_id: 'trace-42',
            created_at: 0,
            question: 'What is in the guide?',
            mode: 'stream',
            status: 'ok',
            config: null,
            candidates: [],
            prompt_messages: null,
            answer: 'answer',
            cited_source_numbers: [],
            timings: {
              embed_ms: 20,
              search_ms: 30,
              rerank_ms: 400,
              generate_ms: 1800,
              total_ms: 2310,
              condense_ms: 0,
              query_expansion_ms: 0,
              context_expansion_ms: 0,
            },
            error: null,
          });
        }
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);
    await screen.findByText('api ok');

    // Reader is the default mode.
    expect(screen.getByRole('radio', { name: /reader/i })).toBeChecked();

    await userEvent.click(screen.getByTestId('rail-traces'));
    await userEvent.click(await screen.findByTestId('traces-panel-row'));

    expect(await screen.findByTestId('inspector')).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /engineer/i })).toBeChecked();
    expect(await screen.findByTestId('trace-tab')).toBeInTheDocument();
  });

  it('releases a pinned trace when the conversation changes', async () => {
    // pinnedTraceId was set when a trace row was clicked and cleared only by toggling
    // the inspector shut. inspectedTraceId falls back to it whenever the turns array
    // no longer contains a matching turn — so after switching conversations the
    // Retrieval/Trace tabs kept rendering the previous conversation's trace while the
    // Sources tab (which reads the live turn) correctly went empty.
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
        if (url.endsWith('/documents')) return jsonResponse({ filenames: [], chunk_counts: {} });
        if (url.endsWith('/traces')) {
          return jsonResponse({
            traces: [
              {
                trace_id: 'trace-42',
                created_at: Date.now() / 1000,
                question: 'What is in the guide?',
                mode: 'stream',
                status: 'ok',
                llm_provider: 'ollama',
                total_ms: 2310,
                candidate_count: 12,
                kept_count: 6,
              },
            ],
          });
        }
        if (url.includes('/traces/trace-42')) {
          return jsonResponse({
            trace_id: 'trace-42',
            created_at: 0,
            question: 'What is in the guide?',
            mode: 'stream',
            status: 'ok',
            config: null,
            candidates: [],
            prompt_messages: null,
            answer: 'answer',
            cited_source_numbers: [],
            timings: {
              embed_ms: 20,
              search_ms: 30,
              rerank_ms: 400,
              generate_ms: 1800,
              total_ms: 2310,
              condense_ms: 0,
              query_expansion_ms: 0,
              context_expansion_ms: 0,
            },
            error: null,
          });
        }
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);
    await screen.findByText('api ok');

    await userEvent.click(screen.getByTestId('rail-traces'));
    await userEvent.click(await screen.findByTestId('traces-panel-row'));
    expect(await screen.findByTestId('trace-tab')).toBeInTheDocument();

    // Start a fresh conversation: the pinned trace belongs to the turns we just left.
    await userEvent.click(screen.getByTestId('rail-chat'));
    await userEvent.click(screen.getByTestId('context-panel-action'));

    expect(screen.queryByTestId('trace-tab')).not.toBeInTheDocument();
  });

  it('shows a clicked trace even when this conversation already has answers', async () => {
    // The inspector fell back to the latest answer whenever the pinned trace wasn't one
    // of this conversation's turns, so clicking an older trace showed the current
    // answer's trace instead.
    const traceDetail = (id: string, question: string) => ({
      trace_id: id, created_at: 0, question, mode: 'stream', status: 'ok', config: null,
      candidates: [], prompt_messages: null, answer: 'a', cited_source_numbers: [],
      timings: null, error: null,
    });
    const fetchedTraces: string[] = [];
    const encoder = new TextEncoder();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
        if (url.endsWith('/documents')) return jsonResponse({ filenames: ['g.md'], chunk_counts: { 'g.md': 1 } });
        if (url.endsWith('/questions/stream')) {
          const frames = [
            { type: 'sources', sources: [], trace_id: 'trace-current' },
            { type: 'done', answer: 'Answer.', sources: [], trace_id: 'trace-current',
              timings: { embed_ms: 1, search_ms: 1, rerank_ms: 1, generate_ms: 1, total_ms: 4 } },
          ];
          const body = new ReadableStream<Uint8Array>({
            start(controller) {
              for (const frame of frames) controller.enqueue(encoder.encode(`data: ${JSON.stringify(frame)}\n\n`));
              controller.close();
            },
          });
          return new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
        }
        if (url.endsWith('/traces')) {
          return jsonResponse({
            traces: [{ trace_id: 'trace-old', created_at: Date.now() / 1000, question: 'An older question',
              mode: 'stream', status: 'ok', llm_provider: 'ollama', total_ms: 10, candidate_count: 1, kept_count: 1 }],
          });
        }
        const match = url.match(/\/traces\/(trace-[a-z]+)/);
        if (match) {
          fetchedTraces.push(match[1]);
          return jsonResponse(traceDetail(match[1], match[1] === 'trace-old' ? 'An older question' : 'A question'));
        }
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);
    await screen.findByText('api ok');
    await userEvent.type(screen.getByTestId('question-textarea'), 'A question');
    await userEvent.click(screen.getByTestId('question-submit'));
    await screen.findByText('Answer.');

    await userEvent.click(screen.getByTestId('rail-traces'));
    await userEvent.click(await screen.findByTestId('traces-panel-row'));

    await waitFor(() => expect(fetchedTraces.at(-1)).toBe('trace-old'));
    expect(fetchedTraces).not.toContain('trace-current');

    // Closing and reopening from the palette follows the conversation again, as the
    // header toggle does; the palette's own copy of the toggle kept the old pin.
    for (const label of ['Hide inspector', 'Show inspector']) {
      await userEvent.keyboard('{Control>}k{/Control}');
      await userEvent.click(await screen.findByRole('option', { name: new RegExp(label) }));
    }
    await waitFor(() => expect(fetchedTraces.at(-1)).toBe('trace-current'));
  });

  it('does not fetch the trace — and does not show it as evicted — while the answer is still streaming', async () => {
    // Regression test for a real bug found via live end-to-end testing against real
    // Ollama generation (a real gap of several seconds between events, which a
    // synchronous mocked stream can't reproduce on its own — this test manually
    // controls frame delivery to recreate the gap). The server sends trace_id in the
    // `sources` SSE event, before generation starts; the trace itself is only written
    // server-side at the terminal (`done`) event. Fetching in that gap 404s, and
    // because traceId never changes across it, the inspector used to get permanently
    // stuck reporting a real trace as "no longer on the server."
    const encoder = new TextEncoder();
    let pushFrame: (frame: string) => void = () => {};
    let closeStream: () => void = () => {};
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        pushFrame = (frame: string) => controller.enqueue(encoder.encode(frame));
        closeStream = () => controller.close();
      },
    });

    const getTraceCalls: string[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
        if (url.endsWith('/documents')) return jsonResponse({ filenames: ['g.md'], chunk_counts: { 'g.md': 4 } });
        if (url.endsWith('/questions/stream')) {
          return new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
        }
        if (url.includes('/traces/trace-live')) {
          getTraceCalls.push(url);
          return jsonResponse({
            trace_id: 'trace-live',
            created_at: 0,
            question: 'q',
            mode: 'stream',
            status: 'ok',
            config: null,
            candidates: [],
            prompt_messages: null,
            answer: 'Grounded answer [1].',
            cited_source_numbers: [1],
            timings: {
              embed_ms: 20, search_ms: 30, rerank_ms: 400, generate_ms: 18000, total_ms: 18500,
              condense_ms: 0, query_expansion_ms: 0, context_expansion_ms: 5,
            },
            error: null,
          });
        }
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);
    await screen.findByText('api ok');

    // Open the inspector on Retrieval (which is trace-backed) BEFORE the trace_id
    // even exists, matching a user who has it open while asking a question.
    await userEvent.click(screen.getByTestId('toggle-inspector'));
    await userEvent.click(screen.getByRole('radio', { name: /engineer/i }));
    // findBy: the inspector is a lazy chunk, loaded already only if an earlier test
    // happened to open it.
    await userEvent.click(await screen.findByTestId('inspector-tab-retrieval'));

    await userEvent.type(screen.getByTestId('question-textarea'), 'A question');
    await userEvent.click(screen.getByTestId('question-submit'));

    // `sources` arrives: trace_id is now known, but the server has not written the
    // trace yet — real generation is still running.
    pushFrame(
      `data: ${JSON.stringify({ type: 'sources', sources: [], trace_id: 'trace-live' })}\n\n`,
    );
    await screen.findByTestId('inspector-trace-pending');
    expect(getTraceCalls).toHaveLength(0);

    // Real generation takes a while; nothing about the inspector should change or
    // fetch during it, however long it runs.
    pushFrame(`data: ${JSON.stringify({ type: 'delta', text: 'Grounded ' })}\n\n`);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(screen.getByTestId('inspector-trace-pending')).toBeInTheDocument();
    expect(getTraceCalls).toHaveLength(0);

    // Terminal event: this is the moment the server actually persists the trace.
    pushFrame(
      `data: ${JSON.stringify({
        type: 'done',
        answer: 'Grounded answer [1].',
        sources: [{ source_number: 1, filename: 'g.md', page: 1, section: 'S', chunk_id: 'c1', text: 't' }],
        timings: { embed_ms: 20, search_ms: 30, rerank_ms: 400, generate_ms: 18000, total_ms: 18500, condense_ms: 0 },
        trace_id: 'trace-live',
      })}\n\n`,
    );
    closeStream();

    expect(await screen.findByTestId('retrieval-tab')).toBeInTheDocument();
    expect(getTraceCalls).toHaveLength(1);
    // Never shown as evicted at any point — the whole bug was a false eviction verdict.
    expect(screen.queryByTestId('inspector-trace-unavailable')).not.toBeInTheDocument();
  });

  it('persists the toggled theme to localStorage under the docrag-theme key', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );
    const setItemSpy = vi.spyOn(Storage.prototype, 'setItem');

    render(<App />);
    await screen.findByText('api ok');

    await userEvent.click(screen.getByTestId('theme-toggle'));

    const themeAfterToggle = document.documentElement.dataset.theme;
    expect(setItemSpy).toHaveBeenCalledWith('docrag-theme', themeAfterToggle);
  });

  it('takes a theme picked in another tab, and follows the OS until one is picked', async () => {
    vi.restoreAllMocks();
    let systemListener: ((event: MediaQueryListEvent) => void) | undefined;
    vi.stubGlobal('matchMedia', (query: string) => ({
      matches: false,
      media: query,
      addEventListener: (_type: string, listener: (event: MediaQueryListEvent) => void) => {
        if (query.includes('prefers-color-scheme')) systemListener = listener;
      },
      removeEventListener: () => {},
    }));
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
        if (url.endsWith('/documents')) return jsonResponse({ filenames: [], chunk_counts: {} });
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );
    document.documentElement.dataset.theme = 'dark';
    render(<App />);
    await screen.findByText('api ok');

    act(() => systemListener?.({ matches: true } as MediaQueryListEvent));
    expect(document.documentElement.dataset.theme).toBe('light');

    act(() => {
      window.dispatchEvent(new StorageEvent('storage', { key: 'docrag-theme', newValue: 'dark' }));
    });
    expect(document.documentElement.dataset.theme).toBe('dark');
  });

  it('still flips the visible theme when localStorage.setItem throws', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('denied', 'SecurityError');
    });

    render(<App />);
    await screen.findByText('api ok');

    const initialTheme = document.documentElement.dataset.theme;
    await userEvent.click(screen.getByTestId('theme-toggle'));

    expect(document.documentElement.dataset.theme).not.toBe(initialTheme);
  });

  it('auto-switches the provider to OpenAI when Ollama is unreachable but OpenAI is available', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) {
          return jsonResponse(makeConfigPayload({ ollama_available: false, openai_available: true }));
        }
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);
    await screen.findByText('api ok');

    await userEvent.click(screen.getByTestId('composer-provider-button'));
    expect(screen.getByTestId('provider-openai')).toBeChecked();
  });

  it('does not keep a remembered Ollama that is down when OpenAI can answer', async () => {
    // The remembered choice used to win whether or not Ollama was reachable.
    vi.restoreAllMocks(); // an earlier test stubs Storage.setItem to throw
    localStorage.setItem('docrag-provider', 'ollama');
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) {
          return jsonResponse(makeConfigPayload({ ollama_available: false, openai_available: true }));
        }
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);
    await screen.findByText('api ok');

    await userEvent.click(screen.getByTestId('composer-provider-button'));
    expect(screen.getByTestId('provider-openai')).toBeChecked();
    // Only this session's pick: the stored preference is kept for when Ollama is back.
    expect(localStorage.getItem('docrag-provider')).toBe('ollama');
  });

  it('treats a failed document listing as unknown, not empty, and retries it', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      let listings = 0;
      vi.stubGlobal(
        'fetch',
        vi.fn(async (input: RequestInfo | URL) => {
          const url = String(input);
          if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
          if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
          if (url.endsWith('/documents')) {
            listings += 1;
            if (listings === 1) {
              return new Response(JSON.stringify({ detail: 'boom' }), {
                status: 500,
                headers: { 'Content-Type': 'application/json' },
              });
            }
            return jsonResponse({ filenames: ['guide.txt'], chunk_counts: { 'guide.txt': 1 } });
          }
          throw new Error(`unexpected fetch: ${url}`);
        }),
      );

      render(<App />);
      expect(await screen.findByTestId('question-hint')).toHaveTextContent(
        'Couldn’t load the document list',
      );
      expect(screen.queryByText('Upload a document to start.')).not.toBeInTheDocument();
      expect(screen.getByTestId('question-textarea')).not.toBeDisabled();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(5000);
      });
      await waitFor(() => expect(listings).toBe(2));
      await waitFor(() => expect(screen.queryByTestId('question-hint')).not.toBeInTheDocument());
    } finally {
      vi.useRealTimers();
    }
  });

  it('notices when the server goes away after boot, and when it comes back', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      let up = true;
      vi.stubGlobal(
        'fetch',
        vi.fn(async (input: RequestInfo | URL) => {
          const url = String(input);
          if (!up) throw new TypeError('Failed to fetch');
          if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
          if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
          if (url.endsWith('/documents')) {
            return jsonResponse({ filenames: ['g.md'], chunk_counts: { 'g.md': 1 } });
          }
          throw new Error(`unexpected fetch: ${url}`);
        }),
      );

      render(<App />);
      expect(await screen.findByText('api ok')).toBeInTheDocument();

      up = false;
      await act(async () => {
        await vi.advanceTimersByTimeAsync(30_000);
      });
      expect(await screen.findByText('api unreachable')).toBeInTheDocument();
      expect(screen.getByTestId('question-hint')).toHaveTextContent('Can’t reach the server');
      // The conversation stays on screen: not the boot-time "cannot reach" page.
      expect(screen.getByTestId('question-textarea')).toBeInTheDocument();

      up = true;
      await act(async () => {
        await vi.advanceTimersByTimeAsync(5_000);
      });
      expect(await screen.findByText('api ok')).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it('marks the server unreachable the moment the browser goes offline', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
        if (url.endsWith('/documents')) return jsonResponse({ filenames: ['g.md'], chunk_counts: { 'g.md': 1 } });
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );
    render(<App />);
    await screen.findByText('api ok');

    // Sent until the listener (registered once booted) has seen it.
    await waitFor(() => {
      act(() => {
        window.dispatchEvent(new Event('offline'));
      });
      expect(screen.getByText('api unreachable')).toBeInTheDocument();
    });
  });

  it('offers a skip link to the question as the first Tab stop', async () => {
    stubUploadAndQuestionFetch();
    render(<App />);
    await screen.findByText('api ok');
    await openCorpusPanel();
    await userEvent.upload(screen.getByTestId('upload-input'), [makeFile('a.txt')]);
    await userEvent.click(screen.getByTestId('upload-button'));
    await vi.waitFor(() => expect(screen.getByTestId('question-textarea')).toBeEnabled());
    (document.activeElement as HTMLElement | null)?.blur();

    await userEvent.tab();
    const skip = screen.getByRole('link', { name: 'Skip to question' });
    expect(skip).toHaveFocus();
    await userEvent.keyboard('{Enter}');
    expect(screen.getByTestId('question-textarea')).toHaveFocus();
  });

  it('keeps a passage search when another panel is opened and closed', async () => {
    const fetchMock = stubUploadAndQuestionFetch();
    const base = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) =>
      String(input).endsWith('/search')
        ? jsonResponse({
            results: [
              {
                rank: 1,
                filename: 'a.txt',
                page: 1,
                section: 'Intro',
                chunk_id: 'c1',
                chunk_ordinal: 1,
                text: 'Alpha passage.',
                retrieval_score: 0.1,
                rerank_score: 0.9,
              },
            ],
          })
        : base(input, init),
    );
    render(<App />);
    await screen.findByText('api ok');
    await userEvent.click(screen.getByTestId('rail-search'));
    await userEvent.type(screen.getByTestId('search-input'), 'alpha');
    await userEvent.click(screen.getByTestId('search-submit'));
    expect(await screen.findByTestId('search-result')).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('rail-chat'));
    expect(screen.queryByTestId('search-result')).not.toBeVisible();
    await userEvent.click(screen.getByTestId('rail-search'));

    expect(screen.getByTestId('search-input')).toHaveValue('alpha');
    expect(screen.getByTestId('search-result')).toBeVisible();
  });

  it('shows the unreachable banner when health check fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));

    render(<App />);

    expect(await screen.findByText('api unreachable')).toBeInTheDocument();
    expect(screen.getByText(/Cannot reach the DocRAG API/)).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent(/Cannot reach the DocRAG API/);
    // The toast live regions stay mounted, so a failure toast pushed now is announced.
    expect(screen.getByTestId('toast-failures')).toBeInTheDocument();
  });

  it('boots again once the API answers, by itself or on Retry now', async () => {
    let up = false;
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (!up) throw new TypeError('Failed to fetch');
      if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
      if (url.endsWith('/documents')) return jsonResponse({ filenames: [] });
      if (url.endsWith('/access')) return jsonResponse({ access: 'open' });
      throw new Error(`unexpected fetch: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      render(<App />);
      expect(await screen.findByText('api unreachable')).toBeInTheDocument();

      // Still down at the first automatic retry (2s later).
      const callsBefore = fetchMock.mock.calls.length;
      await act(() => vi.advanceTimersByTimeAsync(2000));
      expect(fetchMock.mock.calls.length).toBeGreaterThan(callsBefore);
      expect(screen.getByText('api unreachable')).toBeInTheDocument();

      up = true;
      await userEvent.click(screen.getByTestId('api-retry'));
      expect(await screen.findByText('api ok')).toBeInTheDocument();
      expect(screen.queryByTestId('api-retry')).not.toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it('fetches the indexed corpus on boot and renders it in the corpus panel', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
      if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
      if (url.endsWith('/documents') && (!init?.method || init.method === 'GET')) {
        return jsonResponse({ filenames: ['existing.pdf'], chunk_counts: { 'existing.pdf': 12 } });
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    render(<App />);
    await screen.findByText('api ok');
    await openCorpusPanel();

    expect(fetchMock.mock.calls.some(([input]) => String(input).endsWith('/documents'))).toBe(true);
    const items = await screen.findAllByTestId('corpus-panel-item');
    expect(items).toHaveLength(1);
    expect(items[0]).toHaveTextContent('existing.pdf');
    expect(items[0]).toHaveTextContent('12 chunks');
    expect(screen.getByTestId('context-panel-footer')).toHaveTextContent('12 chunks');
  });

  it('deleting an indexed document removes it from the corpus panel and search scope', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
      if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
      if (url.endsWith('/documents/existing.pdf') && init?.method === 'DELETE') {
        return jsonResponse({ filename: 'existing.pdf', points_deleted: 3 });
      }
      if (url.endsWith('/documents') && (!init?.method || init.method === 'GET')) {
        return jsonResponse({ filenames: ['existing.pdf'], chunk_counts: { 'existing.pdf': 3 } });
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    render(<App />);
    await screen.findByText('api ok');
    await openCorpusPanel();
    await screen.findAllByTestId('corpus-panel-item');

    await userEvent.click(screen.getByLabelText('Delete existing.pdf'));
    await userEvent.click(screen.getByText('Confirm'));

    await vi.waitFor(() => {
      expect(screen.queryAllByTestId('corpus-panel-item')).toHaveLength(0);
    });
  });

  it('disables the question input with an upload hint when no session documents exist', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
        if (url.endsWith('/documents')) return jsonResponse({ filenames: [], chunk_counts: {} });
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);
    await screen.findByText('api ok');

    expect(screen.getByTestId('question-textarea')).toBeDisabled();
    expect(screen.getByTestId('question-hint')).toHaveTextContent('Upload a document to start.');
  });

  function makeFile(name: string, sizeBytes = 10): File {
    const file = new File(['x'], name, { type: 'text/plain' });
    Object.defineProperty(file, 'size', { value: sizeBytes });
    return file;
  }

  function sseResponse(frames: string[]): Response {
    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const frame of frames) controller.enqueue(encoder.encode(frame));
        controller.close();
      },
    });
    return new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
  }

  function doneFrame(): string {
    return `data: ${JSON.stringify({
      type: 'done',
      answer: 'Answer.',
      sources: [],
      timings: { embed_ms: 1, search_ms: 1, rerank_ms: 1, generate_ms: 1, total_ms: 4 },
    })}\n\n`;
  }

  function stubUploadAndQuestionFetch(answerFrames: string[] = [doneFrame()]): ReturnType<typeof vi.fn> {
    let jobCounter = 0;
    const jobFilenames = new Map<string, string>();
    const indexed = new Set<string>();
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
      if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
      if (url.endsWith('/documents') && init?.method === 'POST') {
        const form = init.body as FormData;
        const file = form.get('file') as File;
        jobCounter += 1;
        const jobId = `job-${jobCounter}`;
        jobFilenames.set(jobId, file.name);
        return jsonResponse({ job_id: jobId, filename: file.name, state: 'queued' });
      }
      if (url.includes('/documents/jobs/job-')) {
        const jobId = url.split('/').pop() as string;
        const filename = jobFilenames.get(jobId) as string;
        indexed.add(filename);
        return jsonResponse({
          job_id: jobId,
          filename,
          state: 'done',
          chunks_total: 1,
          chunks_done: 1,
          error: null,
          result: { filename, sections_parsed: 1, chunks_ingested: 1, collection_name: 'docrag_documents' },
        });
      }
      if (url.endsWith('/documents') && (!init?.method || init.method === 'GET')) {
        const chunk_counts = Object.fromEntries([...indexed].map((f) => [f, 1]));
        return jsonResponse({ filenames: [...indexed], chunk_counts });
      }
      if (url.endsWith('/questions/stream')) {
        return sseResponse(answerFrames);
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    return fetchMock;
  }

  it('uploading files enables the question input and adds them to the search scope', async () => {
    stubUploadAndQuestionFetch();

    render(<App />);
    await screen.findByText('api ok');
    await openCorpusPanel();

    const input = screen.getByTestId('upload-input') as HTMLInputElement;
    await userEvent.upload(input, [makeFile('a.txt'), makeFile('b.txt')]);
    await userEvent.click(screen.getByTestId('upload-button'));

    await vi.waitFor(() => {
      expect(screen.getAllByTestId('corpus-panel-item')).toHaveLength(2);
    });
    expect(screen.getByTestId('question-textarea')).toBeEnabled();
    expect(screen.queryByTestId('question-hint')).not.toBeInTheDocument();

    await userEvent.click(screen.getByTestId('composer-scope-button'));
    expect(screen.getAllByTestId('scope-item')).toHaveLength(2);
  });

  it('asking with "All documents" selected searches the whole corpus (no filenames)', async () => {
    const fetchMock = stubUploadAndQuestionFetch();

    render(<App />);
    await screen.findByText('api ok');
    await openCorpusPanel();

    await userEvent.upload(screen.getByTestId('upload-input'), [makeFile('a.txt'), makeFile('b.txt')]);
    await userEvent.click(screen.getByTestId('upload-button'));
    await vi.waitFor(() => {
      expect(screen.getAllByTestId('corpus-panel-item')).toHaveLength(2);
    });

    await userEvent.type(screen.getByTestId('question-textarea'), 'What is this about?');
    await userEvent.click(screen.getByTestId('question-submit'));

    await vi.waitFor(() => {
      expect(fetchMock.mock.calls.some(([input]) => String(input).endsWith('/questions/stream'))).toBe(true);
    });
    const streamCall = fetchMock.mock.calls.find(([input]) => String(input).endsWith('/questions/stream'));
    const body = JSON.parse(String(streamCall?.[1]?.body));
    // No document selected = search everything; the client omits `filenames` entirely.
    expect(body.filenames).toBeUndefined();
  });

  it('selecting one document in the scope popover restricts the filenames sent', async () => {
    const fetchMock = stubUploadAndQuestionFetch();

    render(<App />);
    await screen.findByText('api ok');
    await openCorpusPanel();

    await userEvent.upload(screen.getByTestId('upload-input'), [makeFile('a.txt'), makeFile('b.txt')]);
    await userEvent.click(screen.getByTestId('upload-button'));
    await vi.waitFor(() => {
      expect(screen.getAllByTestId('corpus-panel-item')).toHaveLength(2);
    });

    await userEvent.click(screen.getByTestId('composer-scope-button'));
    const scopeItems = screen.getAllByTestId('scope-item');
    const aItem = scopeItems.find((item) => item.textContent?.includes('a.txt'));
    await userEvent.click(aItem?.querySelector('input') as HTMLInputElement);

    await userEvent.type(screen.getByTestId('question-textarea'), 'What is this about?');
    await userEvent.click(screen.getByTestId('question-submit'));

    await vi.waitFor(() => {
      expect(fetchMock.mock.calls.some(([input]) => String(input).endsWith('/questions/stream'))).toBe(true);
    });
    const streamCall = fetchMock.mock.calls.find(([input]) => String(input).endsWith('/questions/stream'));
    const body = JSON.parse(String(streamCall?.[1]?.body));
    expect(body.filenames).toEqual(['a.txt']);
  });

  it('re-uploading a file with the same name does not duplicate it in the search scope', async () => {
    const fetchMock = stubUploadAndQuestionFetch();

    render(<App />);
    await screen.findByText('api ok');
    await openCorpusPanel();

    await userEvent.upload(screen.getByTestId('upload-input'), [makeFile('a.txt')]);
    await userEvent.click(screen.getByTestId('upload-button'));
    await vi.waitFor(() => {
      expect(screen.getAllByTestId('corpus-panel-item')).toHaveLength(1);
    });

    await userEvent.upload(screen.getByTestId('upload-input'), [makeFile('a.txt')]);
    await userEvent.click(screen.getByTestId('upload-button'));

    await vi.waitFor(() => {
      expect(fetchMock.mock.calls.some(([input]) => String(input).includes('/documents/jobs/job-2'))).toBe(
        true,
      );
    });
    expect(screen.getAllByTestId('corpus-panel-item')).toHaveLength(1);
  });

  it('asking a question renders the answer and, in engineer mode, a duration/model meta line', async () => {
    stubUploadAndQuestionFetch();

    render(<App />);
    await screen.findByText('api ok');
    await openCorpusPanel();
    await userEvent.upload(screen.getByTestId('upload-input'), [makeFile('a.txt')]);
    await userEvent.click(screen.getByTestId('upload-button'));
    await vi.waitFor(() => {
      expect(screen.getAllByTestId('corpus-panel-item')).toHaveLength(1);
    });

    await userEvent.click(screen.getByTestId('mode-engineer'));
    await userEvent.type(screen.getByTestId('question-textarea'), 'What is this about?');
    await userEvent.click(screen.getByTestId('question-submit'));

    expect(await screen.findByText('Answer.')).toBeInTheDocument();
    expect(await screen.findByText('0.00s')).toBeInTheDocument();
  });

  it('drops a hovered citation preview when the conversation changes under it', async () => {
    const source = {
      source_number: 1,
      filename: 'a.txt',
      page: 1,
      section: 'Intro',
      chunk_id: 'c1',
      text: 'Alpha passage.',
    };
    stubUploadAndQuestionFetch([
      `data: ${JSON.stringify({
        type: 'done',
        answer: 'Alpha [1].',
        sources: [source],
        timings: { embed_ms: 1, search_ms: 1, rerank_ms: 1, generate_ms: 1, total_ms: 4 },
      })}\n\n`,
    ]);
    render(<App />);
    await screen.findByText('api ok');
    await openCorpusPanel();
    await userEvent.upload(screen.getByTestId('upload-input'), [makeFile('a.txt')]);
    await userEvent.click(screen.getByTestId('upload-button'));
    await vi.waitFor(() => expect(screen.getAllByTestId('corpus-panel-item')).toHaveLength(1));
    await userEvent.type(screen.getByTestId('question-textarea'), 'What is alpha?');
    await userEvent.click(screen.getByTestId('question-submit'));

    const citation = (await screen.findAllByTestId('inline-citation'))[0];
    fireEvent.mouseEnter(citation);
    const preview = await screen.findByTestId('source-preview');
    // The citation is described by the preview it shows, for screen readers.
    expect(citation).toHaveAttribute('aria-describedby', preview.id);
    expect(preview).toHaveAttribute('role', 'tooltip');
    // Escape dismisses it without moving the pointer (WCAG 1.4.13).
    fireEvent.keyDown(window, { key: 'Escape' });
    await waitFor(() => expect(screen.queryByTestId('source-preview')).not.toBeInTheDocument());
    fireEvent.mouseLeave(citation);
    fireEvent.mouseEnter(citation);
    expect(await screen.findByTestId('source-preview')).toBeInTheDocument();

    // A new chat from something that never moves the pointer off the citation (as the
    // palette or a shortcut doesn't): the button unmounts with no mouseleave or blur.
    fireEvent.click(screen.getByTestId('rail-chat'));
    fireEvent.click(screen.getByTestId('context-panel-action'));

    await waitFor(() => expect(screen.queryByTestId('source-preview')).not.toBeInTheDocument());
  });

  it('opens another conversation while an answer streams, and holds sending there', async () => {
    let finishStream!: () => void;
    const fetchMock = stubUploadAndQuestionFetch();
    const base = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (!String(input).endsWith('/questions/stream')) return base(input, init);
      const encoder = new TextEncoder();
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          finishStream = () => {
            controller.enqueue(encoder.encode(doneFrame()));
            controller.close();
          };
        },
      });
      return new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
    });
    render(<App />);
    await screen.findByText('api ok');
    await openCorpusPanel();
    await userEvent.upload(screen.getByTestId('upload-input'), [makeFile('a.txt')]);
    await userEvent.click(screen.getByTestId('upload-button'));
    await vi.waitFor(() => expect(screen.getByTestId('question-textarea')).toBeEnabled());
    await userEvent.type(screen.getByTestId('question-textarea'), 'First question');
    await userEvent.click(screen.getByTestId('question-submit'));
    expect(await screen.findByRole('button', { name: 'Stop' })).toBeInTheDocument();

    await userEvent.click(screen.getByTestId('rail-chat'));
    await userEvent.click(screen.getByTestId('context-panel-action'));

    // A fresh conversation: typing works, sending waits, and there's no Stop for it.
    expect(screen.getByTestId('question-submit')).toHaveTextContent('Ask');
    expect(screen.getByTestId('question-hint')).toHaveTextContent('Still answering in “First question”');
    await userEvent.type(screen.getByTestId('question-textarea'), 'Next');
    expect(screen.getByTestId('question-submit')).toBeDisabled();

    await act(async () => finishStream());
    await vi.waitFor(() => expect(screen.getByTestId('question-submit')).toBeEnabled());
    // The answer landed in the conversation it was asked in.
    await userEvent.click(screen.getByRole('button', { name: /^First question/ }));
    expect(await screen.findByText('Answer.')).toBeInTheDocument();
  });

  it('clicking thumbs-up on an answer POSTs feedback with the full derived payload', async () => {
    let jobCounter = 0;
    const jobFilenames = new Map<string, string>();
    const indexed = new Set<string>();
    let feedbackRequestBody: unknown = null;
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload({ feedback_enabled: true }));
        if (url.endsWith('/documents') && init?.method === 'POST') {
          const form = init.body as FormData;
          const file = form.get('file') as File;
          jobCounter += 1;
          const jobId = `job-${jobCounter}`;
          jobFilenames.set(jobId, file.name);
          return jsonResponse({ job_id: jobId, filename: file.name, state: 'queued' });
        }
        if (url.includes('/documents/jobs/job-')) {
          const jobId = url.split('/').pop() as string;
          const filename = jobFilenames.get(jobId) as string;
          indexed.add(filename);
          return jsonResponse({
            job_id: jobId,
            filename,
            state: 'done',
            chunks_total: 1,
            chunks_done: 1,
            error: null,
            result: { filename, sections_parsed: 1, chunks_ingested: 1, collection_name: 'docrag_documents' },
          });
        }
        if (url.endsWith('/documents') && (!init?.method || init.method === 'GET')) {
          const chunk_counts = Object.fromEntries([...indexed].map((f) => [f, 1]));
          return jsonResponse({ filenames: [...indexed], chunk_counts });
        }
        if (url.endsWith('/questions/stream')) {
          const frame = `data: ${JSON.stringify({
            type: 'done',
            answer: 'The answer is 42 [1].',
            sources: [
              { source_number: 1, filename: 'a.txt', page: 1, section: 'Body', chunk_id: 'c1', text: 'x' },
            ],
            timings: { embed_ms: 1, search_ms: 1, rerank_ms: 1, generate_ms: 1, total_ms: 4 },
            trace_id: 'trace-abc',
          })}\n\n`;
          return sseResponse([frame]);
        }
        if (url.endsWith('/feedback') && init?.method === 'POST') {
          feedbackRequestBody = JSON.parse(init.body as string);
          return jsonResponse({ id: 'feedback-1' });
        }
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);
    await screen.findByText('api ok');
    await openCorpusPanel();
    await userEvent.upload(screen.getByTestId('upload-input'), [makeFile('a.txt')]);
    await userEvent.click(screen.getByTestId('upload-button'));
    await vi.waitFor(() => {
      expect(screen.getAllByTestId('corpus-panel-item')).toHaveLength(1);
    });

    await userEvent.type(screen.getByTestId('question-textarea'), 'What is the answer?');
    await userEvent.click(screen.getByTestId('question-submit'));
    await screen.findByText(/The answer is 42/);

    await userEvent.click(screen.getByTestId('chat-message-feedback-up'));

    await vi.waitFor(() => expect(feedbackRequestBody).not.toBeNull());
    expect(feedbackRequestBody).toEqual({
      trace_id: 'trace-abc',
      question: 'What is the answer?',
      answer_excerpt: 'The answer is 42 [1].',
      cited_filenames: ['a.txt'],
      rating: 'up',
      citation_source_number: null,
      kind: 'question',
    });
  });

  it('switching rail panels shows the corresponding contextual panel', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) return jsonResponse(makeConfigPayload());
        if (url.endsWith('/documents')) return jsonResponse({ filenames: [], chunk_counts: {} });
        if (url.endsWith('/traces')) return jsonResponse({ traces: [] });
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);
    await screen.findByText('api ok');

    expect(screen.getByTestId('rail-chat')).toHaveAttribute('aria-pressed', 'true');

    await userEvent.click(screen.getByTestId('rail-traces'));
    expect(screen.getByTestId('rail-traces')).toHaveAttribute('aria-pressed', 'true');
    expect(await screen.findByText('No traces yet.')).toBeInTheDocument();

    await openCorpusPanel();
    expect(screen.getByTestId('upload-dropzone')).toBeInTheDocument();
  });
});

describe('App workspaces', () => {
  const turn = (id: string, role: 'user' | 'assistant', content: string) => ({
    id, role, content, sources: [], timestamp: 1_700_000_000_000, timings: null, traceId: null,
  });

  function stubServer() {
    const listedFor: (string | undefined)[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input), 'http://localhost');
      const headers = (init?.headers ?? {}) as Record<string, string>;
      if (url.pathname === '/health') return jsonResponse({ status: 'ok' });
      if (url.pathname === '/config') return jsonResponse(makeConfigPayload({ workspaces_enabled: true }));
      if (url.pathname === '/access') return jsonResponse({ access: 'open' });
      if (url.pathname === '/workspaces') return jsonResponse({ enabled: true, workspaces: ['legal'] });
      if (url.pathname === '/documents') {
        listedFor.push(headers['X-Workspace']);
        const filenames = headers['X-Workspace'] === 'legal' ? ['contract.txt'] : ['guide.md'];
        return jsonResponse({ filenames, chunk_counts: Object.fromEntries(filenames.map((f) => [f, 1])) });
      }
      throw new Error(`unexpected fetch: ${url.pathname}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    return listedFor;
  }

  afterEach(async () => {
    const { setWorkspace } = await import('../src/api/client');
    setWorkspace('');
  });

  it('opens a fresh chat when the workspace changes, and returns to a conversation’s own workspace', async () => {
    localStorage.setItem(
      'docrag-chat-history',
      JSON.stringify({
        version: 2,
        activeConversationId: 'default-chat',
        conversations: [
          { id: 'default-chat', title: 'Guide question', createdAt: 2, updatedAt: 2,
            turns: [turn('d1', 'user', 'What is in the guide?'), turn('d2', 'assistant', 'Hybrid retrieval.')] },
          { id: 'legal-chat', title: 'Contract question', createdAt: 1, updatedAt: 1, workspace: 'legal',
            turns: [turn('l1', 'user', 'What does the contract say?'), turn('l2', 'assistant', 'Kappa.')] },
        ],
      }),
    );
    const listedFor = stubServer();
    render(<App />);
    expect(await screen.findByText('What is in the guide?')).toBeInTheDocument();

    await openCorpusPanel();
    const select = await screen.findByTestId('workspace-select');
    await waitFor(() => expect(screen.getByRole('option', { name: 'legal' })).toBeInTheDocument());
    await userEvent.selectOptions(select, 'legal');

    // The guide conversation stays in the default workspace: a new chat opens here.
    await waitFor(() => expect(screen.queryByText('What is in the guide?')).not.toBeInTheDocument());
    await waitFor(() => expect(listedFor.at(-1)).toBe('legal'));

    // Opening the guide conversation brings the default workspace back.
    await userEvent.click(screen.getByTestId('rail-chat'));
    const row = screen.getAllByTestId('conversation-select').find((r) => r.textContent?.includes('Guide question'))!;
    await userEvent.click(row);
    expect(await screen.findByText('What is in the guide?')).toBeInTheDocument();
    await waitFor(() => expect(listedFor.at(-1)).toBeUndefined());

    // The legal one is labelled, and opening it switches to legal.
    const legalRow = screen.getAllByTestId('conversation-select').find((r) => r.textContent?.includes('Contract question'))!;
    expect(legalRow).toHaveTextContent('legal');
    await userEvent.click(legalRow);
    expect(await screen.findByText('What does the contract say?')).toBeInTheDocument();
    await waitFor(() => expect(listedFor.at(-1)).toBe('legal'));
    await openCorpusPanel();
    expect(screen.getByTestId('workspace-select')).toHaveValue('legal');
  });

  it('says so when a conversation’s workspace was deleted, instead of switching', async () => {
    localStorage.setItem(
      'docrag-chat-history',
      JSON.stringify({
        version: 2,
        activeConversationId: 'gone-chat',
        conversations: [
          { id: 'gone-chat', title: 'Old', createdAt: 1, updatedAt: 1, workspace: 'archive',
            turns: [turn('g1', 'user', 'Old question'), turn('g2', 'assistant', 'Old answer.')] },
        ],
      }),
    );
    const listedFor = stubServer();
    render(<App />);
    expect(await screen.findByText('Workspace no longer exists')).toBeInTheDocument();
    expect(listedFor.every((header) => header === undefined)).toBe(true);
  });
});

describe('App workspace switching', () => {
  type Reply = Response | Promise<Response>;
  const listing = (filenames: string[]) =>
    jsonResponse({ filenames, chunk_counts: Object.fromEntries(filenames.map((f) => [f, 1])) });
  const CORPUS: Record<string, string[]> = { '': ['guide.md', 'shared.md'], legal: ['contract.txt', 'shared.md'] };

  function server(
    routes: {
      config?: () => Reply;
      workspaces?: () => Reply;
      documents?: (workspace: string) => Reply;
    } = {},
  ) {
    const calls: { path: string; workspace?: string }[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = new URL(String(input), 'http://localhost');
        const workspace = ((init?.headers ?? {}) as Record<string, string>)['X-Workspace'];
        calls.push({ path: url.pathname, workspace });
        if (url.pathname === '/health') return jsonResponse({ status: 'ok' });
        if (url.pathname === '/config') {
          return routes.config?.() ?? jsonResponse(makeConfigPayload({ workspaces_enabled: true }));
        }
        if (url.pathname === '/access') return jsonResponse({ access: 'open' });
        if (url.pathname === '/workspaces') {
          return routes.workspaces?.() ?? jsonResponse({ enabled: true, workspaces: ['legal'] });
        }
        if (url.pathname === '/documents') {
          return routes.documents?.(workspace ?? '') ?? listing(CORPUS[workspace ?? '']);
        }
        if (url.pathname === '/search') {
          return jsonResponse({
            results: [
              { rank: 1, filename: 'guide.md', page: 1, section: 'Intro', chunk_id: 'c1', chunk_ordinal: 0,
                text: 'Hybrid retrieval.', retrieval_score: 0.5, rerank_score: 0.9 },
            ],
            timings: { embed_ms: 1, search_ms: 1, rerank_ms: 1 },
          });
        }
        throw new Error(`unexpected fetch: ${url.pathname}`);
      }),
    );
    return calls;
  }

  afterEach(async () => {
    const { setWorkspace } = await import('../src/api/client');
    setWorkspace('');
  });

  const card = (name: string) =>
    screen.queryAllByTestId('corpus-panel-item').find((item) => item.dataset.corpusCard === name);

  it('leaves nothing of the previous workspace’s corpus to act on while the next one loads', async () => {
    let answerLegal!: () => void;
    server({
      documents: (workspace) =>
        workspace === 'legal'
          ? new Promise((resolve) => (answerLegal = () => resolve(listing(CORPUS.legal))))
          : listing(CORPUS['']),
    });
    render(<App />);
    await screen.findByText('api ok');
    await openCorpusPanel();
    await waitFor(() => expect(card('guide.md')).toBeDefined());
    await userEvent.click(screen.getByLabelText('Select shared.md'));
    expect(screen.getByTestId('corpus-bulk')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole('option', { name: 'legal' })).toBeInTheDocument());

    await userEvent.selectOptions(screen.getByTestId('workspace-select'), 'legal');

    // Not the default workspace's cards, nor its selection ("1 selected" over a
    // shared.md that exists in both), and not "Upload a document to start." either.
    expect(card('guide.md')).toBeUndefined();
    expect(card('shared.md')).toBeUndefined();
    expect(screen.queryByTestId('corpus-bulk')).not.toBeInTheDocument();
    expect(screen.getByTestId('question-hint')).not.toHaveTextContent('Upload a document to start.');

    await act(async () => answerLegal());
    await waitFor(() => expect(card('contract.txt')).toBeDefined());
    expect(card('shared.md')).toBeDefined();
    expect(screen.queryByTestId('corpus-bulk')).not.toBeInTheDocument();
  });

  it('retries a listing that failed after a switch instead of showing an empty corpus', async () => {
    server({
      documents: (workspace) =>
        workspace === 'legal'
          ? new Response(JSON.stringify({ detail: 'Qdrant is down.' }), { status: 503 })
          : listing(CORPUS['']),
    });
    render(<App />);
    await screen.findByText('api ok');
    await openCorpusPanel();
    await waitFor(() => expect(screen.getByRole('option', { name: 'legal' })).toBeInTheDocument());

    await userEvent.selectOptions(screen.getByTestId('workspace-select'), 'legal');

    await waitFor(() =>
      expect(screen.getByTestId('question-hint')).toHaveTextContent('Couldn’t load the document list — retrying.'),
    );
  });

  it('clears search results of the previous workspace', async () => {
    server();
    render(<App />);
    await screen.findByText('api ok');
    await userEvent.click(screen.getByTestId('rail-search'));
    await userEvent.type(screen.getByTestId('search-input'), 'retrieval{Enter}');
    expect(await screen.findByTestId('search-result')).toBeInTheDocument();

    await openCorpusPanel();
    await waitFor(() => expect(screen.getByRole('option', { name: 'legal' })).toBeInTheDocument());
    await userEvent.selectOptions(screen.getByTestId('workspace-select'), 'legal');
    await userEvent.click(screen.getByTestId('rail-search'));

    expect(screen.queryByTestId('search-result')).not.toBeInTheDocument();
    expect(screen.getByTestId('search-input')).toHaveValue('');
  });

  it('asks for /config and /access without waiting on the remembered workspace’s check', async () => {
    const { setWorkspace } = await import('../src/api/client');
    setWorkspace('legal');
    const calls = server({ workspaces: () => new Promise<Response>(() => {}) });
    render(<App />);

    expect(await screen.findByText('api ok')).toBeInTheDocument();
    expect(calls.find((call) => call.path === '/access')?.workspace).toBeUndefined();
    // The listing still waits: it may be read from a workspace that is gone.
    expect(calls.some((call) => call.path === '/documents')).toBe(false);
  });

  it('falls back to the default workspace when the one in use was deleted elsewhere', async () => {
    const { setWorkspace } = await import('../src/api/client');
    setWorkspace('legal');
    const calls = server({
      documents: (workspace) =>
        workspace === 'legal'
          ? new Response(JSON.stringify({ detail: "No workspace named 'legal'." }), { status: 404 })
          : listing(CORPUS['']),
    });
    render(<App />);

    expect(await screen.findByText('Workspace no longer exists')).toBeInTheDocument();
    await waitFor(() => expect(calls.filter((c) => c.path === '/documents').at(-1)?.workspace).toBeUndefined());
    await openCorpusPanel();
    expect(screen.getByTestId('workspace-select')).toHaveValue('');
    await waitFor(() => expect(card('guide.md')).toBeDefined());
  });

  it('reads the workspace list again after Settings are saved', async () => {
    const calls = server();
    render(<App />);
    await screen.findByText('api ok');
    await waitFor(() => expect(calls.filter((c) => c.path === '/workspaces')).toHaveLength(1));

    await userEvent.click(screen.getByTestId('open-settings'));
    await userEvent.click(await screen.findByTestId('settings-save'));

    await waitFor(() => expect(calls.filter((c) => c.path === '/workspaces')).toHaveLength(2));
  });

  it('keeps Summarize disabled until the app has booted', async () => {
    server({ config: () => new Promise<Response>(() => {}) });
    render(<App />);
    await openCorpusPanel();
    await waitFor(() => expect(card('guide.md')).toBeDefined());
    expect(screen.getByLabelText('Summarize guide.md')).toBeDisabled();
  });
});
