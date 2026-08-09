import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from '../src/App';

afterEach(() => {
  vi.unstubAllGlobals();
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

  it('shows the unreachable banner when health check fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));

    render(<App />);

    expect(await screen.findByText('api unreachable')).toBeInTheDocument();
    expect(screen.getByText(/Cannot reach the DocRAG API/)).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent(/Cannot reach the DocRAG API/);
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

  function stubUploadAndQuestionFetch(): ReturnType<typeof vi.fn> {
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
        return sseResponse([doneFrame()]);
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
