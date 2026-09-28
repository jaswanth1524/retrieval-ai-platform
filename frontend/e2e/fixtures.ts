import { test as base, expect, type Page, type Route } from '@playwright/test';

/** Frames for POST /questions/stream, in the server's SSE wire format. */
export function sse(...events: object[]): string {
  return events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join('');
}

export const TIMINGS = {
  embed_ms: 5,
  search_ms: 5,
  rerank_ms: 50,
  generate_ms: 400,
  total_ms: 460,
  condense_ms: 0,
  query_expansion_ms: 0,
  context_expansion_ms: 2,
  citation_retry_ms: 0,
};

export const GUIDE_SOURCE = {
  source_number: 1,
  filename: 'guide.md',
  page: 1,
  section: 'Retrieval',
  chunk_id: 'c1',
  text: 'DocRAG fuses dense and BM25 results with Reciprocal Rank Fusion.',
};

/** A complete, successful answer stream citing GUIDE_SOURCE. */
export function answerStream(answer = 'DocRAG uses hybrid retrieval [1].'): string {
  return sse(
    { type: 'stage', stage: 'searching' },
    { type: 'sources', sources: [GUIDE_SOURCE], trace_id: 'trace-1' },
    { type: 'delta', text: answer.slice(0, 10) },
    { type: 'delta', text: answer.slice(10) },
    { type: 'done', answer, sources: [GUIDE_SOURCE], timings: TIMINGS, trace_id: 'trace-1' },
  );
}

const CONFIG = {
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
  rerank_min_score: 0.15,
  max_upload_bytes: 52_428_800,
  rerank_top_k_limit: 50,
  max_context_chunks_limit: 20,
  llm_temperature_max: 2,
  feedback_enabled: false,
};

type JobStep = { status: number; body?: object };

/**
 * In-memory stand-in for the DocRAG API, installed with page.route before the app
 * loads. Tests adjust the public fields (or the handlers) to shape each scenario.
 */
export class MockApi {
  documents: string[] = ['guide.md'];
  /** Body for POST /questions/stream, or a function for held/custom streams. */
  stream: string | ((route: Route) => Promise<void>) = answerStream();
  /** Response for POST /documents; default accepts the upload as job "job-1". */
  uploadResponse: JobStep | null = null;
  /** Successive GET /documents/jobs/{id} responses; the last one repeats. */
  jobSteps: JobStep[] = [];
  private jobPolls = 0;
  /** URLs a test deliberately answered with a 4xx/5xx. */
  readonly intentionalFailures = new Set<string>();
  /** API requests nothing here handles — a spec or app regression, never expected. */
  readonly unmocked: string[] = [];

  async install(page: Page): Promise<void> {
    await page.route(
      (url) => /^\/(health|config|documents|questions|traces|feedback)(\/|$)/.test(url.pathname),
      (route) => this.handle(route),
    );
  }

  private json(route: Route, status: number, body: object): Promise<void> {
    return route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
  }

  /** Serve a response a test configured; an error status there is the scenario. */
  private configured(route: Route, status: number, body: object): Promise<void> {
    if (status >= 400) this.intentionalFailures.add(route.request().url());
    return this.json(route, status, body);
  }

  private async handle(route: Route): Promise<void> {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const method = request.method();

    if (path === '/health') return this.json(route, 200, { status: 'ok' });
    if (path === '/config') return this.json(route, 200, CONFIG);
    if (path === '/traces') return this.json(route, 200, { traces: [] });
    if (path.startsWith('/traces/')) {
      return this.json(route, 200, {
        trace_id: path.slice('/traces/'.length),
        created_at: 0,
        question: '',
        mode: 'stream',
        status: 'ok',
        config: null,
        candidates: [],
        prompt_messages: null,
        answer: null,
        cited_source_numbers: [],
        timings: null,
        error: null,
      });
    }

    if (path === '/questions/stream' && method === 'POST') {
      if (typeof this.stream === 'function') return this.stream(route);
      return route.fulfill({ status: 200, contentType: 'text/event-stream', body: this.stream });
    }

    if (path === '/documents' && method === 'GET') {
      const counts = Object.fromEntries(this.documents.map((name) => [name, 1]));
      return this.json(route, 200, { filenames: this.documents, chunk_counts: counts });
    }

    if (path === '/documents' && method === 'POST') {
      const filename = /filename="([^"]+)"/.exec(request.postData() ?? '')?.[1] ?? 'upload.txt';
      const response = this.uploadResponse ?? {
        status: 202,
        body: { job_id: 'job-1', filename, state: 'queued' },
      };
      if (response.status === 202) this.pendingFilename = filename;
      return this.configured(route, response.status, response.body ?? {});
    }

    if (path.startsWith('/documents/jobs/')) {
      const step = this.jobSteps[Math.min(this.jobPolls, this.jobSteps.length - 1)] ?? {
        status: 200,
        body: this.jobBody('done'),
      };
      this.jobPolls += 1;
      const body = step.body ?? {};
      if ((body as { state?: string }).state === 'done' && this.pendingFilename) {
        this.documents = [...this.documents, this.pendingFilename];
        this.pendingFilename = null;
      }
      return this.configured(route, step.status, body);
    }

    this.unmocked.push(`${method} ${path}`);
    return this.json(route, 404, { detail: `Unmocked ${method} ${path}` });
  }

  private pendingFilename: string | null = null;

  jobBody(state: 'queued' | 'parsing' | 'embedding' | 'done' | 'failed', error: string | null = null) {
    return {
      job_id: 'job-1',
      filename: this.pendingFilename ?? 'upload.txt',
      state,
      chunks_total: 4,
      chunks_done: state === 'done' ? 4 : 1,
      error,
      result: null,
    };
  }
}

type Fixtures = { api: MockApi; consoleErrors: string[] };

export const test = base.extend<Fixtures>({
  // Every spec fails on a console error or uncaught exception, not just on its own
  // assertions — a green run should mean the page stayed clean the whole time.
  consoleErrors: [
    async ({ page, api }, provide) => {
      const errors: string[] = [];
      page.on('console', (message) => {
        if (message.type() !== 'error') return;
        // The browser logs its own line for every 4xx/5xx. Only the responses a test
        // configured as failures are exempt — an unexpected 401/404 still fails.
        const resourceStatus = /^Failed to load resource: the server responded with a status of \d+/;
        if (resourceStatus.test(message.text()) && api.intentionalFailures.has(message.location().url)) {
          return;
        }
        errors.push(message.text());
      });
      page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
      await provide(errors);
      expect(errors, 'console errors during the test').toEqual([]);
      expect(api.unmocked, 'API requests the mock does not handle').toEqual([]);
    },
    { auto: true },
  ],
  api: async ({ page }, provide) => {
    const api = new MockApi();
    await api.install(page);
    await provide(api);
  },
});

export { expect };
