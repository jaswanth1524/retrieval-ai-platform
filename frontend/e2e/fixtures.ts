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

/** A GET /traces row, as the trace browser lists it. */
export function traceSummary(traceId = 'trace-1', question = 'What is DocRAG?') {
  return {
    trace_id: traceId,
    created_at: Date.now() / 1000,
    question,
    mode: 'stream',
    status: 'ok',
    llm_provider: 'ollama',
    total_ms: 460,
    candidate_count: 12,
    kept_count: 6,
  };
}

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
  raw_documents_enabled: false,
};

type JobStep = { status: number; body?: object };

/**
 * In-memory stand-in for the DocRAG API, installed with page.route before the app
 * loads. Tests adjust the public fields (or the handlers) to shape each scenario.
 */
export class MockApi {
  documents: string[] = ['guide.md'];
  /** Rows for GET /traces (the trace browser). */
  traces: object[] = [];
  /** Merged over the default /config body (e.g. raw_documents_enabled). */
  config: Partial<typeof CONFIG> = {};
  /** When set, every guarded route answers 401 unless X-API-Key matches. */
  apiKey: string | null = null;
  /** Indexed by an older chunker / have a stored original (GET /documents). */
  staleFilenames: string[] = [];
  reindexableFilenames: string[] = [];
  /** Each tagged document's tags (GET /documents, PATCH /documents/{f}/tags). */
  tags: Record<string, string[]> = {};
  /** Chunks for GET /documents/{f}/content; null serves the default two (incl. GUIDE_SOURCE). */
  contentChunks: object[] | null = null;
  /** Every POST /questions/stream body the app sent, oldest first. */
  readonly questionBodies: Record<string, unknown>[] = [];
  /** Body for POST /questions/stream, or a function for held/custom streams. */
  stream: string | ((route: Route) => Promise<void>) = answerStream();
  /** Response for POST /documents; default accepts the upload as job "job-1". */
  uploadResponse: JobStep | null = null;
  /** How many uploads to turn away first with 503 + Retry-After (a full indexing queue). */
  uploadsQueueFull = 0;
  /** Successive GET /documents/jobs/{id} responses; the last one repeats. */
  jobSteps: JobStep[] = [];
  /** What GET /access reports the browser's key unlocks. */
  access: 'full' | 'read' | 'open' = 'open';
  /** GET /feedback rows (newest first). */
  feedbackItems: object[] = [];
  /** Filenames a POST /import restores (each becomes a job). */
  importFilenames: string[] = [];
  /** Bodies of every POST /import, for a spec to inspect. */
  readonly imports: string[] = [];
  /** In-flight ingest jobs by id: the file each one indexes and how often it was polled.
   *  Every job walks `jobSteps` on its own, so a bulk re-index runs several at once. */
  private readonly jobs = new Map<string, { filename: string; polls: number }>();
  /** URLs a test deliberately answered with a 4xx/5xx. */
  readonly intentionalFailures = new Set<string>();
  /** API requests nothing here handles — a spec or app regression, never expected. */
  readonly unmocked: string[] = [];

  async install(page: Page): Promise<void> {
    await page.route(
      (url) =>
        /^\/(health|config|access|documents|questions|traces|feedback|import|export|search)(\/|$)/.test(
          url.pathname,
        ),
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
    if (path === '/config') return this.json(route, 200, { ...CONFIG, ...this.config });
    if (this.apiKey !== null && request.headers()['x-api-key'] !== this.apiKey) {
      return this.configured(route, 401, { detail: 'Missing or invalid API key.' });
    }
    if (path === '/access') return this.json(route, 200, { access: this.access });
    if (path === '/traces') return this.json(route, 200, { traces: this.traces });
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
      this.questionBodies.push(JSON.parse(request.postData() ?? '{}') as Record<string, unknown>);
      if (typeof this.stream === 'function') return this.stream(route);
      return route.fulfill({ status: 200, contentType: 'text/event-stream', body: this.stream });
    }

    if (path === '/documents' && method === 'GET') {
      const counts = Object.fromEntries(this.documents.map((name) => [name, 1]));
      return this.json(route, 200, {
        filenames: this.documents,
        chunk_counts: counts,
        stale_filenames: this.staleFilenames.filter((name) => this.documents.includes(name)),
        reindexable_filenames: this.reindexableFilenames.filter((name) =>
          this.documents.includes(name),
        ),
        tags: this.tags,
      });
    }

    if (path === '/documents' && method === 'POST') {
      const filename = /filename="([^"]+)"/.exec(request.postData() ?? '')?.[1] ?? 'upload.txt';
      if (this.uploadResponse) {
        const { status, body } = this.uploadResponse;
        return this.configured(route, status, body ?? {});
      }
      if (this.uploadsQueueFull > 0) {
        this.uploadsQueueFull -= 1;
        this.intentionalFailures.add(request.url());
        return route.fulfill({
          status: 503,
          contentType: 'application/json',
          headers: { 'Retry-After': '1' },
          body: JSON.stringify({ detail: 'Too many documents are already waiting to be indexed.' }),
        });
      }
      return this.json(route, 202, this.startJob(filename));
    }

    if (path.startsWith('/documents/jobs/')) {
      const jobId = path.slice('/documents/jobs/'.length);
      const job = this.jobs.get(jobId);
      if (!job) return this.configured(route, 404, { detail: `Unknown job '${jobId}'.` });
      const step = this.jobSteps[Math.min(job.polls, this.jobSteps.length - 1)] ?? {
        status: 200,
        body: this.jobBody('done'),
      };
      job.polls += 1;
      const body: Record<string, unknown> = { ...step.body, job_id: jobId, filename: job.filename };
      if (body.state === 'done') {
        if (!this.documents.includes(job.filename)) this.documents = [...this.documents, job.filename];
        this.staleFilenames = this.staleFilenames.filter((name) => name !== job.filename);
      }
      return this.configured(route, step.status, body);
    }

    if (path === '/documents/reindex' && method === 'POST') {
      const due = this.staleFilenames.filter(
        (name) => this.documents.includes(name) && this.reindexableFilenames.includes(name),
      );
      return this.json(route, 202, { jobs: due.map((name) => this.startJob(name)), deferred: [] });
    }

    const documentRoute = /^\/documents\/([^/]+)(?:\/(content|reindex|tags|original))?$/.exec(path);
    if (documentRoute) {
      const filename = decodeURIComponent(documentRoute[1]);
      const action = documentRoute[2];
      if (!this.documents.includes(filename)) {
        return this.configured(route, 404, { detail: `No indexed document named '${filename}'.` });
      }
      if (!action && method === 'DELETE') {
        this.documents = this.documents.filter((name) => name !== filename);
        return this.json(route, 200, { filename, points_deleted: 1 });
      }
      if (action === 'content' && method === 'GET') {
        const all = (this.contentChunks ?? [
          { chunk_id: 'c0', page: 1, section: 'Intro', text: 'DocRAG overview.', chunk_ordinal: 1 },
          { chunk_id: GUIDE_SOURCE.chunk_id, page: 1, section: GUIDE_SOURCE.section, text: GUIDE_SOURCE.text, chunk_ordinal: 2 },
        ]) as { chunk_id: string; chunk_ordinal: number }[];
        // Pages like the server: `around` + `radius`, or `start`/`end` ordinals.
        const params = new URL(request.url()).searchParams;
        const around = params.get('around');
        if (around === null && !params.has('start') && !params.has('end')) {
          return this.json(route, 200, { filename, chunks: all, total_chunks: all.length });
        }
        let first: number;
        let last: number;
        let targetFound: boolean | null = null;
        if (around !== null) {
          const hit = all.find((chunk) => chunk.chunk_id === around);
          targetFound = hit !== undefined;
          const radius = Number(params.get('radius') ?? 30);
          first = Math.max(1, (hit?.chunk_ordinal ?? 1) - radius);
          last = (hit?.chunk_ordinal ?? 1) + radius;
        } else {
          first = Number(params.get('start') ?? 1);
          last = Number(params.get('end') ?? first + 200);
        }
        return this.json(route, 200, {
          filename,
          chunks: all.filter((chunk) => chunk.chunk_ordinal >= first && chunk.chunk_ordinal <= last),
          total_chunks: all.length,
          target_found: targetFound,
        });
      }
      if (action === 'tags' && method === 'PATCH') {
        const { tags } = JSON.parse(request.postData() ?? '{}') as { tags: string[] };
        if (tags.length > 0) this.tags[filename] = tags;
        else delete this.tags[filename];
        return this.json(route, 200, { filename, tags });
      }
      if (action === 'reindex' && method === 'POST') {
        return this.json(route, 202, this.startJob(filename));
      }
      if (action === 'original' && method === 'GET') {
        return route.fulfill({ status: 200, contentType: 'text/markdown', body: `# ${filename}` });
      }
    }

    if (path === '/feedback' && method === 'POST') {
      return this.json(route, 201, { id: 1 });
    }
    if (path === '/feedback' && method === 'GET') {
      return this.json(route, 200, { feedback: this.feedbackItems });
    }
    if (path === '/import' && method === 'POST') {
      this.imports.push(request.postData() ?? '');
      return this.json(route, 202, {
        jobs: this.importFilenames.map((name) => this.startJob(name)),
        deferred: [],
        skipped: [],
        missing_originals: [],
        rejected: [],
        feedback_imported: 2,
        feedback_skipped: 0,
      });
    }

    this.unmocked.push(`${method} ${path}`);
    return this.json(route, 404, { detail: `Unmocked ${method} ${path}` });
  }

  private startJob(filename: string) {
    const jobId = `job-${this.jobs.size + 1}`;
    this.jobs.set(jobId, { filename, polls: 0 });
    return { job_id: jobId, filename, state: 'queued' as const };
  }

  /** A job-status body; the handler fills in the polled job's own id and filename. */
  jobBody(state: 'queued' | 'parsing' | 'embedding' | 'done' | 'failed', error: string | null = null) {
    return {
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
