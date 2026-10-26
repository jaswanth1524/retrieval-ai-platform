import { answerStream, expect, test, GUIDE_SOURCE, sse, TIMINGS } from '../fixtures';

test.beforeEach(async ({ api }) => {
  api.documents = ['guide.md', 'notes.md'];
});

async function ask(page: import('@playwright/test').Page, question: string) {
  await page.getByTestId('question-textarea').fill(question);
  await page.getByTestId('question-submit').click();
  await expect(page.getByTestId('question-submit')).toHaveText('Ask');
}

test('an answer is asked again without one of its sources', async ({ page, api }) => {
  const scored = { ...GUIDE_SOURCE, rerank_score: 0.91 };
  api.stream = sse(
    { type: 'sources', sources: [scored], trace_id: 'trace-1' },
    { type: 'delta', text: 'DocRAG uses hybrid retrieval [1].' },
    {
      type: 'done',
      answer: 'DocRAG uses hybrid retrieval [1].',
      sources: [scored],
      timings: TIMINGS,
      trace_id: 'trace-1',
    },
  );
  await page.goto('/');
  await ask(page, 'How does retrieval work?');
  await expect(page.getByTestId('citation-score').first()).toHaveText(/0\.91/);

  await page.getByRole('button', { name: 'Ask again without source 1' }).first().click();
  await expect.poll(() => api.questionBodies.length).toBe(2);
  expect(api.questionBodies[1]).toMatchObject({
    question: 'How does retrieval work?',
    exclude_chunk_ids: [GUIDE_SOURCE.chunk_id],
    use_cache: false,
  });
});

test('a saved answer survives a reload and is listed under Saved', async ({ page, api }) => {
  api.stream = answerStream();
  await page.goto('/');
  await ask(page, 'What is DocRAG?');
  await page.getByTestId('chat-message-save').click();
  await expect(page.getByTestId('chat-message-save')).toHaveText('★ Saved');

  await page.reload();
  await expect(page.getByTestId('chat-message-save')).toHaveText('★ Saved');
  await page.getByTestId('rail-chat').click();
  await page.getByTestId('conversation-filter-saved').click();
  await expect(page.getByTestId('conversation-item')).toHaveCount(1);
  await expect(page.getByTestId('conversation-saved-count')).toContainText('1');
});

test('j and k move between answers', async ({ page, api }) => {
  api.stream = answerStream();
  await page.goto('/');
  await ask(page, 'First question?');
  await ask(page, 'Second question?');
  await page.getByTestId('chat-thread').click({ position: { x: 5, y: 5 } });

  await page.keyboard.press('k');
  const answers = page.locator('[data-testid="chat-message"][data-role="assistant"]');
  await expect(answers.last()).toBeFocused();
  await page.keyboard.press('k');
  await expect(answers.first()).toBeFocused();
  await page.keyboard.press('j');
  await expect(answers.last()).toBeFocused();
  await expect(answers.last()).toHaveAttribute('aria-label', 'Answer 2');
});

test('the asked questions panel shows the answer rate and the most asked', async ({ page, api }) => {
  api.config = { query_log_enabled: true };
  const now = Date.now() / 1000;
  const row = (question: string, status: string) => ({
    created_at: now,
    question,
    status,
    top_score: 0.4,
    mode: 'stream',
    trace_id: null,
  });
  api.unanswered = [
    row('Who approves travel?', 'insufficient_context'),
    row('How many leave days?', 'ok'),
    row('how many leave days?', 'ok'),
  ];
  await page.goto('/');
  await page.getByTestId('rail-unanswered').click();
  await expect(page.getByTestId('questions-summary')).toContainText('3 asked · 67% answered');
  await expect(page.getByTestId('unanswered-row')).toHaveCount(1);
  await page.getByTestId('questions-view-repeated').click();
  await expect(page.getByTestId('unanswered-row')).toContainText('×2');
});

test('a job started elsewhere can be cancelled and skipped files are named', async ({ page, api }) => {
  api.config = { watch_enabled: true };
  api.activeJobs = [
    {
      job_id: 'outside-1',
      filename: 'watched.md',
      state: 'embedding',
      chunks_total: 4,
      chunks_done: 1,
      error: null,
      result: null,
    },
  ];
  api.watchStatus = {
    folder: '/inbox',
    last_scan_at: null,
    files: 2,
    problems: ['/inbox/link.pdf: not a regular file'],
    deferred: [],
    retrying: [],
    gave_up: [],
  };
  api.finishedJobs = [
    { job_id: 'old-1', filename: 'broken.pdf', state: 'failed', error: 'Unreadable PDF.' },
  ];
  await page.goto('/');
  await page.getByTestId('rail-corpus').click();
  await expect(page.getByTestId('watch-problem')).toContainText('link.pdf');
  await page.getByRole('button', { name: 'Cancel indexing watched.md' }).click();
  await expect.poll(() => api.cancelledOutsideJobs).toEqual(['outside-1']);
  await page.getByText('Recent failures').click();
  await expect(page.getByTestId('failed-job')).toContainText('broken.pdf: Unreadable PDF.');
});

test('search filters narrow the request and the match is marked', async ({ page, api }) => {
  api.searchResults = [
    {
      rank: 1,
      filename: 'guide.md',
      page: 1,
      section: 'Retrieval',
      chunk_id: 'c1',
      chunk_ordinal: 1,
      text: GUIDE_SOURCE.text,
      retrieval_score: 0.03,
      rerank_score: 0.93,
      spans: [{ start: 0, end: 20, quote: 'DocRAG fuses dense and BM25 results' }],
    },
  ];
  await page.goto('/');
  await page.getByTestId('rail-search').click();
  await page.getByText('Filters').click();
  await page.getByTestId('search-filter-section').fill('Retrieval');
  await page.getByTestId('search-filter-page-min').fill('1');
  await page.getByTestId('search-input').fill('fusion');
  await page.getByTestId('search-submit').click();
  await expect(page.getByTestId('search-match')).toHaveText('DocRAG fuses dense and BM25 results');
  expect(api.searches.at(-1)).toMatchObject({ query: 'fusion', section: 'Retrieval', page_min: 1 });
});
