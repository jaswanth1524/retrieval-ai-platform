import { readFileSync } from 'node:fs';
import { answerStream, expect, GUIDE_SOURCE, test, traceSummary } from '../fixtures';

test.beforeEach(async ({ api }) => {
  api.documents = ['guide.md', 'notes.md'];
});

async function open(page: import('@playwright/test').Page) {
  await page.goto('/');
  await expect(page.getByTestId('question-textarea')).toBeEnabled();
}

async function ask(page: import('@playwright/test').Page, question: string) {
  const box = page.getByTestId('question-textarea');
  await box.fill(question);
  await box.press('Enter');
  await expect(page.getByTestId('question-submit')).toHaveText('Ask');
}

test('asking about a passage of another document ignores the conversation scope', async ({
  page,
  api,
}) => {
  api.stream = answerStream('DocRAG uses hybrid retrieval [1].');
  await open(page);
  await ask(page, 'What retrieval strategy does DocRAG use?');
  // Scoped to notes.md; the passage asked about is guide.md's.
  await page.getByTestId('composer-scope-button').click();
  await page.getByTestId('scope-item').filter({ hasText: 'notes.md' }).click();
  await page.keyboard.press('Escape');

  await page.getByRole('button', { name: `Open ${GUIDE_SOURCE.filename} at source 1` }).click();
  const viewer = page.getByRole('dialog');
  const passage = viewer.getByTestId('viewer-chunk').filter({ hasText: GUIDE_SOURCE.text });
  await passage.getByTestId('viewer-ask').click();
  await passage.getByTestId('viewer-ask-input').fill('Why hybrid?');
  // Escape closes the box and puts focus back on its button.
  await page.keyboard.press('Escape');
  await expect(passage.getByTestId('viewer-ask')).toBeFocused();
  await passage.getByTestId('viewer-ask').click();
  await passage.getByTestId('viewer-ask-input').fill('Why hybrid?');
  await passage.getByTestId('viewer-ask-submit').click();

  await expect.poll(() => api.questionBodies.length).toBe(2);
  const body = api.questionBodies[1];
  expect(body.chunk_ids).toEqual([GUIDE_SOURCE.chunk_id]);
  expect(body.filenames).toBeUndefined();
  expect(body.tags ?? []).toEqual([]);
});

test('a saved answer is listed under Saved answers, opened and exported', async ({ page, api }) => {
  api.stream = answerStream('DocRAG uses hybrid retrieval [1].');
  await open(page);
  await ask(page, 'What retrieval strategy does DocRAG use?');
  await page.getByTestId('chat-message-save').click();

  await page.getByTestId('rail-saved').click();
  await expect(page.getByTestId('saved-row')).toContainText('What retrieval strategy does DocRAG use?');
  const download = page.waitForEvent('download');
  await page.getByTestId('context-panel-action').click();
  const file = await download;
  expect(file.suggestedFilename()).toBe('docrag-saved-answers.md');
  const markdown = readFileSync((await file.path())!, 'utf8');
  expect(markdown).toContain('## What retrieval strategy does DocRAG use?');
  expect(markdown).toContain('DocRAG uses hybrid retrieval [1].');

  await page.getByTestId('saved-unsave').click();
  await expect(page.getByTestId('saved-row')).toHaveCount(0);
});

test('passages like one in the viewer are searched for in the Search panel', async ({ page, api }) => {
  api.stream = answerStream('DocRAG uses hybrid retrieval [1].');
  await open(page);
  await ask(page, 'What retrieval strategy does DocRAG use?');
  await page.getByRole('button', { name: `Open ${GUIDE_SOURCE.filename} at source 1` }).click();
  await page
    .getByRole('dialog')
    .getByTestId('viewer-chunk')
    .filter({ hasText: GUIDE_SOURCE.text })
    .getByTestId('viewer-similar')
    .click();

  await expect(page.getByTestId('search-like')).toContainText(`Passages like ${GUIDE_SOURCE.filename}`);
  expect(api.searches.at(-1)).toMatchObject({ query: '', like_chunk_id: GUIDE_SOURCE.chunk_id });
});

test('stricter answers from Settings ride with the next question', async ({ page, api }) => {
  await open(page);
  await page.getByTestId('open-settings').click();
  await page.getByTestId('settings-passages-only').check();
  await page.getByTestId('settings-strong-only').check();
  await page.getByTestId('settings-save').click();
  await expect(page.getByTestId('settings-modal')).toBeHidden();

  await ask(page, 'What retrieval strategy does DocRAG use?');
  expect(api.questionBodies.at(-1)).toMatchObject({ passages_only: true, strong_matches_only: true });
});

test('a failed job is retried from the corpus panel', async ({ page, api }) => {
  api.finishedJobs = [
    {
      job_id: 'old-job',
      filename: 'notes.md',
      state: 'failed',
      chunks_total: 0,
      chunks_done: 0,
      error: 'The URL did not answer in time.',
      result: null,
      kind: 'url',
      retryable: true,
    },
  ];
  await open(page);
  await page.getByTestId('rail-corpus').click();
  await page.getByText('Recent failures').click();
  await page.getByTestId('finished-job-retry').click();
  await expect.poll(() => api.retriedJobs).toEqual(['old-job']);
});

test('tags typed for uploads go with each upload, cleaned up', async ({ page, api }) => {
  await open(page);
  await page.getByTestId('rail-corpus').click();
  const field = page.getByTestId('upload-tags');
  await field.fill(' hr, Policy,  hr ');
  await field.press('Enter');
  await expect(field).toHaveValue('hr, Policy');
  await page.getByTestId('upload-input').setInputFiles({
    name: 'notes.txt',
    mimeType: 'text/plain',
    buffer: Buffer.from('Roadmap notes.'),
  });
  await page.getByTestId('upload-button').click();
  await expect.poll(() => api.uploadBodies.length).toBe(1);
  const tags = [...api.uploadBodies[0].matchAll(/name="tags"\r\n\r\n([^\r]*)/g)].map((match) => match[1]);
  expect(tags).toEqual(['hr', 'Policy']);
});

test('a file is previewed without being indexed', async ({ page, api }) => {
  await open(page);
  await page.getByTestId('rail-corpus').click();
  await page.getByTestId('preview-file-input').setInputFiles({
    name: 'draft.md',
    mimeType: 'text/markdown',
    buffer: Buffer.from('# Overview\n\nWhat the parser read.'),
  });
  await expect(page.getByTestId('preview-result')).toContainText('draft.md: 2 passages in 1 section');
  expect(api.previews).toEqual(['draft.md']);
  expect(api.documents).not.toContain('draft.md');
});

test('a fetched document says when it is checked, and how often can be changed', async ({
  page,
  api,
}) => {
  api.sourceUrls = { 'guide.md': 'https://example.com/guide.md' };
  api.urlRefresh = {
    'guide.md': {
      refresh_minutes: null,
      last_checked: Date.now() / 1000 - 7200,
      last_outcome: 'unchanged',
      failures: 0,
      next_due: Date.now() / 1000 + 3600,
    },
  };
  await open(page);
  await page.getByTestId('rail-corpus').click();
  await expect(page.getByTestId('corpus-refresh-state')).toContainText('unchanged 2h ago · next in 1 h');
  await page.getByTestId('corpus-refresh-every').selectOption({ label: 'never' });
  await expect.poll(() => api.refreshSettings).toEqual(['guide.md=0']);
});

test('the asked questions download as CSV, formulas defused', async ({ page, api }) => {
  api.config = { query_log_enabled: true };
  api.unanswered = [
    {
      created_at: Date.now() / 1000,
      question: '=cmd|calc',
      status: 'insufficient_context',
      top_score: 0.05,
      mode: 'stream',
      trace_id: null,
    },
  ];
  await open(page);
  await page.getByTestId('rail-unanswered').click();
  const download = page.waitForEvent('download');
  await page.getByTestId('questions-export').click();
  const csv = readFileSync((await (await download).path())!, 'utf8');
  expect(csv.split('\r\n')[0]).toBe(
    'question,times_asked,times_not_answered,last_asked,last_status,best_match,mode',
  );
  expect(csv).toContain("'=cmd|calc,1,1,");
});

test('the trace list asks the server for traces answered one way', async ({ page, api }) => {
  api.traces = [
    { ...traceSummary('t-answered'), answer_status: 'answered' },
    { ...traceSummary('t-missed'), answer_status: 'no_context' },
  ];
  await open(page);
  await page.getByTestId('rail-traces').click();
  await expect(page.getByTestId('traces-panel-row')).toHaveCount(2);
  await page.getByTestId('traces-answer-filter').selectOption('unanswered');
  await expect(page.getByTestId('traces-panel-row')).toHaveCount(1);
  expect(api.traceQueries.at(-1)).toEqual(['insufficient_context', 'no_context']);
});
