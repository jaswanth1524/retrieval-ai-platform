import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, test, traceSummary } from '../fixtures';

test.beforeEach(async ({ api }) => {
  api.documents = ['guide.md', 'notes.md'];
});

test('the unanswered questions panel puts one back in the composer', async ({ page, api }) => {
  api.config = { query_log_enabled: true };
  api.unanswered = [
    {
      created_at: Date.now() / 1000,
      question: 'Who approves travel?',
      status: 'insufficient_context',
      top_score: 0.05,
      mode: 'stream',
      trace_id: null,
    },
  ];
  await page.goto('/');
  await page.getByTestId('rail-unanswered').click();
  await expect(page.getByTestId('unanswered-row')).toContainText('Who approves travel?');
  await page.getByTestId('unanswered-row').getByRole('button', { name: 'Ask', exact: true }).click();
  await expect(page.getByTestId('question-textarea')).toHaveValue('Who approves travel?');
});

test('the unanswered panel is not offered when the query log is off', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('rail-corpus')).toBeVisible();
  await expect(page.getByTestId('rail-unanswered')).toHaveCount(0);
});

test('an earlier version of a document is restored after a confirm', async ({ page, api }) => {
  api.config = { raw_documents_enabled: true, document_versions_enabled: true };
  api.reindexableFilenames = ['guide.md'];
  api.versions = {
    'guide.md': [{ version: '20260101-120000', saved_at: Date.now() / 1000 - 3600, size: 4096 }],
  };
  api.jobSteps = [{ status: 200, body: { state: 'done', chunks_done: 2, chunks_total: 2 } }];
  await page.goto('/');
  await page.getByTestId('rail-corpus').click();
  const card = page.locator('[data-corpus-card="guide.md"]');
  await card.getByTestId('corpus-versions-toggle').click();
  await expect(card.getByTestId('corpus-versions')).toContainText('4 KB');
  await card.getByTestId('corpus-version-restore').click();
  await card.getByRole('button', { name: 'Restore', exact: true }).click();
  await expect.poll(() => api.restoredVersions).toEqual(['guide.md@20260101-120000']);
  await expect(page.getByTestId('corpus-totals')).toContainText('2 documents');
});

test('the corpus panel takes a whole folder', async ({ page }) => {
  await page.goto('/');
  await page.getByTestId('rail-corpus').click();
  // A real folder: a directory input is only handed a path to one.
  const folder = mkdtempSync(join(tmpdir(), 'docrag-folder-'));
  writeFileSync(join(folder, 'a.md'), '# A');
  writeFileSync(join(folder, 'photo.jpg'), 'jpg');
  await page.getByTestId('upload-folder-input').setInputFiles(folder);
  await expect(page.getByTestId('upload-staged-item')).toHaveCount(1);
  await expect(page.getByTestId('upload-staged-item')).toContainText('a.md');
  await expect(page.getByTestId('upload-folder-note')).toContainText('1 file of other types');
});

test('a conversation exports as a web page and traces as JSON Lines', async ({ page, api }) => {
  api.traces = [traceSummary()];
  await page.goto('/');
  const box = page.getByTestId('question-textarea');
  await box.fill('What retrieval strategy does DocRAG use?');
  await box.press('Enter');
  await expect(page.getByText('DocRAG uses hybrid retrieval')).toBeVisible();

  await page.keyboard.press('ControlOrMeta+k');
  const [htmlDownload] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('option', { name: /as a web page/ }).click(),
  ]);
  expect(htmlDownload.suggestedFilename()).toMatch(/\.html$/);

  await page.getByTestId('rail-traces').click();
  const [traces] = await Promise.all([
    page.waitForEvent('download'),
    page.getByTestId('traces-export').click(),
  ]);
  expect(traces.suggestedFilename()).toBe('docrag-traces-test.jsonl');
});

test('picked passages are asked about on their own', async ({ page, api }) => {
  await page.goto('/');
  await page.getByTestId('rail-search').click();
  await page.getByTestId('search-input').fill('fusion');
  await page.getByTestId('search-submit').click();
  await page.getByTestId('search-pick').first().check();
  await page.getByTestId('search-ask-picked').click();
  await expect(page.getByText('DocRAG uses hybrid retrieval')).toBeVisible();
  expect(api.questionBodies.at(-1)).toMatchObject({ question: 'fusion', chunk_ids: ['c1'] });
});
