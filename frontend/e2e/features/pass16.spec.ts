import { answerStream, expect, test, traceSummary } from '../fixtures';

async function open(page: import('@playwright/test').Page) {
  await page.goto('/');
  await expect(page.getByTestId('question-textarea')).toBeEnabled();
}

async function ask(page: import('@playwright/test').Page, question = 'What retrieval strategy does DocRAG use?') {
  const box = page.getByTestId('question-textarea');
  await box.fill(question);
  await box.press('Enter');
  await expect(page.getByTestId('question-submit')).toHaveText('Ask');
}

test('an earlier answer is compared with the current one and used again', async ({ page, api }) => {
  api.stream = answerStream('Retrieval is hybrid [1].');
  await open(page);
  await ask(page);
  api.stream = answerStream('Retrieval is dense and sparse [1].');
  await page.getByTestId('chat-message-regenerate').click();
  await expect(page.getByTestId('question-submit')).toHaveText('Ask');
  await page.getByTestId('answer-versions-open').click();
  await page.getByTestId('answer-version-compare').click();
  await expect(page.getByTestId('answer-version-diff').locator('del')).toContainText('hybrid');
  await page.getByTestId('answer-version-use').click();
  await expect(page.locator('[data-role="assistant"]').last()).toContainText('Retrieval is hybrid');
  await expect(page.locator('[data-role="assistant"]')).toHaveCount(1);
});

test('an answer worth a second look is counted on the rail and marked reviewed', async ({ page, api }) => {
  api.stream = answerStream('The documents do not say whether there is a gym.', { answer_status: 'insufficient_context', sources: [] });
  await open(page);
  await ask(page, 'Is there a gym?');
  await expect(page.getByTestId('rail-badge-review')).toHaveText('1');
  await page.getByTestId('rail-review').click();
  await page.getByTestId('review-mark').click();
  await expect(page.getByTestId('review-empty')).toContainText('has been reviewed');
  await expect(page.getByTestId('rail-badge-review')).toHaveCount(0);
});

test('a passage link opens the viewer at that passage', async ({ page, api }) => {
  api.stream = answerStream();
  await open(page);
  await page.evaluate(() => {
    window.location.hash = 'd=guide.md';
  });
  await expect(page.getByRole('dialog')).toContainText('guide.md');
  await expect(page.getByTestId('viewer-copy-link')).toBeVisible();
});

test('a label is renamed everywhere and its conversations exported', async ({ page, api }) => {
  api.stream = answerStream();
  await open(page);
  await ask(page);
  await page.getByTestId('rail-chat').click();
  await page.getByTestId('conversation-label').first().click();
  await page.getByTestId('conversation-labels-input').fill('retrieval');
  await page.getByTestId('conversation-labels-input').press('Enter');
  await page.getByTestId('conversation-label-filter').click();
  await page.getByTestId('conversation-label-rename').click();
  await page.getByTestId('conversation-label-rename-input').fill('search');
  await page.getByTestId('conversation-label-rename-input').press('Enter');
  await expect(page.getByTestId('conversation-labels')).toHaveText(/#search/);
  const download = page.waitForEvent('download');
  await page.getByTestId('conversation-label-export').click();
  expect((await download).suggestedFilename()).toMatch(/^docrag-search-/);
});

test('a bar is tried on recent questions from the score report', async ({ page, api }) => {
  api.traces = [traceSummary('t-1', 'first?')];
  api.whatIf = { settings: { rerank_min_score: 0.1 }, checked: 4, would_answer: 3, newly_answered: 3, newly_unanswered: 0, items: [] };
  await open(page);
  await page.getByTestId('rail-traces').click();
  await page.getByTestId('traces-score-report').click();
  await page.getByTestId('what-if-bar').fill('0.1');
  await page.getByTestId('what-if-run').click();
  await expect(page.getByTestId('what-if-result')).toContainText('3 of 4 would be answered');
  expect(api.whatIfs.at(-1)).toMatchObject({ rerank_min_score: 0.1, only_unanswered: true });
});

test('a sitemap import shows its progress and can be cancelled', async ({ page, api }) => {
  api.sitemapImports = [
    {
      id: 'imp1',
      url: 'https://example.com/sitemap.xml',
      state: 'running',
      total: 10,
      done: 2,
      queued: 2,
      failed: 0,
      skipped: 0,
      unchanged: 0,
      cancelled_pages: 0,
      started_at: Date.now() / 1000,
      finished_at: null,
      problems: [],
    },
  ];
  await open(page);
  await page.getByTestId('rail-corpus').click();
  await expect(page.getByTestId('sitemap-import')).toContainText('2 of 10 page(s)');
  await page.getByTestId('sitemap-import-cancel').click();
  await expect.poll(() => api.sitemapCancels).toEqual(['imp1']);
});

test('Administration repairs what the consistency check found', async ({ page, api }) => {
  api.backups = {
    backups: [{ name: 'docrag-20261101-000000.zip', size: 2048, created_at: Date.now() / 1000, verified: true }],
    interval_hours: 24,
    keep: 7,
  };
  api.consistency = {
    documents: 1,
    originals_kept: true,
    problems: 1,
    originals_without_document: ['orphan.txt'],
    documents_without_original: [],
    versions_without_document: [],
    incomplete: [],
    stale: [],
    duplicate_groups: [],
    in_flight: [],
  };
  await open(page);
  await page.getByTestId('open-settings').click();
  await page.getByTestId('settings-admin-toggle').locator('summary').click();
  await page.getByTestId('backup-delete').click();
  await page.getByTestId('backup-delete-confirm').click();
  await expect.poll(() => api.backupDeletes).toEqual(['docrag-20261101-000000.zip']);
  await page.getByTestId('consistency-run').click();
  await page.getByTestId('consistency-repair-delete_orphan_originals').check();
  await page.getByTestId('consistency-repair-run').click();
  await page.getByTestId('consistency-repair-confirm').click();
  await expect.poll(() => api.repairs).toEqual([{ delete_orphan_originals: true }]);
});
