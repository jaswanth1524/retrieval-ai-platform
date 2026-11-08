import { answerStream, expect, test } from '../fixtures';

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

test('a deleted conversation comes back with Undo', async ({ page, api }) => {
  api.stream = answerStream();
  await open(page);
  await ask(page);
  await page.getByTestId('rail-chat').click();
  const title = await page.getByTestId('conversation-select').first().getAttribute('title');
  await page.getByRole('button', { name: `Delete ${title}` }).click();
  await page.getByTestId('conversation-delete-confirm').click();
  await expect(page.locator(`[data-testid="conversation-select"][title="${title}"]`)).toHaveCount(0);
  await page.getByTestId('toast-action').click();
  // Back in the list, and open again.
  await expect(page.locator(`[data-testid="conversation-select"][title="${title}"]`)).toHaveAttribute(
    'aria-current',
    'true',
  );
});

test('a link to an answer opens its conversation again', async ({ page, api }) => {
  api.stream = answerStream();
  await open(page);
  await ask(page);
  const answerId = await page.locator('[data-role="assistant"]').last().getAttribute('data-turn-id');
  const conversation = await page.evaluate(() => {
    const stored = JSON.parse(localStorage.getItem('docrag-chat-history') ?? '{}');
    return stored.activeConversationId as string;
  });
  // Another conversation is open when the link is followed.
  await page.getByTestId('rail-chat').click();
  await page.getByTestId('context-panel-action').click();
  await expect(page.locator(`[data-turn-id="${answerId}"]`)).toHaveCount(0);
  await page.evaluate((hash) => {
    window.location.hash = hash;
  }, `c=${conversation}&t=${answerId}`);
  await expect(page.locator(`[data-turn-id="${answerId}"]`)).toBeVisible();
});

test('Reformat answers again and keeps the earlier answer', async ({ page, api }) => {
  api.stream = answerStream('First answer [1].');
  await open(page);
  await ask(page);
  api.stream = answerStream('| a | b |\n|---|---|\n| 1 | 2 | [1]');
  await page.getByTestId('chat-message-reformat').click();
  await page.getByTestId('chat-message-reformat-table').click();
  await expect(page.getByTestId('question-submit')).toHaveText('Ask');
  expect(api.questionBodies.at(-1)).toMatchObject({ answer_format: 'table', use_cache: false });
  await page.getByTestId('answer-versions-open').click();
  await expect(page.getByTestId('answer-version-content')).toContainText('First answer');
  await expect(page.locator('[data-role="assistant"]')).toHaveCount(1);
});

test('the viewer finds a word among its passages', async ({ page, api }) => {
  api.stream = answerStream();
  await open(page);
  await ask(page);
  await page.getByTestId('citation-card').first().click();
  const viewer = page.getByRole('dialog');
  await viewer.getByTestId('viewer-find-open').click();
  await viewer.getByTestId('viewer-find-input').fill('fusion');
  await expect(viewer.getByTestId('viewer-find-count')).toHaveText('1 of 1 passages');
  await expect(viewer.getByTestId('viewer-found')).toHaveText('Fusion');
});

test('conversations are labelled and filtered by label', async ({ page, api }) => {
  api.stream = answerStream();
  await open(page);
  await ask(page);
  await page.getByTestId('rail-chat').click();
  await page.getByTestId('conversation-label').first().click();
  await page.getByTestId('conversation-labels-input').fill('retrieval');
  await page.getByTestId('conversation-labels-input').press('Enter');
  await expect(page.getByTestId('conversation-labels')).toHaveText(/#retrieval/);
  await page.getByTestId('conversation-label-filter').click();
  await expect(page.getByTestId('conversation-select')).toHaveCount(1);
});

test('a duplicate keeps itself and deletes its copy', async ({ page, api }) => {
  api.documents = ['guide.md', 'copy.md'];
  api.duplicateGroups = [['copy.md', 'guide.md']];
  await open(page);
  await page.getByTestId('rail-corpus').click();
  const card = page.locator('[data-corpus-card="guide.md"]');
  await card.getByTestId('corpus-keep-this').click();
  await card.getByTestId('corpus-keep-this-confirm').click();
  await expect(page.locator('[data-corpus-card="copy.md"]')).toHaveCount(0);
  expect(api.documents).toEqual(['guide.md']);
});

test('Administration lists backups, runs one and checks the index', async ({ page, api }) => {
  api.backups = {
    backups: [{ name: 'docrag-20261101-000000.zip', size: 2048, created_at: Date.now() / 1000, verified: true }],
    interval_hours: 24,
    keep: 7,
  };
  api.answerCache = {
    enabled: true,
    entries: [{ question: 'what is docrag?', age_seconds: 12 }],
    total_entries: 1,
    max_entries: 256,
    ttl_seconds: 3600,
    in_flight: 0,
  };
  await open(page);
  await page.getByTestId('open-settings').click();
  await page.getByTestId('settings-admin-toggle').locator('summary').click();
  await expect(page.getByTestId('backup-item')).toContainText('verified');
  await page.getByTestId('backups-run').click();
  await expect.poll(() => api.backupRuns).toBe(1);
  await page.getByTestId('answer-cache-clear').click();
  await expect.poll(() => api.cacheClears).toBe(1);
  await page.getByTestId('workspace-settings-min-score').fill('0.3');
  await page.getByTestId('workspace-settings-save').click();
  await expect.poll(() => api.settingsPuts.at(-1)).toMatchObject({ rerank_min_score: 0.3 });
  await page.getByTestId('consistency-run').click();
  await expect(page.getByTestId('consistency-result')).toContainText('agree with their originals');
});
