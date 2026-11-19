// Corpus cards: upload tags, preview, URL refresh, duplicates, citing questions.
import { answerStream, expect, test } from '../fixtures';
import { open, ask } from '../helpers';

test('tags typed for uploads go with each upload, cleaned up', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
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
  api.documents = ['guide.md', 'notes.md'];
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

test('a fetched document says when it is checked, and how often can be changed', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
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

test('a cited source uploaded again after the answer is flagged on its card', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  api.stream = answerStream();
  // notes.md listed as it was; guide.md's upload time later than the answer below.
  api.uploadedAts = { 'guide.md': Date.now() / 1000 + 3600, 'notes.md': 1 };
  await open(page);
  await ask(page);
  await expect(page.getByTestId('citation-changed')).toHaveText('document changed since');
  await page.getByTestId('chat-message-save').click();
  await page.getByTestId('rail-saved').click();
  await expect(page.getByTestId('saved-changed')).toContainText('changed since: guide.md');
});

test('a document card lists the questions that cited it, here and in the log', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  api.config = { query_log_enabled: true };
  api.unanswered = [
    {
      created_at: Date.now() / 1000,
      question: 'How is retrieval fused?',
      status: 'ok',
      top_score: 0.8,
      mode: 'stream',
      trace_id: null,
      filenames: ['guide.md'],
    },
  ];
  api.stream = answerStream();
  await open(page);
  await ask(page);

  await page.getByTestId('rail-corpus').click();
  const card = page.getByTestId('corpus-panel-item').filter({ hasText: 'guide.md' });
  await card.getByTestId('corpus-usage-toggle').click();
  await expect(card.getByTestId('corpus-usage-answer')).toHaveText(/What retrieval strategy/);
  await expect(card.getByTestId('corpus-usage-logged')).toContainText('1 logged question cited it');
  expect(api.queryLogFilenames).toEqual(['guide.md']);
});

test('a document with the same bytes as another says so on its card', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  api.duplicateGroups = [['guide.md', 'notes.md']];
  await open(page);
  await page.getByTestId('rail-corpus').click();
  await expect(page.locator('[data-corpus-card="guide.md"]')).toContainText('same content as notes.md');
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
