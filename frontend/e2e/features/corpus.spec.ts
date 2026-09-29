import { expect, test } from '../fixtures';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('question-textarea')).toBeEnabled();
  await page.getByRole('button', { name: 'Corpus' }).click();
});

test('deleting a document asks first, and Cancel keeps it', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  await page.reload();
  await page.getByRole('button', { name: 'Corpus' }).click();
  const card = page.getByTestId('corpus-panel-item').filter({ hasText: 'notes.md' });

  await card.getByRole('button', { name: 'Delete notes.md' }).click();
  await expect(card.getByRole('button', { name: 'Cancel' })).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(card.getByRole('button', { name: 'Delete notes.md' })).toBeVisible();

  await card.getByRole('button', { name: 'Delete notes.md' }).click();
  await card.getByRole('button', { name: 'Confirm' }).click();
  await expect(card).toHaveCount(0);
  expect(api.documents).toEqual(['guide.md']);
});

test('a stale document re-indexes from its stored original', async ({ page, api }) => {
  api.config = { raw_documents_enabled: true };
  api.staleFilenames = ['guide.md'];
  api.reindexableFilenames = ['guide.md'];
  api.jobSteps = [
    { status: 200, body: api.jobBody('embedding') },
    { status: 200, body: api.jobBody('done') },
  ];
  await page.reload();
  await page.getByRole('button', { name: 'Corpus' }).click();
  const card = page.getByTestId('corpus-panel-item').filter({ hasText: 'guide.md' });
  await expect(card).toContainText('older chunking');

  await card.getByRole('button', { name: 'Re-index guide.md' }).click();

  await expect(page.getByTestId('corpus-panel-item').filter({ hasText: 'guide.md' })).toHaveCount(1);
  await expect(card).not.toContainText('older chunking', { timeout: 10_000 });
  await expect(card.getByRole('button', { name: 'Delete guide.md' })).toBeVisible();
});

test('a failed upload can be retried without picking the file again', async ({ page, api }) => {
  api.jobSteps = [{ status: 200, body: api.jobBody('failed', 'Qdrant was unavailable.') }];
  await page.getByTestId('upload-input').setInputFiles({
    name: 'notes.txt',
    mimeType: 'text/plain',
    buffer: Buffer.from('Roadmap notes.'),
  });
  await page.getByTestId('upload-button').click();
  const failed = page.getByTestId('corpus-panel-item').filter({ hasText: 'Qdrant was unavailable.' });
  await expect(failed).toBeVisible();

  api.jobSteps = [{ status: 200, body: api.jobBody('done') }];
  await failed.getByRole('button', { name: 'Retry notes.txt' }).click();

  const indexed = page.getByTestId('corpus-panel-item').filter({ hasText: 'notes.txt' });
  await expect(indexed).toHaveCount(1, { timeout: 10_000 });
  await expect(indexed.getByRole('button', { name: 'Delete notes.txt' })).toBeVisible();
});

test('after a chunker upgrade, every stale document re-indexes in one go', async ({ page, api }) => {
  api.config = { raw_documents_enabled: true };
  api.documents = ['guide.md', 'notes.md', 'faq.md'];
  api.staleFilenames = ['guide.md', 'notes.md'];
  api.reindexableFilenames = ['guide.md', 'notes.md', 'faq.md'];
  api.jobSteps = [
    { status: 200, body: api.jobBody('embedding') },
    { status: 200, body: api.jobBody('done') },
  ];
  await page.reload();
  await page.getByRole('button', { name: 'Corpus' }).click();

  await page.getByTestId('reindex-all-stale').click();

  await expect(page.getByTestId('reindex-all-stale')).toHaveCount(0, { timeout: 10_000 });
  await expect(page.getByTestId('corpus-panel-item').filter({ hasText: 'older chunking' })).toHaveCount(0);
  await expect(page.getByTestId('corpus-panel-item')).toHaveCount(3);
});
