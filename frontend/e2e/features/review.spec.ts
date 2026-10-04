import { answerStream, expect, GUIDE_SOURCE, test } from '../fixtures';

async function ask(page: import('@playwright/test').Page, question: string) {
  const box = page.getByTestId('question-textarea');
  await box.fill(question);
  await box.press('Enter');
  await expect(page.getByTestId('question-submit')).toHaveText('Ask');
}

test('an inline [n] marker opens the source it cites', async ({ page, api }) => {
  api.stream = answerStream('DocRAG uses hybrid retrieval [1].');
  await page.goto('/');
  await ask(page, 'What retrieval strategy does DocRAG use?');

  await page.getByRole('button', { name: `Source 1: ${GUIDE_SOURCE.filename}, page 1` }).click();

  await expect(page.getByRole('dialog', { name: `Source: ${GUIDE_SOURCE.filename}` })).toBeVisible();
});

test('the read-only key hides what it cannot do', async ({ page, api }) => {
  api.access = 'read';
  api.config = { feedback_enabled: true };
  await page.goto('/');
  await page.getByTestId('rail-corpus').click();

  await expect(page.getByTestId('corpus-readonly')).toBeVisible();
  await expect(page.getByTestId('upload-dropzone')).toHaveCount(0);
  await expect(page.getByRole('button', { name: `Delete ${GUIDE_SOURCE.filename}` })).toHaveCount(0);
  await expect(page.getByTestId('rail-feedback')).toHaveCount(0);

  await page.keyboard.press('ControlOrMeta+k');
  await page.getByRole('combobox').fill('backup');
  await expect(page.getByTestId('palette-empty')).toBeVisible();
});

test('the feedback panel lists ratings and filters to thumbs-down', async ({ page, api }) => {
  api.config = { feedback_enabled: true };
  api.access = 'full';
  api.feedbackItems = [
    {
      id: 'f1',
      trace_id: null,
      question: 'Where are uploads stored?',
      answer_excerpt: 'Not in the documents.',
      cited_filenames: [],
      rating: 'down',
      citation_source_number: null,
      created_at: Date.now() / 1000,
    },
    {
      id: 'f2',
      trace_id: null,
      question: 'What is RRF?',
      answer_excerpt: 'Reciprocal Rank Fusion [1].',
      cited_filenames: ['guide.md'],
      rating: 'up',
      citation_source_number: null,
      created_at: Date.now() / 1000,
    },
  ];
  await page.goto('/');
  await page.getByTestId('rail-feedback').click();

  await expect(page.getByTestId('feedback-panel-row')).toHaveCount(2);
  await page.getByRole('button', { name: '👎 Down' }).click();
  await expect(page.getByTestId('feedback-panel-row')).toHaveCount(1);
  await expect(page.getByTestId('feedback-panel-row')).toContainText('Where are uploads stored?');
});

test('a backup restores from the palette and its documents index', async ({ page, api }) => {
  api.importFilenames = ['restored.md'];
  await page.goto('/');
  await expect(page.getByTestId('question-textarea')).toBeEnabled();

  await page.keyboard.press('ControlOrMeta+k');
  await page.getByRole('combobox').fill('restore');
  const chooser = page.waitForEvent('filechooser');
  await page.getByRole('option', { name: /Restore documents from a backup/ }).click();
  await (await chooser).setFiles({
    name: 'docrag-export.zip',
    mimeType: 'application/zip',
    buffer: Buffer.from('PK'),
  });

  await expect(page.getByText(/Restoring 1 document/)).toBeVisible();
  await expect(
    page.getByTestId('corpus-panel-item').filter({ hasText: 'restored.md' }).getByLabel('Delete restored.md'),
  ).toBeVisible();
  expect(api.imports).toHaveLength(1);
});

test('a stored original downloads from its corpus card', async ({ page, api }) => {
  api.config = { raw_documents_enabled: true };
  api.reindexableFilenames = [GUIDE_SOURCE.filename];
  await page.goto('/');
  await page.getByTestId('rail-corpus').click();

  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: `Download the original of ${GUIDE_SOURCE.filename}` }).click();

  expect((await download).suggestedFilename()).toBe(GUIDE_SOURCE.filename);
});
