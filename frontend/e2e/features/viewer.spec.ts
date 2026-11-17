// The document viewer: find, passage links, outline.
import { answerStream, expect, test } from '../fixtures';
import { open, ask } from '../helpers';

test('the viewer finds a word among its passages', async ({ page, api }) => {
  api.stream = answerStream();
  await open(page);
  await ask(page);
  await page.getByTestId('citation-card').first().click();
  const viewer = page.getByRole('dialog');
  await viewer.getByTestId('viewer-find-open').click();
  await viewer.getByTestId('viewer-find-input').fill('fusion');
  await expect(viewer.getByTestId('viewer-find-count')).toHaveText('1 of 1');
  await expect(viewer.getByTestId('viewer-found')).toHaveText('Fusion');
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

test('the viewer lists a document\'s contents and opens a section', async ({ page, api }) => {
  api.stream = answerStream('DocRAG uses hybrid retrieval [1].');
  await open(page);
  await ask(page);
  await page.getByTestId('citation-card').first().click();
  await page.getByTestId('viewer-outline-open').click();
  await expect(page.getByTestId('viewer-outline-entry')).toHaveCount(2);
  await page.getByTestId('viewer-outline-entry').first().click();
  await expect(page.getByTestId('viewer-outline')).toHaveCount(0);
});
