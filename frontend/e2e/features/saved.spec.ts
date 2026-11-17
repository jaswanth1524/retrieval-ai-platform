// Saved answers, quotes and the passage basket.
import { readFileSync } from 'node:fs';
import { answerStream, expect, test } from '../fixtures';
import { open, ask, askWithButton } from '../helpers';

test('a saved answer survives a reload and is listed under Saved', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  api.stream = answerStream();
  await page.goto('/');
  await askWithButton(page, 'What is DocRAG?');
  await page.getByTestId('chat-message-save').click();
  await expect(page.getByTestId('chat-message-save')).toHaveText('★ Saved');

  await page.reload();
  await expect(page.getByTestId('chat-message-save')).toHaveText('★ Saved');
  await page.getByTestId('rail-chat').click();
  await page.getByTestId('conversation-filter-saved').click();
  await expect(page.getByTestId('conversation-item')).toHaveCount(1);
  await expect(page.getByTestId('conversation-saved-count')).toContainText('1');
});

test('a saved answer is listed under Saved answers, opened and exported', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
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

test('passages collected in the basket are asked about together', async ({ page, api }) => {
  api.stream = answerStream('DocRAG fuses both searches [1].');
  await open(page);
  await page.getByTestId('rail-search').click();
  await page.getByTestId('search-input').fill('fusion');
  await page.getByTestId('search-submit').click();
  await page.getByTestId('search-pick').first().check();
  await page.getByTestId('search-collect').click();
  await page.getByTestId('rail-collected').click();
  await expect(page.getByTestId('basket-row')).toHaveCount(1);
  await page.getByTestId('basket-question').fill('How are the results combined?');
  await page.getByTestId('basket-ask').click();
  await expect(page.getByText('DocRAG fuses both searches')).toBeVisible();
  expect(api.questionBodies.at(-1)).toMatchObject({
    question: 'How are the results combined?',
    chunk_ids: ['c1'],
  });
});

test('a selection in an answer is saved as a quote, and an answer’s sources go to the basket', async ({
  page,
  api,
}) => {
  api.stream = answerStream('DocRAG uses hybrid retrieval with reranking [1].');
  await open(page);
  await ask(page);
  const text = page.locator('.chat-message__markdown p').last();
  await text.evaluate((element) => {
    // A mouse selection takes focus out of the question box.
    (document.activeElement as HTMLElement | null)?.blur();
    const range = document.createRange();
    range.selectNodeContents(element.firstChild ?? element);
    const selection = window.getSelection();
    selection?.removeAllRanges();
    selection?.addRange(range);
  });
  await page.getByTestId('selection-save').click();
  await expect(page.getByTestId('selection-save')).toHaveText('Saved');
  await page.getByTestId('chat-message-ask-within').click();
  await expect(page.getByTestId('basket-row')).toHaveCount(1);
  await expect(page.getByTestId('basket-question')).toBeFocused();
  await expect(page.getByTestId('quote-row')).toHaveCount(1);
  await expect(page.getByTestId('quote-row')).toContainText('DocRAG uses hybrid retrieval');
});
