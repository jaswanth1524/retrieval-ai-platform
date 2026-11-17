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

test('Administration starts a rebuild and reads the audit log', async ({ page, api }) => {
  api.auditEntries = [
    { created_at: Date.now() / 1000, workspace: 'default', event: 'document.delete', key: 'full', fields: { filename: 'old.pdf' } },
  ];
  await open(page);
  await page.getByTestId('open-settings').click();
  await page.getByTestId('settings-admin-toggle').locator('summary').click();
  await page.getByTestId('rebuild-run').click();
  await page.getByRole('button', { name: 'Rebuild', exact: true }).click();
  await expect(page.getByTestId('rebuild-result')).toContainText('Indexing 2 documents');
  expect(api.rebuilds).toBe(1);
  await page.getByTestId('audit-load').click();
  await expect(page.getByTestId('audit-entry')).toContainText('document.delete (full) filename=old.pdf');
  await page.getByTestId('notify-test-run').click();
  await expect(page.getByTestId('notify-test-result')).toContainText('Notifications are off');
});

test('a follow-up says what it was searched as', async ({ page, api }) => {
  api.stream = answerStream('Twenty-eight days [1].', { retrieval_query: 'How many vacation days do staff get?' });
  await open(page);
  await ask(page, 'And for full-time staff?');
  await expect(page.getByTestId('chat-message-searched-as')).toContainText(
    'Searched as: How many vacation days do staff get?',
  );
});
