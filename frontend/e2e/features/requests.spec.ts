import { answerStream, expect, GUIDE_SOURCE, test } from '../fixtures';

test.beforeEach(async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  await page.goto('/');
  await expect(page.getByTestId('question-textarea')).toBeEnabled();
});

async function ask(page: import('@playwright/test').Page, question: string) {
  const box = page.getByTestId('question-textarea');
  await box.fill(question);
  await box.press('Enter');
  await expect(page.getByTestId('question-textarea')).toBeEnabled();
}

test('a scoped search sends only the chosen documents', async ({ page, api }) => {
  await page.getByTestId('composer-scope-button').click();
  await page.getByTestId('scope-item').filter({ hasText: 'notes.md' }).click();
  await page.keyboard.press('Escape');
  await ask(page, 'What is on the roadmap?');

  expect(api.questionBodies.at(-1)?.filenames).toEqual(['notes.md']);
});

test('saved retrieval settings travel with the next question', async ({ page, api }) => {
  await page.getByTestId('open-settings').click();
  const topK = page.getByTestId('settings-rerank-top-k');
  await topK.focus();
  await page.keyboard.press('ArrowRight');
  const chosen = Number(await topK.inputValue());
  await page.getByTestId('settings-save').click();
  await expect(page.getByTestId('settings-modal')).toBeHidden();

  await ask(page, 'What retrieval strategy does DocRAG use?');

  expect(api.questionBodies.at(-1)?.rerank_top_k).toBe(chosen);
});

test('a citation opens the source viewer at that chunk, and closing returns focus', async ({
  page,
  api,
}) => {
  api.stream = answerStream('DocRAG uses hybrid retrieval [1].');
  await ask(page, 'What retrieval strategy does DocRAG use?');

  const citation = page.getByRole('button', { name: `Open ${GUIDE_SOURCE.filename} at source 1` });
  await citation.click();
  const viewer = page.getByRole('dialog');
  await expect(viewer.getByText(GUIDE_SOURCE.text)).toBeVisible();

  await page.keyboard.press('Escape');
  await expect(viewer).toBeHidden();
  await expect(citation).toBeFocused();
});

test('conversations survive a reload and can be switched between', async ({ page, api }) => {
  api.stream = answerStream('First answer [1].');
  await ask(page, 'First question?');
  await page.keyboard.press('ControlOrMeta+Shift+o');
  api.stream = answerStream('Second answer [1].');
  await ask(page, 'Second question?');

  await page.reload();
  await expect(page.getByText('Second answer')).toBeVisible();

  await page.getByTestId('conversation-select').filter({ hasText: 'First question' }).click();
  await expect(page.getByText('First answer')).toBeVisible();
  await expect(page.getByText('Second answer')).toHaveCount(0);
});
