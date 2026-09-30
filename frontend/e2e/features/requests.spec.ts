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
  // The textarea stays editable while an answer streams; Stop turning back into Ask is
  // what marks the answer finished.
  await expect(page.getByTestId('question-submit')).toHaveText('Ask');
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

test('a citation into a since re-indexed document says the passage changed', async ({
  page,
  api,
}) => {
  api.stream = answerStream('DocRAG uses hybrid retrieval [1].');
  await ask(page, 'What retrieval strategy does DocRAG use?');
  // Re-chunking gives every chunk a new id, so the cited one is gone.
  api.contentChunks = [
    { chunk_id: 'n0', page: 1, section: 'Intro', text: 'DocRAG overview, rewritten.', chunk_ordinal: 1 },
  ];

  await page.getByRole('button', { name: `Open ${GUIDE_SOURCE.filename} at source 1` }).click();

  const viewer = page.getByRole('dialog');
  await expect(viewer.getByTestId('viewer-passage-missing')).toBeVisible();
  await expect(viewer.getByText('DocRAG overview, rewritten.')).toBeVisible();
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

test('tagging a document lets a question be scoped by that tag', async ({ page, api }) => {
  api.documents = ['guide.md', 'contract.md'];
  await page.reload();
  await page.getByTestId('rail-corpus').click();
  await page.getByLabel('Edit tags for contract.md').click();
  await page.getByTestId('corpus-tags-input').fill('legal');
  await page.getByTestId('corpus-tags-input').press('Enter');
  await expect(page.getByText('#legal')).toBeVisible();

  await page.getByTestId('composer-scope-button').click();
  await page.getByLabel('#legal').check();
  await page.keyboard.press('Escape');
  await ask(page, 'What are the terms?');

  expect(api.questionBodies.at(-1)?.tags).toEqual(['legal']);
  await expect(page.getByTestId('composer-scope-button')).toContainText('#legal');

  // "All documents" clears files and tags together (two scope updates in one click).
  await page.getByTestId('composer-scope-button').click();
  await page.getByTestId('scope-item').filter({ hasText: 'guide.md' }).click();
  await page.getByTestId('scope-all').check();
  await page.keyboard.press('Escape');
  await ask(page, 'Everything?');
  expect(api.questionBodies.at(-1)?.tags).toBeUndefined();
  expect(api.questionBodies.at(-1)?.filenames).toBeUndefined();
});

test('Regenerate asks past the answer cache', async ({ page, api }) => {
  await ask(page, 'What is DocRAG?');
  expect(api.questionBodies.at(-1)?.use_cache).toBeUndefined();

  await page.getByTestId('chat-message-regenerate').click();
  await expect(page.getByTestId('question-submit')).toHaveText('Ask');

  expect(api.questionBodies.at(-1)?.use_cache).toBe(false);
});

test('editing a question asks it again in a new conversation', async ({ page, api }) => {
  api.stream = answerStream('First answer [1].');
  await ask(page, 'Original question?');
  api.stream = answerStream('Edited answer [1].');

  await page.getByTestId('chat-message-edit').click();
  await page.getByTestId('chat-message-edit-input').fill('Edited question?');
  await page.getByTestId('chat-message-edit-send').click();

  await expect(page.getByText('Edited answer')).toBeVisible();
  // The thread shows the fork; the original conversation is kept alongside it.
  await expect(page.getByTestId('chat-thread').getByText('Original question?')).toHaveCount(0);
  await expect(page.getByTestId('conversation-select')).toHaveCount(2);
  await expect(page.getByTestId('conversation-select').filter({ hasText: 'Original question?' })).toBeVisible();
});

test('the source viewer fetches a page around the citation, then more on demand', async ({
  page,
  api,
}) => {
  // A long document: the cited chunk is number 80 of 120.
  api.contentChunks = Array.from({ length: 120 }, (_, i) => ({
    chunk_id: i === 79 ? GUIDE_SOURCE.chunk_id : `k${i}`,
    page: 1,
    section: 'Body',
    text: i === 79 ? GUIDE_SOURCE.text : `Passage number ${i + 1}.`,
    chunk_ordinal: i + 1,
  }));
  await ask(page, 'What does the long document say?');
  const contentRequests: string[] = [];
  page.on('request', (request) => {
    if (request.url().includes('/content')) contentRequests.push(request.url());
  });

  await page.getByRole('button', { name: `Open ${GUIDE_SOURCE.filename} at source 1` }).click();

  const viewer = page.getByRole('dialog');
  await expect(viewer.getByText(GUIDE_SOURCE.text)).toBeVisible();
  await expect(viewer.getByTestId('viewer-chunk')).toHaveCount(61);
  await expect(viewer.getByTestId('viewer-load-earlier')).toHaveText('Load earlier (49 hidden)');
  expect(contentRequests[0]).toContain(`around=${GUIDE_SOURCE.chunk_id}`);

  await viewer.getByTestId('viewer-load-earlier').click();
  await expect(viewer.getByTestId('viewer-load-earlier')).toHaveText('Load earlier (19 hidden)');
  await viewer.getByTestId('viewer-show-all').click();
  await expect(viewer.getByTestId('viewer-chunk')).toHaveCount(120);
});
