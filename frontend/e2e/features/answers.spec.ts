// Answer actions: ask again, regenerate, reformat, continue, versions, scope.
import { readFileSync } from 'node:fs';
import { answerStream, expect, GUIDE_SOURCE, sse, test, TIMINGS } from '../fixtures';
import { open, ask, askWithButton } from '../helpers';

test('an answer is asked again without one of its sources', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  const scored = { ...GUIDE_SOURCE, rerank_score: 0.91 };
  api.stream = sse(
    { type: 'sources', sources: [scored], trace_id: 'trace-1' },
    { type: 'delta', text: 'DocRAG uses hybrid retrieval [1].' },
    {
      type: 'done',
      answer: 'DocRAG uses hybrid retrieval [1].',
      sources: [scored],
      timings: TIMINGS,
      trace_id: 'trace-1',
    },
  );
  await page.goto('/');
  await askWithButton(page, 'How does retrieval work?');
  await expect(page.getByTestId('citation-score').first()).toHaveText(/0\.91/);

  await page.getByRole('button', { name: 'Ask again without source 1' }).first().click();
  await expect.poll(() => api.questionBodies.length).toBe(2);
  expect(api.questionBodies[1]).toMatchObject({
    question: 'How does retrieval work?',
    exclude_chunk_ids: [GUIDE_SOURCE.chunk_id],
    use_cache: false,
  });
});

test('j and k move between answers', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  api.stream = answerStream();
  await page.goto('/');
  await askWithButton(page, 'First question?');
  await askWithButton(page, 'Second question?');
  await page.getByTestId('chat-thread').click({ position: { x: 5, y: 5 } });

  await page.keyboard.press('k');
  const answers = page.locator('[data-testid="chat-message"][data-role="assistant"]');
  await expect(answers.last()).toBeFocused();
  await page.keyboard.press('k');
  await expect(answers.first()).toBeFocused();
  await page.keyboard.press('j');
  await expect(answers.last()).toBeFocused();
  await expect(answers.last()).toHaveAttribute('aria-label', 'Answer 2');
});

test('asking about a passage of another document ignores the conversation scope', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  api.stream = answerStream('DocRAG uses hybrid retrieval [1].');
  await open(page);
  await ask(page, 'What retrieval strategy does DocRAG use?');
  // Scoped to notes.md; the passage asked about is guide.md's.
  await page.getByTestId('composer-scope-button').click();
  await page.getByTestId('scope-item').filter({ hasText: 'notes.md' }).click();
  await page.keyboard.press('Escape');

  await page.getByRole('button', { name: `Open ${GUIDE_SOURCE.filename} at source 1` }).click();
  const viewer = page.getByRole('dialog');
  const passage = viewer.getByTestId('viewer-chunk').filter({ hasText: GUIDE_SOURCE.text });
  await passage.getByTestId('viewer-ask').click();
  await passage.getByTestId('viewer-ask-input').fill('Why hybrid?');
  // Escape closes the box and puts focus back on its button.
  await page.keyboard.press('Escape');
  await expect(passage.getByTestId('viewer-ask')).toBeFocused();
  await passage.getByTestId('viewer-ask').click();
  await passage.getByTestId('viewer-ask-input').fill('Why hybrid?');
  await passage.getByTestId('viewer-ask-submit').click();

  await expect.poll(() => api.questionBodies.length).toBe(2);
  const body = api.questionBodies[1];
  expect(body.chunk_ids).toEqual([GUIDE_SOURCE.chunk_id]);
  expect(body.filenames).toBeUndefined();
  expect(body.tags ?? []).toEqual([]);
});

test('stricter answers from Settings ride with the next question', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  await open(page);
  await page.getByTestId('open-settings').click();
  await page.getByTestId('settings-passages-only').check();
  await page.getByTestId('settings-strong-only').check();
  await page.getByTestId('settings-save').click();
  await expect(page.getByTestId('settings-modal')).toBeHidden();

  await ask(page, 'What retrieval strategy does DocRAG use?');
  expect(api.questionBodies.at(-1)).toMatchObject({ passages_only: true, strong_matches_only: true });
});

test('cited sources export as CSV, and a number key opens that source', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  api.stream = answerStream();
  await open(page);
  await ask(page);

  await page.keyboard.press('ControlOrMeta+k');
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('option', { name: /Export cited sources/ }).click(),
  ]);
  expect(download.suggestedFilename()).toMatch(/sources\.csv$/);
  const csv = readFileSync((await download.path())!, 'utf8');
  expect(csv.split('\r\n')[0]).toBe('conversation,question,answer,source,filename,page,section,chunk_id,quote');
  expect(csv).toContain(`,1,1,guide.md,p.1,Retrieval,${GUIDE_SOURCE.chunk_id},`);

  await page.locator('body').click({ position: { x: 5, y: 5 } });
  await page.keyboard.press('1');
  await expect(page.getByRole('dialog').getByTestId('viewer-chunk').first()).toBeVisible();
});

test('a scope is saved under a name and picked again', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  await open(page);
  await page.getByTestId('composer-scope-button').click();
  await expect(page.getByTestId('scope-preset-save')).toBeDisabled();
  await page.getByTestId('scope-item').filter({ hasText: 'notes.md' }).click();
  await page.getByTestId('scope-preset-save').click();
  await page.getByTestId('scope-preset-name').fill('Notes only');
  await page.getByTestId('scope-preset-confirm').click();
  await expect(page.getByTestId('scope-preset')).toHaveText(/Notes only/);

  await page.getByTestId('scope-all').check();
  await expect(page.getByTestId('composer-scope-button')).not.toContainText('notes.md');
  await page.getByTestId('scope-preset').getByRole('button', { name: 'Notes only', exact: true }).click();
  await expect(page.getByTestId('composer-scope-button')).toContainText('notes.md');
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

test('a cut-off answer is continued in place', async ({ page, api }) => {
  api.stream = answerStream('DocRAG uses hybrid', { truncated: true, trace_id: 'trace-cut' });
  api.continueStream = answerStream('DocRAG uses hybrid retrieval [1].', { trace_id: 'trace-2' });
  await open(page);
  await ask(page);
  await expect(page.getByText('The model reached its output limit')).toBeVisible();
  await page.getByTestId('chat-message-continue').click();
  await expect(page.getByText('DocRAG uses hybrid retrieval')).toBeVisible();
  expect(api.continueBodies.at(-1)).toEqual({ trace_id: 'trace-cut' });
  await expect(page.getByTestId('chat-message-continue')).toHaveCount(0);
  // One answer, replaced in place (the cut-off one kept as its earlier version).
  await expect(page.getByTestId('chat-message-regenerate')).toHaveCount(1);
});

test('a follow-up says what it was searched as', async ({ page, api }) => {
  api.stream = answerStream('Twenty-eight days [1].', { retrieval_query: 'How many vacation days do staff get?' });
  await open(page);
  await ask(page, 'And for full-time staff?');
  await expect(page.getByTestId('chat-message-searched-as')).toContainText(
    'Searched as: How many vacation days do staff get?',
  );
});
