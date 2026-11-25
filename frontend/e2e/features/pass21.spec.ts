// Pass 21: an answer's self-check, Recheck, shortcuts with the viewer docked, templates,
// several conversations deleted at once.
import { GUIDE_SOURCE, TIMINGS, answerStream, expect, sse, test } from '../fixtures';
import { ask, open } from '../helpers';

test('sentences the cited passages don’t back are listed under the answer', async ({ page, api }) => {
  api.stream = answerStream('DocRAG uses hybrid retrieval [1]. The moon is cheese [1].', {
    verification: [
      { sentence: 'DocRAG uses hybrid retrieval [1].', status: 'supported', source_number: 1 },
      { sentence: 'The moon is cheese [1].', status: 'unsupported' },
    ],
  });
  await open(page);
  await ask(page);
  const check = page.getByTestId('answer-self-check');
  await expect(check).toContainText('One sentence isn’t backed by the passages it cites');
  await check.locator('summary').click();
  await expect(check).toContainText('“The moon is cheese [1].” — no passage says this');
});

test('a saved answer is rechecked against the documents now', async ({ page, api }) => {
  api.answerRecheck = {
    sources: [{ source_number: 1, filename: 'guide.md', chunk_id: 'c1', state: 'changed' }],
    sentences: [{ sentence: 'DocRAG uses hybrid retrieval [1].', status: 'numbers_differ' }],
    still_holds: false,
  };
  api.stream = answerStream();
  await open(page);
  await ask(page);
  await page.getByTestId('chat-message-save').click();
  await page.getByTestId('rail-saved').click();
  await page.getByTestId('saved-recheck').click();
  await expect(page.getByTestId('saved-recheck-result')).toHaveText(
    'Check it: 1 sentence no longer backed, 1 cited passage changed.',
  );
  expect(api.answerRecheckBodies[0]).toMatchObject({
    answer: 'DocRAG uses hybrid retrieval [1].',
    sources: [{ source_number: 1, filename: GUIDE_SOURCE.filename, chunk_id: GUIDE_SOURCE.chunk_id }],
  });
});

test('keyboard shortcuts keep working with the viewer docked', async ({ page, api }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  api.stream = sse(
    { type: 'sources', sources: [GUIDE_SOURCE], trace_id: 'trace-1' },
    { type: 'done', answer: 'DocRAG uses hybrid retrieval [1].', sources: [GUIDE_SOURCE], timings: TIMINGS, trace_id: 'trace-1' },
  );
  await open(page);
  await ask(page);
  await page.getByTestId('citation-card').first().click();
  await page.getByTestId('viewer-dock').click();
  await expect(page.getByTestId('document-viewer')).toHaveAttribute('role', 'complementary');
  await page.locator('body').click({ position: { x: 5, y: 5 } });
  await page.keyboard.press('?');
  await expect(page.getByTestId('shortcuts-dialog')).toBeVisible();
});

test('a question saved as a template is put back with its blank selected', async ({ page }) => {
  await open(page);
  const box = page.getByTestId('question-textarea');
  await box.fill('When does {thing} start?');
  await page.getByTestId('composer-templates-button').click();
  await page.getByTestId('template-name').fill('Start date');
  await page.getByTestId('template-save').click();
  await box.fill('');
  await page.getByTestId('composer-templates-button').click();
  await page.getByTestId('template-insert').click();
  await expect(box).toHaveValue('When does {thing} start?');
  await page.keyboard.type('the audit');
  await expect(box).toHaveValue('When does the audit start?');
});

test('several conversations are deleted at once and come back with one Undo', async ({ page, api }) => {
  await page.addInitScript(() => {
    if (sessionStorage.getItem('seeded')) return;
    sessionStorage.setItem('seeded', '1');
    const turn = (id: string, role: string, content: string) => ({
      id,
      role,
      content,
      sources: [],
      timestamp: 1,
      timings: null,
      traceId: null,
    });
    const conversation = (id: string, title: string, updatedAt: number) => ({
      id,
      title,
      createdAt: 1,
      updatedAt,
      turns: [turn(`${id}-q`, 'user', title), turn(`${id}-a`, 'assistant', 'An answer.')],
    });
    localStorage.setItem(
      'docrag-chat-history',
      JSON.stringify({
        version: 2,
        activeConversationId: 'one',
        conversations: [
          conversation('one', 'First question', 3),
          conversation('two', 'Second question', 2),
          conversation('three', 'Third question', 1),
        ],
      }),
    );
  });
  api.stream = answerStream();
  await open(page);
  await page.getByTestId('rail-chat').click();
  await page.getByTestId('conversation-select-mode').click();
  await page.getByRole('checkbox', { name: 'Select Second question' }).check();
  await page.getByRole('checkbox', { name: 'Select Third question' }).check();
  await page.getByTestId('conversation-bulk-delete').click();
  await page.getByTestId('conversation-bulk-delete-confirm').click();
  await expect(page.getByTestId('conversation-item')).toHaveCount(1);
  await page.getByTestId('toast-action').click();
  await expect(page.getByTestId('conversation-item')).toHaveCount(3);
});

test('making room for an import keeps what Recently deleted already holds', async ({ page }) => {
  await page.addInitScript(() => {
    if (sessionStorage.getItem('seeded')) return;
    sessionStorage.setItem('seeded', '1');
    const now = Date.now();
    const conversation = (id: string, updatedAt: number) => ({
      id,
      title: `Chat ${id}`,
      createdAt: 1,
      updatedAt,
      turns: [
        { id: `${id}-q`, role: 'user', content: `Question ${id}`, sources: [], timestamp: 1, timings: null, traceId: null },
        { id: `${id}-a`, role: 'assistant', content: 'An answer.', sources: [], timestamp: 1, timings: null, traceId: null },
      ],
    });
    localStorage.setItem(
      'docrag-chat-history',
      JSON.stringify({
        version: 2,
        activeConversationId: 'h0',
        conversations: Array.from({ length: 100 }, (_, i) => conversation(`h${i}`, now - i * 1000)),
      }),
    );
    localStorage.setItem(
      'docrag-trash',
      JSON.stringify(
        Array.from({ length: 18 }, (_, i) => ({ conversation: conversation(`t${i}`, 1), deletedAt: now - 5000 })),
      ),
    );
  });
  await open(page);
  let asked = '';
  page.once('dialog', (dialog) => {
    asked = dialog.message();
    void dialog.accept();
  });
  const imported = Array.from({ length: 5 }, (_, i) => ({
    title: `Imported ${i}`,
    turns: [
      { id: `i${i}-q`, role: 'user', content: `Imported question ${i}`, sources: [], timestamp: 1, timings: null, traceId: null },
      { id: `i${i}-a`, role: 'assistant', content: 'Imported answer.', sources: [], timestamp: 1, timings: null, traceId: null },
    ],
  }));
  await page.getByTestId('import-conversation-input').setInputFiles({
    name: 'chats.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify({ docrag_conversations: imported })),
  });
  await expect.poll(() => asked).toContain('2 of them go to Recently deleted, the rest are deleted for good');
  const trash = await page.evaluate(() => JSON.parse(localStorage.getItem('docrag-trash') ?? '[]'));
  const ids: string[] = trash.map((entry: { conversation: { id: string } }) => entry.conversation.id);
  expect(ids).toHaveLength(20);
  for (let i = 0; i < 18; i += 1) expect(ids).toContain(`t${i}`);
});

test('on a phone the conversation header fits, workspace chooser and all', async ({ page, api }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  api.config = { workspaces_enabled: true };
  api.workspaceNames = ['legal'];
  api.stream = answerStream();
  await open(page);
  await ask(page);
  const header = page.locator('.chat-header');
  const { scrollWidth, clientWidth } = await header.evaluate((el) => ({
    scrollWidth: el.scrollWidth,
    clientWidth: el.clientWidth,
  }));
  expect(scrollWidth).toBeLessThanOrEqual(clientWidth);
  const engineer = page.getByTestId('mode-engineer');
  await expect(engineer).toHaveAccessibleName('Engineer');
  const box = await engineer.boundingBox();
  expect(box && box.x + box.width).toBeLessThanOrEqual(390);
});
