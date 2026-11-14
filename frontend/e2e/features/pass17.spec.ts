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

test('a deleted conversation can be restored from Recently deleted after its toast', async ({ page, api }) => {
  api.stream = answerStream();
  await open(page);
  await ask(page);
  await page.getByTestId('rail-chat').click();
  const title = await page.getByTestId('conversation-select').first().getAttribute('title');
  await page.getByRole('button', { name: `Delete ${title}` }).click();
  await page.getByTestId('conversation-delete-confirm').click();
  await expect(page.locator(`[data-testid="conversation-select"][title="${title}"]`)).toHaveCount(0);
  await page.getByText(/Recently deleted \(1\)/).click();
  await page.getByTestId('recently-deleted-restore').click();
  await expect(page.locator(`[data-testid="conversation-select"][title="${title}"]`)).toHaveCount(1);
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

test('a rename in another tab survives this tab leaving mid-answer', async ({ page, api, context }) => {
  api.stream = answerStream('First answer [1].');
  await page.goto('/');
  const box = page.getByTestId('question-textarea');
  await expect(box).toBeEnabled();
  await box.fill('First question about retrieval?');
  await box.press('Enter');
  await expect(page.getByTestId('question-submit')).toHaveText('Ask');

  // A second conversation whose answer never finishes.
  await page.keyboard.press('ControlOrMeta+Shift+O');
  let release: () => void = () => {};
  const held = new Promise<void>((resolve) => (release = resolve));
  api.stream = async (route) => {
    await held;
    await route.fulfill({ status: 200, contentType: 'text/event-stream', body: answerStream('Late [1].') });
  };
  await box.fill('Second question, held open?');
  await box.press('Enter');
  await expect(page.getByTestId('question-submit')).not.toHaveText('Ask');

  // Another tab renames the first conversation.
  const other = await context.newPage();
  await api.install(other);
  await other.goto('/');
  await expect(other.getByTestId('question-textarea')).toBeEnabled();
  await other.getByTestId('rail-chat').click();
  const row = other.locator('[data-testid="conversation-select"]', { hasText: 'First question' });
  await row.dblclick();
  const input = other.getByRole('textbox', { name: /^Rename / });
  await input.fill('Renamed elsewhere');
  await input.press('Enter');
  await expect(other.locator('[data-testid="conversation-select"][title="Renamed elsewhere"]')).toHaveCount(1);
  await other.waitForTimeout(600); // the coalesced IndexedDB write

  // This tab leaves mid-answer: its snapshot must not undo the rename.
  await page.reload();
  release();
  await expect(page.getByTestId('question-textarea')).toBeEnabled();
  await page.getByTestId('rail-chat').click();
  await expect(page.locator('[data-testid="conversation-select"][title="Renamed elsewhere"]')).toHaveCount(1);
  await other.reload();
  await other.getByTestId('rail-chat').click();
  await expect(other.locator('[data-testid="conversation-select"][title="Renamed elsewhere"]')).toHaveCount(1);
});
