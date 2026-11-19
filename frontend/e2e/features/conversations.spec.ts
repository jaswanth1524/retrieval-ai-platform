// The conversation list: delete and restore, links, labels, other tabs.
import { answerStream, expect, test } from '../fixtures';
import { open, ask } from '../helpers';

test('a deleted conversation comes back with Undo', async ({ page, api }) => {
  api.stream = answerStream();
  await open(page);
  await ask(page);
  await page.getByTestId('rail-chat').click();
  const title = await page.getByTestId('conversation-select').first().getAttribute('title');
  await page.getByRole('button', { name: `Delete ${title}` }).click();
  await page.getByTestId('conversation-delete-confirm').click();
  await expect(page.locator(`[data-testid="conversation-select"][title="${title}"]`)).toHaveCount(0);
  await page.getByTestId('toast-action').click();
  // Back in the list, and open again.
  await expect(page.locator(`[data-testid="conversation-select"][title="${title}"]`)).toHaveAttribute(
    'aria-current',
    'true',
  );
});

test('a link to an answer opens its conversation again', async ({ page, api }) => {
  api.stream = answerStream();
  await open(page);
  await ask(page);
  const answerId = await page.locator('[data-role="assistant"]').last().getAttribute('data-turn-id');
  const conversation = await page.evaluate(() => {
    const stored = JSON.parse(localStorage.getItem('docrag-chat-history') ?? '{}');
    return stored.activeConversationId as string;
  });
  // Another conversation is open when the link is followed.
  await page.getByTestId('rail-chat').click();
  await page.getByTestId('context-panel-action').click();
  await expect(page.locator(`[data-turn-id="${answerId}"]`)).toHaveCount(0);
  await page.evaluate((hash) => {
    window.location.hash = hash;
  }, `c=${conversation}&t=${answerId}`);
  await expect(page.locator(`[data-turn-id="${answerId}"]`)).toBeVisible();
});

test('conversations are labelled and filtered by label', async ({ page, api }) => {
  api.stream = answerStream();
  await open(page);
  await ask(page);
  await page.getByTestId('rail-chat').click();
  await page.getByTestId('conversation-label').first().click();
  await page.getByTestId('conversation-labels-input').fill('retrieval');
  await page.getByTestId('conversation-labels-input').press('Enter');
  await expect(page.getByTestId('conversation-labels')).toHaveText(/#retrieval/);
  await page.getByTestId('conversation-label-filter').click();
  await expect(page.getByTestId('conversation-select')).toHaveCount(1);
});

test('a label is renamed everywhere and its conversations exported', async ({ page, api }) => {
  api.stream = answerStream();
  await open(page);
  await ask(page);
  await page.getByTestId('rail-chat').click();
  await page.getByTestId('conversation-label').first().click();
  await page.getByTestId('conversation-labels-input').fill('retrieval');
  await page.getByTestId('conversation-labels-input').press('Enter');
  await page.getByTestId('conversation-label-filter').click();
  await page.getByTestId('conversation-label-rename').click();
  await page.getByTestId('conversation-label-rename-input').fill('search');
  await page.getByTestId('conversation-label-rename-input').press('Enter');
  await expect(page.getByTestId('conversation-labels')).toHaveText(/#search/);
  const download = page.waitForEvent('download');
  await page.getByTestId('conversation-label-export').click();
  expect((await download).suggestedFilename()).toMatch(/^docrag-search-/);
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
