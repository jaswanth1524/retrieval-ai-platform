// Shared answers (SHARES_ENABLED): share by link, open the link, delete the share.
import { answerStream, expect, test } from '../fixtures';
import { ask, open } from '../helpers';

test('an answer is shared by link, opened read-only from it, and deleted', async ({
  page,
  api,
  context,
}) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  api.config = { shares_enabled: true };
  api.stream = answerStream('DocRAG uses hybrid retrieval [1].');
  await open(page);
  await ask(page, 'What retrieval strategy does DocRAG use?');
  await page.getByTestId('chat-message-share-link').click();
  await expect(page.getByText('Share link copied')).toBeVisible();
  expect(api.shareBodies).toHaveLength(1);
  expect(api.shareBodies[0]).toMatchObject({
    title: 'What retrieval strategy does DocRAG use?',
    turns: [
      {
        question: 'What retrieval strategy does DocRAG use?',
        answer: 'DocRAG uses hybrid retrieval [1].',
        answer_status: 'answered',
      },
    ],
  });
  const link = await page.evaluate(() => navigator.clipboard.readText());
  expect(link).toMatch(/#s=share000001$/);

  await page.goto(link);
  const view = page.getByTestId('shared-view');
  await expect(view).toBeVisible();
  await expect(view).toContainText('DocRAG uses hybrid retrieval');
  await expect(page.getByTestId('shared-view-meta')).toContainText('Shared by bob');
  await view.getByRole('button', { name: 'Close' }).click();
  await expect(view).toBeHidden();

  // Opened again and kept as a conversation of this browser.
  await page.goto(link);
  await page.getByTestId('shared-view-import').click();
  await expect(page.getByText('Conversation imported')).toBeVisible();
  await expect(view).toBeHidden();
  await page.getByTestId('rail-chat').click();
  await expect(page.getByTestId('conversation-item').filter({ hasText: 'What retrieval strategy' })).toHaveCount(2);

  await page.keyboard.press('ControlOrMeta+Comma');
  await expect(page.getByTestId('shared-link')).toHaveCount(1);
  await page.getByTestId('shared-link-delete').click();
  await expect(page.getByTestId('shared-link')).toHaveCount(0);
  expect(Object.keys(api.shares)).toEqual([]);
});

test('a share link that opens nothing says so', async ({ page, api }) => {
  api.config = { shares_enabled: true };
  await page.goto('/#s=missing00000');
  await expect(page.getByTestId('shared-view-error')).toContainText('No such share');
});

test('with sharing off, no Share link button is offered', async ({ page, api }) => {
  api.stream = answerStream('DocRAG uses hybrid retrieval [1].');
  await open(page);
  await ask(page, 'What retrieval strategy does DocRAG use?');
  await expect(page.getByTestId('chat-message-link')).toBeVisible();
  await expect(page.getByTestId('chat-message-share-link')).toHaveCount(0);
});
