import { expect, test } from '../fixtures';

test('@smoke workspace loads with the composer ready', async ({ page, api }) => {
  api.documents = ['guide.md'];
  await page.goto('/');

  await expect(page.getByRole('img', { name: 'DocRAG' })).toBeVisible();
  await expect(page.getByTestId('question-textarea')).toBeEnabled();
});

test('@smoke an unreachable API gets a clear message, not a blank page', async ({ page, consoleErrors }) => {
  // No `api` fixture: every API call fails at the network layer.
  await page.route(
    (url) => /^\/(health|config|documents|traces)(\/|$)/.test(url.pathname),
    (route) => route.abort('connectionrefused'),
  );
  await page.goto('/');

  await expect(page.getByRole('alert')).toContainText('Cannot reach the DocRAG API');
  // The refused connections are this scenario; anything else still fails the test.
  expect(consoleErrors.every((error) => error.includes('net::ERR_CONNECTION_REFUSED'))).toBe(true);
  consoleErrors.splice(0);
});
