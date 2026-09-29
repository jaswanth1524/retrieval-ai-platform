import { expect, test } from '../fixtures';

test('a dialog whose code chunk is gone (stale tab after a redeploy) does not blank the app', async ({
  page,
  consoleErrors,
}) => {
  await page.goto('/');
  await expect(page.getByTestId('question-textarea')).toBeEnabled();
  // The new server no longer has the old build's hashed chunk names.
  await page.route(/\/assets\/SettingsModal-[^/]+\.js$/, (route) => route.fulfill({ status: 404 }));

  await page.getByTestId('open-settings').click();

  await expect(page.getByTestId('lazy-chunk-error')).toContainText('DocRAG was updated');
  await expect(page.getByTestId('question-textarea')).toBeVisible();
  await page.getByTestId('lazy-chunk-error').getByRole('button', { name: 'Dismiss' }).click();
  await expect(page.getByTestId('lazy-chunk-error')).toHaveCount(0);

  // The failed chunk fetch and React's report of the caught error are this scenario.
  expect(consoleErrors.length).toBeGreaterThan(0);
  consoleErrors.splice(0);
});
