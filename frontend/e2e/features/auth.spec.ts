import { expect, test } from '../fixtures';

test('a keyed server asks for the key, and saving it unlocks the corpus', async ({ page, api }) => {
  api.apiKey = 'secret-key';
  await page.goto('/');

  await expect(page.getByTestId('context-panel-footer')).toContainText('api key required');

  await page.getByTestId('open-settings').click();
  await page.getByTestId('settings-api-key').fill('secret-key');
  await page.getByTestId('settings-save').click();

  await expect(page.getByTestId('context-panel-footer')).toContainText('api ok');
  await expect(page.getByTestId('question-textarea')).toBeEnabled();
});
