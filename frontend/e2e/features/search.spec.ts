import { expect, GUIDE_SOURCE, test } from '../fixtures';

test('searching passages lists ranked results and opens one in the document', async ({ page, api }) => {
  await page.goto('/');
  await page.getByTestId('rail-search').click();

  await page.getByTestId('search-input').fill('reciprocal rank fusion');
  await page.getByTestId('search-submit').click();

  await expect(page.getByTestId('search-result')).toHaveCount(1);
  await expect(page.getByTestId('search-result')).toContainText(GUIDE_SOURCE.filename);
  await expect(page.getByTestId('search-result')).toContainText('0.93');
  expect(api.searches).toEqual([{ query: 'reciprocal rank fusion' }]);

  await page.getByTestId('search-result').click();
  await expect(page.getByRole('dialog', { name: `Source: ${GUIDE_SOURCE.filename}` })).toBeVisible();
});

test('a busy server is said plainly, and the read-only key can still search', async ({ page, api }) => {
  api.access = 'read';
  api.searchResults = null;
  await page.goto('/');
  await page.getByTestId('rail-search').click();

  await page.getByTestId('search-input').fill('anything');
  await page.getByTestId('search-submit').click();

  await expect(page.getByTestId('search-error')).toContainText('busy answering questions');
});
