// Pass 22: a flagged sentence leads to its self-check line, Tab through a template's
// blanks, two Rechecks at once.
import { answerStream, expect, test } from '../fixtures';
import { ask, open } from '../helpers';

test('a flagged sentence opens its line in the self-check', async ({ page, api }) => {
  api.stream = answerStream('DocRAG uses hybrid retrieval [1]. The moon is made of cheese [1].', {
    verification: [
      { sentence: 'DocRAG uses hybrid retrieval [1].', status: 'supported', source_number: 1 },
      { sentence: 'The moon is made of cheese [1].', status: 'unsupported' },
    ],
  });
  await open(page);
  await ask(page);
  const mark = page.getByTestId('answer-unbacked');
  await expect(mark).toContainText('The moon is made of cheese');
  await mark.getByRole('button').click();
  const check = page.getByTestId('answer-self-check');
  await expect(check).toHaveAttribute('open', '');
  await expect(check.locator('li:focus')).toContainText('The moon is made of cheese');
});

test('Tab goes from one blank of a template to the next', async ({ page }) => {
  await open(page);
  const box = page.getByTestId('question-textarea');
  await box.fill('Compare {first} with {second}.');
  await page.getByTestId('composer-templates-button').click();
  await page.getByTestId('template-name').fill('Compare');
  await page.getByTestId('template-save').click();
  await box.fill('');
  await page.getByTestId('composer-templates-button').click();
  await page.getByTestId('template-insert').click();
  await page.keyboard.type('leave');
  await page.keyboard.press('Tab');
  await page.keyboard.type('expenses');
  await expect(box).toHaveValue('Compare leave with expenses.');
});

test('a template picked with a question typed goes after it', async ({ page }) => {
  await open(page);
  const box = page.getByTestId('question-textarea');
  await box.fill('When does {thing} start?');
  await page.getByTestId('composer-templates-button').click();
  await page.getByTestId('template-name').fill('Start');
  await page.getByTestId('template-save').click();
  await box.fill('Something typed already');
  await page.getByTestId('composer-templates-button').click();
  await page.getByTestId('template-insert').click();
  await expect(box).toHaveValue('Something typed already\n\nWhen does {thing} start?');
});
