import type { Page } from '@playwright/test';
import { expect } from './fixtures';

export const DEFAULT_QUESTION = 'What retrieval strategy does DocRAG use?';

/** Loads the app and waits until a question can be typed. */
export async function open(page: Page) {
  await page.goto('/');
  await expect(page.getByTestId('question-textarea')).toBeEnabled();
}

/** Asks with Enter and waits for the answer to finish. */
export async function ask(page: Page, question = DEFAULT_QUESTION) {
  const box = page.getByTestId('question-textarea');
  await box.fill(question);
  await box.press('Enter');
  await expect(page.getByTestId('question-submit')).toHaveText('Ask');
}

/** Asks with the submit button and waits for the answer to finish. */
export async function askWithButton(page: Page, question: string) {
  await page.getByTestId('question-textarea').fill(question);
  await page.getByTestId('question-submit').click();
  await expect(page.getByTestId('question-submit')).toHaveText('Ask');
}
