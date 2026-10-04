import { answerStream, expect, GUIDE_SOURCE, sse, test } from '../fixtures';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('question-textarea')).toBeEnabled();
});

async function ask(page: import('@playwright/test').Page, question: string) {
  const box = page.getByTestId('question-textarea');
  await box.fill(question);
  await box.press('Enter');
}

test('streams an answer with its citation and hands focus back to the composer', async ({ page, api }) => {
  api.stream = answerStream('DocRAG uses hybrid retrieval [1].');
  await ask(page, 'What retrieval strategy does DocRAG use?');

  await expect(page.getByText('DocRAG uses hybrid retrieval')).toBeVisible();
  await expect(page.getByText(GUIDE_SOURCE.filename).first()).toBeVisible();
  await expect(page.getByTestId('question-textarea')).toBeEnabled();
  await expect(page.getByTestId('question-textarea')).toBeFocused();
});

test('Stop mid-stream keeps the question and shows no error', async ({ page, api }) => {
  let aborted!: () => void;
  const requestAborted = new Promise<void>((resolve) => (aborted = resolve));
  // Hold the stream open: the server is still "retrieving" when Stop is pressed.
  api.stream = async (route) => {
    await requestAborted;
    await route.abort().catch(() => undefined);
  };
  page.on('requestfailed', (request) => {
    if (request.url().endsWith('/questions/stream')) aborted();
  });

  await ask(page, 'How do uploads report progress?');
  await expect(page.getByTestId('streaming-skeleton')).toContainText('retrieving…');
  await page.getByRole('button', { name: 'Stop' }).click();

  await expect(page.getByTestId('question-textarea')).toBeEnabled();
  await expect(page.getByText('How do uploads report progress?').first()).toBeVisible();
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('a stream that closes without finishing is an error with a retry, not a silent answer', async ({ page, api }) => {
  api.stream = sse(
    { type: 'stage', stage: 'searching' },
    { type: 'sources', sources: [GUIDE_SOURCE], trace_id: 'trace-2' },
    { type: 'delta', text: 'half an ans' },
  );
  await ask(page, 'What retrieval strategy does DocRAG use?');

  const error = page.getByRole('alert');
  await expect(error).toContainText('The answer stream ended before the server finished.');
  await expect(error.getByRole('button', { name: 'Retry' })).toBeVisible();

  // Retrying against a healthy stream recovers.
  api.stream = answerStream('Recovered answer [1].');
  await error.getByRole('button', { name: 'Retry' }).click();
  await expect(page.getByText('Recovered answer')).toBeVisible();
});
