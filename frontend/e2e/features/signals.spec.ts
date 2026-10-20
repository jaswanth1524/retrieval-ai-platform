import { answerStream, expect, test } from '../fixtures';

test.beforeEach(async ({ api }) => {
  api.documents = ['guide.md', 'notes.md'];
});

test('an index built for another embedding setup is reported before any question', async ({
  page,
  api,
}) => {
  api.readiness = {
    index_compatible: false,
    index_detail:
      "Collection 'docrag_documents' uses embedding model tag 'old'. Either restore the " +
      'previous embedding settings, or delete every document and upload them again.',
    originals_kept: false,
  };
  await page.goto('/');
  const notice = page.getByTestId('index-mismatch');
  await expect(notice).toContainText('questions are refused');
  await expect(notice).toContainText('delete every document and upload them again');
  // Without originals a backup can't rebuild it, so the app's backup steps aren't offered.
  await expect(notice).not.toContainText('Download a backup');
});

test('a busy server is waited out with a countdown, then asked again', async ({ page, api }) => {
  let refusals = 1;
  api.stream = async (route) => {
    if (refusals > 0) {
      refusals -= 1;
      api.intentionalFailures.add(route.request().url());
      return route.fulfill({
        status: 429,
        contentType: 'application/json',
        headers: { 'Retry-After': '1' },
        body: JSON.stringify({ detail: 'The server is answering as many questions as it can.' }),
      });
    }
    return route.fulfill({ status: 200, contentType: 'text/event-stream', body: answerStream() });
  };
  await page.goto('/');
  const box = page.getByTestId('question-textarea');
  await expect(box).toBeEnabled();
  await box.fill('What retrieval strategy does DocRAG use?');
  await box.press('Enter');
  await expect(page.getByTestId('streaming-skeleton')).toContainText(/asking again in \ds/);
  await expect(page.getByText('DocRAG uses hybrid retrieval')).toBeVisible();
  expect(api.questionBodies).toHaveLength(2);
  await expect(page.getByTestId('chat-message').filter({ hasText: 'Something went wrong' })).toHaveCount(0);
});

test('a question stopped before any answer can be asked again', async ({ page, api }) => {
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  api.stream = async (route) => {
    await held;
    await route.fulfill({ status: 200, contentType: 'text/event-stream', body: answerStream() });
  };
  await page.goto('/');
  const box = page.getByTestId('question-textarea');
  await expect(box).toBeEnabled();
  await box.fill('What retrieval strategy does DocRAG use?');
  await box.press('Enter');
  await page.getByTestId('question-submit').click(); // Stop
  release();
  const again = page.getByTestId('chat-message-ask-again');
  await expect(again).toBeVisible();
  api.stream = answerStream();
  await again.click();
  await expect(page.getByText('DocRAG uses hybrid retrieval')).toBeVisible();
});

test('? lists the keyboard shortcuts', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('question-textarea')).toBeEnabled();
  await page.locator('body').click({ position: { x: 5, y: 5 } });
  await page.keyboard.press('?');
  const dialog = page.getByTestId('shortcuts-dialog');
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText('Search passages');
  // An overlay, whether or not Settings (whose stylesheet is a lazy chunk) was opened:
  // borrowing its backdrop class laid the dialog out as a page column until then.
  await expect(dialog.locator('xpath=..')).toHaveCSS('position', 'fixed');
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
});

test('indexing started elsewhere shows in the corpus panel', async ({ page, api }) => {
  api.activeJobs = [
    {
      job_id: 'w1',
      filename: 'from-the-watched-folder.md',
      state: 'embedding',
      chunks_total: 4,
      chunks_done: 1,
      error: null,
      result: null,
    },
  ];
  await page.goto('/');
  await page.getByTestId('rail-corpus').click();
  await expect(page.getByTestId('active-jobs')).toContainText('from-the-watched-folder.md');
});
