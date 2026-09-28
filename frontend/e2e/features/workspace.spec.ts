import { expect, test } from '../fixtures';

const NOTES = { name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('Roadmap notes.') };

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('question-textarea')).toBeEnabled();
});

async function startUpload(page: import('@playwright/test').Page) {
  await page.getByRole('button', { name: 'Corpus' }).click();
  await page.getByTestId('upload-input').setInputFiles(NOTES);
  await page.getByTestId('upload-button').click();
}

test('typing in Settings survives the re-renders an upload poll causes', async ({ page, api }) => {
  // Keep the job in flight for a few seconds of polling while the user types.
  api.jobSteps = [
    ...Array.from({ length: 4 }, () => ({ status: 200, body: api.jobBody('embedding') })),
    { status: 200, body: api.jobBody('done') },
  ];
  await startUpload(page);

  await page.keyboard.press('ControlOrMeta+Comma');
  const key = page.getByTestId('settings-api-key');
  await key.click();
  await key.pressSequentially('abcdefgh', { delay: 350 });

  await expect(key).toHaveValue('abcdefgh');
  await expect(key).toBeFocused();
});

test('every shortcut the palette advertises is bound, and none stack over Settings', async ({ page }) => {
  const themeToggle = page.getByTestId('theme-toggle');
  const before = await themeToggle.getAttribute('aria-label');
  await page.keyboard.press('ControlOrMeta+j');
  await expect(themeToggle).not.toHaveAttribute('aria-label', before ?? '');

  await page.keyboard.press('ControlOrMeta+k');
  await expect(page.getByRole('dialog')).toHaveCount(1);
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).toHaveCount(0);

  await page.keyboard.press('ControlOrMeta+Comma');
  await expect(page.getByTestId('settings-modal')).toBeVisible();
  await page.keyboard.press('ControlOrMeta+k');
  await expect(page.getByRole('dialog')).toHaveCount(1);
});

test('an upload the server rejects shows its reason', async ({ page, api }) => {
  api.uploadResponse = {
    status: 400,
    body: { detail: "Unsupported document type '.zip'." },
  };
  await startUpload(page);

  await expect(page.getByText("Unsupported document type '.zip'.")).toBeVisible();
});

test('a transient status-poll failure does not fail an upload that keeps indexing', async ({ page, api }) => {
  api.jobSteps = [
    { status: 503, body: { detail: 'busy' } },
    { status: 200, body: api.jobBody('embedding') },
    { status: 200, body: api.jobBody('done') },
  ];
  await startUpload(page);

  await expect(page.getByTestId('corpus-panel-item').filter({ hasText: 'notes.txt' })).toBeVisible({
    timeout: 15_000,
  });
  await expect(page.getByText('busy')).toHaveCount(0);
});
