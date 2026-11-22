// Pass 20: folder passage links, Continue without the raw join, notes across Regenerate.
import { GUIDE_SOURCE, TIMINGS, expect, sse, test } from '../fixtures';
import { ask, open } from '../helpers';

test('a passage link to a document in a folder opens it', async ({ page, api }) => {
  api.documents = ['reports/2024/q1.md'];
  await open(page);
  await page.evaluate(() => {
    window.location.hash = 'd=reports%2F2024%2Fq1.md';
  });
  await expect(page.getByRole('dialog')).toContainText('q1.md');
});

test('a continued answer shows the joined text, never the raw pieces', async ({ page, api }) => {
  api.stream = sse(
    { type: 'sources', sources: [GUIDE_SOURCE], trace_id: 'trace-cut' },
    { type: 'delta', text: 'Economy, but' },
    { type: 'done', answer: 'Economy, but', sources: [GUIDE_SOURCE], timings: TIMINGS, trace_id: 'trace-cut', truncated: true },
  );
  api.continueStream = sse(
    { type: 'sources', sources: [GUIDE_SOURCE], trace_id: 'trace-2' },
    { type: 'delta', text: 'business' },
    { type: 'delta', text: ' class [1].' },
    { type: 'done', answer: 'Economy, but business class [1].', sources: [GUIDE_SOURCE], timings: TIMINGS, trace_id: 'trace-2' },
  );
  await open(page);
  await ask(page);
  await page.getByTestId('chat-message-continue').click();
  const answer = page.locator('[data-role="assistant"]').last();
  await expect(answer).toContainText('Economy, but business class');
  await expect(answer).not.toContainText('butbusiness');
  await expect(page.locator('[data-role="assistant"]')).toHaveCount(1);
});

test('on a wide window the source viewer docks beside the conversation', async ({ page, api }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  api.stream = sse(
    { type: 'sources', sources: [GUIDE_SOURCE], trace_id: 'trace-1' },
    { type: 'delta', text: 'DocRAG uses hybrid retrieval [1].' },
    { type: 'done', answer: 'DocRAG uses hybrid retrieval [1].', sources: [GUIDE_SOURCE], timings: TIMINGS, trace_id: 'trace-1' },
  );
  await open(page);
  await ask(page);
  await page.getByTestId('citation-card').first().click();
  await page.getByTestId('viewer-dock').click();
  const viewer = page.getByTestId('document-viewer');
  await expect(viewer).toHaveAttribute('role', 'complementary');
  await expect(page.getByTestId('viewer-backdrop')).toHaveCount(0);
  // The conversation stays usable beside it.
  await page.getByTestId('question-textarea').fill('Another question');
  await expect(page.getByTestId('question-textarea')).toHaveValue('Another question');
  // Kept for the next document opened.
  await page.getByTestId('viewer-close').click();
  await page.getByTestId('citation-card').first().click();
  await expect(page.getByTestId('document-viewer')).toHaveAttribute('role', 'complementary');
});
