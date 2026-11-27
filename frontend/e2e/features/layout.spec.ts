// What jsdom can't measure: a source card's file name and a saved answer's buttons
// stay readable, on a phone and on a desktop (found live in pass 23).
import { expect, test } from '../fixtures';
import { open } from '../helpers';

const FILES = ['leave.txt', 'travel_policy_2019_archived.md'];
const ASKED_AT = 1_700_000_000_000;

for (const viewport of [
  { width: 390, height: 844 },
  { width: 1280, height: 800 },
]) {
  test(`source names and saved buttons fit at ${viewport.width} px`, async ({ page, api }) => {
    api.documents = FILES;
    // Uploaded after the answer: each card also says the document changed since.
    api.uploadedAts = Object.fromEntries(FILES.map((name) => [name, ASKED_AT / 1000 + 3600]));
    const sources = FILES.map((filename, index) => ({
      source_number: index + 1,
      filename,
      page: 1,
      section: 'Leave',
      chunk_id: `c${index}`,
      text: 'Staff get 28 days.',
      rerank_score: 0.82,
    }));
    const turn = { timestamp: ASKED_AT, timings: null, traceId: null };
    await page.addInitScript(
      (history) => localStorage.setItem('docrag-chat-history', JSON.stringify(history)),
      {
        version: 2,
        activeConversationId: 'c',
        conversations: [
          {
            id: 'c',
            title: 'Leave',
            createdAt: ASKED_AT,
            updatedAt: ASKED_AT,
            turns: [
              { ...turn, id: 'q', role: 'user', content: 'How much leave?', sources: [] },
              { ...turn, id: 'a', role: 'assistant', content: 'Staff get 28 days [1][2].', sources, saved: true },
            ],
          },
        ],
      },
    );
    await page.setViewportSize(viewport);
    await open(page);

    await expect(page.getByTestId('citation-changed').first()).toBeVisible();
    for (const name of FILES) {
      const file = page.locator('.citation-card__file', { hasText: name });
      const cut = await file.evaluate((el) => el.scrollWidth - el.clientWidth);
      // A long name may be shortened with an ellipsis, never to a few letters.
      const shown = await file.evaluate((el) => el.clientWidth);
      expect(cut <= 0 || shown >= 160, `${name}: ${shown} px shown, ${cut} px cut`).toBe(true);
    }

    await page.getByTestId('rail-saved').click();
    const row = page.getByTestId('saved-row');
    await expect(row).toBeVisible();
    for (const id of ['saved-open', 'saved-recheck', 'saved-ask-again', 'saved-note-edit', 'saved-unsave']) {
      const button = page.getByTestId(id);
      // On top at both of its sides: inside the panel, not cut off at its edge.
      await expect
        .poll(() =>
          button.evaluate((el) => {
            const box = el.getBoundingClientRect();
            return [box.left + 2, box.right - 2].every((x) =>
              el.contains(document.elementFromPoint(x, box.top + box.height / 2)),
            );
          }),
        )
        .toBe(true);
      // And its label on one line.
      expect(await button.evaluate((el) => el.getBoundingClientRect().height)).toBeLessThan(30);
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  });
}
