// The Asked questions panel and answers flagged for review.
import { readFileSync } from 'node:fs';
import { answerStream, expect, test } from '../fixtures';
import { open, ask } from '../helpers';

test('the asked questions panel shows the answer rate and the most asked', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  api.config = { query_log_enabled: true };
  const now = Date.now() / 1000;
  const row = (question: string, status: string) => ({
    created_at: now,
    question,
    status,
    top_score: 0.4,
    mode: 'stream',
    trace_id: null,
  });
  api.unanswered = [
    row('Who approves travel?', 'insufficient_context'),
    row('How many leave days?', 'ok'),
    row('how many leave days?', 'ok'),
  ];
  await page.goto('/');
  await page.getByTestId('rail-unanswered').click();
  await expect(page.getByTestId('questions-summary')).toContainText('3 asked · 67% answered');
  await expect(page.getByTestId('unanswered-row')).toHaveCount(1);
  await page.getByTestId('questions-view-repeated').click();
  await expect(page.getByTestId('unanswered-row')).toContainText('×2');
});

test('the asked questions download as CSV, formulas defused', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  api.config = { query_log_enabled: true };
  api.unanswered = [
    {
      created_at: Date.now() / 1000,
      question: '=cmd|calc',
      status: 'insufficient_context',
      top_score: 0.05,
      mode: 'stream',
      trace_id: null,
    },
  ];
  await open(page);
  await page.getByTestId('rail-unanswered').click();
  const download = page.waitForEvent('download');
  await page.getByTestId('questions-export').click();
  const csv = readFileSync((await (await download).path())!, 'utf8');
  expect(csv.split('\r\n')[0]).toBe(
    'question,times_asked,times_not_answered,last_asked,last_status,best_match,mode',
  );
  expect(csv).toContain("'=cmd|calc,1,1,");
});

test('unanswered questions are searched again from the Asked questions panel', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  api.config = { query_log_enabled: true };
  api.unanswered = [
    {
      created_at: Date.now() / 1000,
      question: 'Is there a gym?',
      status: 'insufficient_context',
      top_score: 0.02,
      mode: 'stream',
      trace_id: null,
    },
  ];
  api.recheck = {
    checked: 1,
    covered_now: 1,
    items: [{ question: 'Is there a gym?', status: 'insufficient_context', covered_now: true, top_filename: 'gym.md' }],
  };
  await open(page);
  await page.getByTestId('rail-unanswered').click();
  await page.getByTestId('questions-recheck').click();
  await expect(page.getByTestId('questions-recheck-result')).toContainText('1 of 1 now find a passage');
  await expect(page.getByTestId('questions-recheck-covered')).toContainText('gym.md');
  expect(api.rechecks).toBe(1);
});

test('an answer worth a second look is counted on the rail and marked reviewed', async ({ page, api }) => {
  api.stream = answerStream('The documents do not say whether there is a gym.', { answer_status: 'insufficient_context', sources: [] });
  await open(page);
  await ask(page, 'Is there a gym?');
  await expect(page.getByTestId('rail-badge-review')).toHaveText('1');
  await page.getByTestId('rail-review').click();
  await page.getByTestId('review-mark').click();
  await expect(page.getByTestId('review-empty')).toContainText('has been reviewed');
  await expect(page.getByTestId('rail-badge-review')).toHaveCount(0);
});
