import { readFileSync } from 'node:fs';
import { answerStream, expect, GUIDE_SOURCE, test, traceSummary } from '../fixtures';

test.beforeEach(async ({ api }) => {
  api.documents = ['guide.md', 'notes.md'];
});

async function open(page: import('@playwright/test').Page) {
  await page.goto('/');
  await expect(page.getByTestId('question-textarea')).toBeEnabled();
}

async function ask(page: import('@playwright/test').Page, question = 'What retrieval strategy does DocRAG use?') {
  const box = page.getByTestId('question-textarea');
  await box.fill(question);
  await box.press('Enter');
  await expect(page.getByTestId('question-submit')).toHaveText('Ask');
}

test('a cited source uploaded again after the answer is flagged on its card', async ({ page, api }) => {
  api.stream = answerStream();
  // notes.md listed as it was; guide.md's upload time later than the answer below.
  api.uploadedAts = { 'guide.md': Date.now() / 1000 + 3600, 'notes.md': 1 };
  await open(page);
  await ask(page);
  await expect(page.getByTestId('citation-changed')).toHaveText('document changed since');
  await page.getByTestId('chat-message-save').click();
  await page.getByTestId('rail-saved').click();
  await expect(page.getByTestId('saved-changed')).toContainText('changed since: guide.md');
});

test('cited sources export as CSV, and a number key opens that source', async ({ page, api }) => {
  api.stream = answerStream();
  await open(page);
  await ask(page);

  await page.keyboard.press('ControlOrMeta+k');
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('option', { name: /Export cited sources/ }).click(),
  ]);
  expect(download.suggestedFilename()).toMatch(/sources\.csv$/);
  const csv = readFileSync((await download.path())!, 'utf8');
  expect(csv.split('\r\n')[0]).toBe('conversation,question,answer,source,filename,page,section,chunk_id,quote');
  expect(csv).toContain(`,1,1,guide.md,p.1,Retrieval,${GUIDE_SOURCE.chunk_id},`);

  await page.locator('body').click({ position: { x: 5, y: 5 } });
  await page.keyboard.press('1');
  await expect(page.getByRole('dialog').getByTestId('viewer-chunk').first()).toBeVisible();
});

test('a scope is saved under a name and picked again', async ({ page }) => {
  await open(page);
  await page.getByTestId('composer-scope-button').click();
  await expect(page.getByTestId('scope-preset-save')).toBeDisabled();
  await page.getByTestId('scope-item').filter({ hasText: 'notes.md' }).click();
  await page.getByTestId('scope-preset-save').click();
  await page.getByTestId('scope-preset-name').fill('Notes only');
  await page.getByTestId('scope-preset-confirm').click();
  await expect(page.getByTestId('scope-preset')).toHaveText(/Notes only/);

  await page.getByTestId('scope-all').check();
  await expect(page.getByTestId('composer-scope-button')).not.toContainText('notes.md');
  await page.getByTestId('scope-preset').getByRole('button', { name: 'Notes only', exact: true }).click();
  await expect(page.getByTestId('composer-scope-button')).toContainText('notes.md');
});

test('a document card lists the questions that cited it, here and in the log', async ({ page, api }) => {
  api.config = { query_log_enabled: true };
  api.unanswered = [
    {
      created_at: Date.now() / 1000,
      question: 'How is retrieval fused?',
      status: 'ok',
      top_score: 0.8,
      mode: 'stream',
      trace_id: null,
      filenames: ['guide.md'],
    },
  ];
  api.stream = answerStream();
  await open(page);
  await ask(page);

  await page.getByTestId('rail-corpus').click();
  const card = page.getByTestId('corpus-panel-item').filter({ hasText: 'guide.md' });
  await card.getByTestId('corpus-usage-toggle').click();
  await expect(card.getByTestId('corpus-usage-answer')).toHaveText(/What retrieval strategy/);
  await expect(card.getByTestId('corpus-usage-logged')).toContainText('1 logged question drew on it');
  expect(api.queryLogFilenames).toEqual(['guide.md']);
});

test('the traces panel shows the score report and compares two traces', async ({ page, api }) => {
  api.traces = [traceSummary('t-1', 'first?'), traceSummary('t-2', 'second?')];
  api.scoreReport = {
    current_min_score: 0.15,
    bin_edges: [0, 0.5, 1],
    answered: [0, 30],
    unanswered: [25, 0],
    answered_count: 30,
    unanswered_count: 25,
    suggested_min_score: 0.31,
    note: 'Check a few first.',
    would_change: 4,
  };
  api.traceDiff = {
    question_a: 'first?',
    question_b: 'second?',
    same_answer: false,
    settings: [{ name: 'rerank_top_k', a: 3, b: 5 }],
    candidates: [
      {
        point_id: 'p1',
        filename: 'guide.md',
        page: 1,
        chunk_id: 'c1',
        rank_a: 1,
        rank_b: 3,
        selected_a: true,
        selected_b: false,
        drop_reason_b: 'below_min_score',
      },
    ],
  };
  await open(page);
  await page.getByTestId('rail-traces').click();
  await page.getByTestId('traces-score-report').click();
  await expect(page.getByTestId('score-report-suggestion')).toContainText('RERANK_MIN_SCORE=0.31');
  await expect(page.getByTestId('score-report-bin')).toHaveCount(2);

  await page.getByTestId('traces-compare').click();
  await page.getByTestId('traces-panel-row').nth(0).click();
  await page.getByTestId('traces-panel-row').nth(1).click();
  await expect(page.getByTestId('trace-diff-setting')).toHaveText('rerank_top_k: 3 → 5');
  await expect(page.getByTestId('trace-diff-candidate')).toContainText('no longer in the prompt');
});

test('unanswered questions are searched again from the Asked questions panel', async ({ page, api }) => {
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

test('Settings shows the last scheduled backup', async ({ page, api }) => {
  api.readiness = {
    last_backup: { at: Date.now() / 1000 - 120, workspace: 'default', ok: true, file: 'docrag-20260101-000000.zip' },
  };
  await open(page);
  await page.getByTestId('open-settings').click();
  await expect(page.getByTestId('settings-last-backup')).toContainText('docrag-20260101-000000.zip');
});

test('a document with the same bytes as another says so on its card', async ({ page, api }) => {
  api.duplicateGroups = [['guide.md', 'notes.md']];
  await open(page);
  await page.getByTestId('rail-corpus').click();
  await expect(page.locator('[data-corpus-card="guide.md"]')).toContainText('same content as notes.md');
});
