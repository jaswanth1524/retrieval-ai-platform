// The Traces panel: answer filters, score report, trace comparison, trying a bar.
import { expect, test, traceSummary } from '../fixtures';
import { open } from '../helpers';

test('the trace list asks the server for traces answered one way', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  api.traces = [
    { ...traceSummary('t-answered'), answer_status: 'answered' },
    { ...traceSummary('t-missed'), answer_status: 'no_context' },
  ];
  await open(page);
  await page.getByTestId('rail-traces').click();
  await expect(page.getByTestId('traces-panel-row')).toHaveCount(2);
  await page.getByTestId('traces-answer-filter').selectOption('unanswered');
  await expect(page.getByTestId('traces-panel-row')).toHaveCount(1);
  expect(api.traceQueries.at(-1)).toEqual(['insufficient_context', 'no_context', 'empty_scope']);
});

test('the traces panel shows the score report and compares two traces', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
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

test('a bar is tried on recent questions from the score report', async ({ page, api }) => {
  api.traces = [traceSummary('t-1', 'first?')];
  api.whatIf = { settings: { rerank_min_score: 0.1 }, checked: 4, would_answer: 3, newly_answered: 3, newly_unanswered: 0, items: [] };
  await open(page);
  await page.getByTestId('rail-traces').click();
  await page.getByTestId('traces-score-report').click();
  await page.getByTestId('what-if-bar').fill('0.1');
  await page.getByTestId('what-if-run').click();
  await expect(page.getByTestId('what-if-result')).toContainText('3 of 4 would be answered');
  expect(api.whatIfs.at(-1)).toMatchObject({ rerank_min_score: 0.1, only_unanswered: true });
});
