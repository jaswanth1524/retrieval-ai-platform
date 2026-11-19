// Server-side jobs: jobs started elsewhere, retries, sitemap imports.
import { expect, test } from '../fixtures';
import { open } from '../helpers';

test('a job started elsewhere can be cancelled and skipped files are named', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  api.config = { watch_enabled: true };
  api.activeJobs = [
    {
      job_id: 'outside-1',
      filename: 'watched.md',
      state: 'embedding',
      chunks_total: 4,
      chunks_done: 1,
      error: null,
      result: null,
    },
  ];
  api.watchStatus = {
    folder: '/inbox',
    last_scan_at: null,
    files: 2,
    problems: ['/inbox/link.pdf: not a regular file'],
    deferred: [],
    retrying: [],
    gave_up: [],
  };
  api.finishedJobs = [
    { job_id: 'old-1', filename: 'broken.pdf', state: 'failed', error: 'Unreadable PDF.' },
  ];
  await page.goto('/');
  await page.getByTestId('rail-corpus').click();
  await expect(page.getByTestId('watch-problem')).toContainText('link.pdf');
  await page.getByRole('button', { name: 'Cancel indexing watched.md' }).click();
  await expect.poll(() => api.cancelledOutsideJobs).toEqual(['outside-1']);
  await page.getByText('Recent failures').click();
  await expect(page.getByTestId('failed-job')).toContainText('broken.pdf: Unreadable PDF.');
});

test('a failed job is retried from the corpus panel', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  api.finishedJobs = [
    {
      job_id: 'old-job',
      filename: 'notes.md',
      state: 'failed',
      chunks_total: 0,
      chunks_done: 0,
      error: 'The URL did not answer in time.',
      result: null,
      kind: 'url',
      retryable: true,
    },
  ];
  await open(page);
  await page.getByTestId('rail-corpus').click();
  await page.getByText('Recent failures').click();
  await page.getByTestId('finished-job-retry').click();
  await expect.poll(() => api.retriedJobs).toEqual(['old-job']);
});

test('a sitemap import shows its progress and can be cancelled', async ({ page, api }) => {
  api.sitemapImports = [
    {
      id: 'imp1',
      url: 'https://example.com/sitemap.xml',
      state: 'running',
      total: 10,
      done: 2,
      queued: 2,
      failed: 0,
      skipped: 0,
      unchanged: 0,
      cancelled_pages: 0,
      started_at: Date.now() / 1000,
      finished_at: null,
      problems: [],
    },
  ];
  await open(page);
  await page.getByTestId('rail-corpus').click();
  await expect(page.getByTestId('sitemap-import')).toContainText('2 of 10 page(s)');
  await page.getByTestId('sitemap-import-cancel').click();
  await expect.poll(() => api.sitemapCancels).toEqual(['imp1']);
});
