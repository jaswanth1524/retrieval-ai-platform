// Settings → Administration: backups, consistency, rebuild, audit log.
import { expect, test } from '../fixtures';
import { open } from '../helpers';

test('Settings shows the last scheduled backup', async ({ page, api }) => {
  api.documents = ['guide.md', 'notes.md'];
  api.readiness = {
    last_backup: { at: Date.now() / 1000 - 120, workspace: 'default', ok: true, file: 'docrag-20260101-000000.zip' },
  };
  await open(page);
  await page.getByTestId('open-settings').click();
  await expect(page.getByTestId('settings-last-backup')).toContainText('docrag-20260101-000000.zip');
});

test('Administration lists backups, runs one and checks the index', async ({ page, api }) => {
  api.backups = {
    backups: [{ name: 'docrag-20261101-000000.zip', size: 2048, created_at: Date.now() / 1000, verified: true }],
    interval_hours: 24,
    keep: 7,
  };
  api.answerCache = {
    enabled: true,
    entries: [{ question: 'what is docrag?', age_seconds: 12 }],
    total_entries: 1,
    max_entries: 256,
    ttl_seconds: 3600,
    in_flight: 0,
  };
  await open(page);
  await page.getByTestId('open-settings').click();
  await page.getByTestId('settings-admin-toggle').locator('summary').click();
  await expect(page.getByTestId('backup-item')).toContainText('verified');
  await page.getByTestId('backups-run').click();
  await expect.poll(() => api.backupRuns).toBe(1);
  await page.getByTestId('answer-cache-clear').click();
  // Asked first (pass 17): nothing is cleared until it is confirmed.
  expect(api.cacheClears).toBe(0);
  await page.getByTestId('answer-cache-clear-confirm').click();
  await expect.poll(() => api.cacheClears).toBe(1);
  await page.getByTestId('workspace-settings-min-score').fill('0.3');
  await page.getByTestId('workspace-settings-save').click();
  await expect.poll(() => api.settingsPuts.at(-1)).toMatchObject({ rerank_min_score: 0.3 });
  await page.getByTestId('consistency-run').click();
  await expect(page.getByTestId('consistency-result')).toContainText('agree with their originals');
});

test('Administration repairs what the consistency check found', async ({ page, api }) => {
  api.backups = {
    backups: [{ name: 'docrag-20261101-000000.zip', size: 2048, created_at: Date.now() / 1000, verified: true }],
    interval_hours: 24,
    keep: 7,
  };
  api.consistency = {
    documents: 1,
    originals_kept: true,
    problems: 1,
    originals_without_document: ['orphan.txt'],
    documents_without_original: [],
    versions_without_document: [],
    incomplete: [],
    stale: [],
    duplicate_groups: [],
    in_flight: [],
  };
  await open(page);
  await page.getByTestId('open-settings').click();
  await page.getByTestId('settings-admin-toggle').locator('summary').click();
  await page.getByTestId('backup-delete').click();
  await page.getByTestId('backup-delete-confirm').click();
  await expect.poll(() => api.backupDeletes).toEqual(['docrag-20261101-000000.zip']);
  await page.getByTestId('consistency-run').click();
  await page.getByTestId('consistency-repair-delete_orphan_originals').check();
  await page.getByTestId('consistency-repair-run').click();
  await page.getByTestId('consistency-repair-confirm').click();
  await expect.poll(() => api.repairs).toEqual([{ delete_orphan_originals: true }]);
});

test('Administration starts a rebuild and reads the audit log', async ({ page, api }) => {
  api.auditEntries = [
    { created_at: Date.now() / 1000, workspace: 'default', event: 'document.delete', key: 'full', fields: { filename: 'old.pdf' } },
  ];
  await open(page);
  await page.getByTestId('open-settings').click();
  await page.getByTestId('settings-admin-toggle').locator('summary').click();
  await page.getByTestId('rebuild-run').click();
  await page.getByRole('button', { name: 'Rebuild', exact: true }).click();
  await expect(page.getByTestId('rebuild-result')).toContainText('Indexing 2 documents');
  expect(api.rebuilds).toBe(1);
  await page.getByTestId('audit-load').click();
  await expect(page.getByTestId('audit-entry')).toContainText('document.delete (full) filename=old.pdf');
  await page.getByTestId('notify-test-run').click();
  await expect(page.getByTestId('notify-test-result')).toContainText('Notifications are off');
});
