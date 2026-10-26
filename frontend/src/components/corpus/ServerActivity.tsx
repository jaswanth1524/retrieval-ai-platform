import { memo, useEffect, useRef, useState } from 'react';
import { api, apiErrorMessage } from '../../api/client';
import type { DocumentJobStatusResponse, WatchStatusResponse } from '../../api/types';
import { formatRelativeTime } from '../../utils/relativeTime';

/** How often the jobs list is read while any job runs, and the watcher's status. */
export const ACTIVE_JOBS_POLL_MS = 3_000;
export const WATCH_STATUS_POLL_MS = 30_000;

interface ServerActivityProps {
  workspace: string;
  /** Files this tab is already following (its own uploads): not listed twice. */
  followed: ReadonlySet<string>;
  /** WATCH_DIR is set on the server. */
  watchEnabled: boolean;
  /** The full key: may start a watcher pass and cancel jobs. */
  canScan: boolean;
  /** Jobs listed here finished: the document list is out of date. */
  onJobsFinished: () => void;
}

/**
 * Indexing the server is doing that this tab didn't start — the watched folder, a
 * restore, a bulk re-index, another tab, the CLI — and the watched folder's state. Those
 * jobs used to be invisible: their documents appeared (or didn't) with no word why.
 */
function ServerActivity({
  workspace,
  followed,
  watchEnabled,
  canScan,
  onJobsFinished,
}: ServerActivityProps) {
  const [jobs, setJobs] = useState<DocumentJobStatusResponse[]>([]);
  const [watch, setWatch] = useState<WatchStatusResponse | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);
  // Job ids a cancel was asked for, and why one couldn't be.
  const [cancelling, setCancelling] = useState<ReadonlySet<string>>(() => new Set());
  const [cancelError, setCancelError] = useState<string | null>(null);
  const [failures, setFailures] = useState<DocumentJobStatusResponse[] | null>(null);
  const finishedRef = useRef(onJobsFinished);
  finishedRef.current = onJobsFinished;

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let running = new Set<string>();
    const poll = async () => {
      try {
        const response = await api.listActiveJobs(controller.signal, workspace);
        if (controller.signal.aborted) return;
        setJobs(response.jobs);
        const now = new Set(response.jobs.map((job) => job.job_id));
        // By id, not by count: one job ending as another starts left the count, and the
        // document list, unchanged.
        if ([...running].some((id) => !now.has(id))) finishedRef.current();
        running = now;
      } catch {
        if (controller.signal.aborted) return;
        // An older server, or a refused key: nothing to show.
        setJobs([]);
        running = new Set();
      }
      // Fast while something runs; otherwise only as often as the watcher is checked.
      timer = setTimeout(
        () => void poll(),
        running.size > 0 ? ACTIVE_JOBS_POLL_MS : WATCH_STATUS_POLL_MS,
      );
    };
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [workspace]);

  useEffect(() => {
    if (!watchEnabled) return;
    const controller = new AbortController();
    const read = () =>
      api
        .watchStatus(controller.signal)
        .then((status) => setWatch(status))
        .catch(() => {});
    void read();
    const interval = setInterval(() => void read(), WATCH_STATUS_POLL_MS);
    return () => {
      controller.abort();
      clearInterval(interval);
    };
  }, [watchEnabled]);

  const elsewhere = jobs.filter((job) => !followed.has(job.filename));
  if (elsewhere.length === 0 && !watch) return null;

  const cancel = (job: DocumentJobStatusResponse) => {
    setCancelError(null);
    setCancelling((prev) => new Set(prev).add(job.job_id));
    api.cancelDocumentJob(job.job_id, undefined, workspace).catch((err) => {
      setCancelling((prev) => {
        const next = new Set(prev);
        next.delete(job.job_id);
        return next;
      });
      // 409: it already wrote part of the new version; it has to finish.
      setCancelError(apiErrorMessage(err, `Could not cancel ${job.filename}.`));
    });
  };

  const loadFailures = () => {
    if (failures !== null) return;
    api.listFinishedJobs(true, undefined, workspace).then(
      (response) => setFailures(response.jobs),
      () => setFailures([]),
    );
  };

  const scan = () => {
    setScanError(null);
    api.watchScan().then(
      (status) => setWatch(status),
      (err) => setScanError(apiErrorMessage(err, 'Could not start a pass.')),
    );
  };

  return (
    <section className="corpus-panel__activity" aria-label="Server activity" data-testid="server-activity">
      {watch && (
        <div className="corpus-panel__watch" data-testid="watch-status">
          <span className="mono">
            watching {watch.folder} · {watch.files} file{watch.files === 1 ? '' : 's'}
            {watch.last_scan_at ? ` · ${formatRelativeTime(watch.last_scan_at * 1000)}` : ''}
          </span>
          {canScan && (
            <button
              type="button"
              className="corpus-panel__bulk-button"
              onClick={scan}
              data-testid="watch-scan"
            >
              Scan now
            </button>
          )}
          {watch.retrying.map((retry) => (
            <p key={retry.filename} className="corpus-panel__watch-note" title={retry.error}>
              retrying {retry.filename} (attempt {retry.attempts}): {retry.error}
            </p>
          ))}
          {watch.gave_up.length > 0 && (
            <p className="corpus-panel__watch-note">
              not indexed until changed: {watch.gave_up.join(', ')}
            </p>
          )}
          {watch.deferred.length > 0 && (
            <p className="corpus-panel__watch-note" data-testid="watch-deferred">
              waiting for room in the indexing queue: {watch.deferred.join(', ')}
            </p>
          )}
          {watch.problems.map((problem) => (
            <p key={problem} className="corpus-panel__watch-note" data-testid="watch-problem">
              skipped {problem}
            </p>
          ))}
          <details className="corpus-panel__watch-failures" onToggle={loadFailures}>
            <summary>Recent failures</summary>
            {failures === null ? (
              <p className="corpus-panel__watch-note">Loading…</p>
            ) : failures.length === 0 ? (
              <p className="corpus-panel__watch-note">None kept.</p>
            ) : (
              failures.map((job) => (
                <p key={job.job_id} className="corpus-panel__watch-note" data-testid="failed-job">
                  {job.filename}: {job.error ?? 'failed'}
                </p>
              ))
            )}
          </details>
          {scanError && (
            <p className="corpus-panel__reject" role="alert">
              {scanError}
            </p>
          )}
        </div>
      )}
      {elsewhere.length > 0 && (
        <ul className="corpus-panel__jobs" data-testid="active-jobs">
          {elsewhere.map((job) => (
            <li key={job.job_id} className="mono">
              {job.state}
              {job.chunks_total > 0 ? ` ${job.chunks_done}/${job.chunks_total}` : ''} · {job.filename}
              {canScan && (
                <button
                  type="button"
                  className="corpus-panel__job-cancel"
                  onClick={() => cancel(job)}
                  disabled={cancelling.has(job.job_id)}
                  aria-label={`Cancel indexing ${job.filename}`}
                  data-testid="active-job-cancel"
                >
                  {cancelling.has(job.job_id) ? 'cancelling…' : 'cancel'}
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
      {cancelError && (
        <p className="corpus-panel__reject" role="alert" data-testid="active-job-cancel-error">
          {cancelError}
        </p>
      )}
    </section>
  );
}

export default memo(ServerActivity);
