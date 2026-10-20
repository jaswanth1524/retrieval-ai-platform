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
  /** The full key: may start a watcher pass. */
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
  const finishedRef = useRef(onJobsFinished);
  finishedRef.current = onJobsFinished;

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let running = 0;
    const poll = async () => {
      try {
        const response = await api.listActiveJobs(controller.signal, workspace);
        if (controller.signal.aborted) return;
        setJobs(response.jobs);
        if (running > 0 && response.jobs.length < running) finishedRef.current();
        running = response.jobs.length;
      } catch {
        if (controller.signal.aborted) return;
        // An older server, or a refused key: nothing to show.
        setJobs([]);
        running = 0;
      }
      // Fast while something runs; otherwise only as often as the watcher is checked.
      timer = setTimeout(() => void poll(), running > 0 ? ACTIVE_JOBS_POLL_MS : WATCH_STATUS_POLL_MS);
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
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export default memo(ServerActivity);
