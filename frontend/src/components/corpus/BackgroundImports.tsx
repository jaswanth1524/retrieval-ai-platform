import { useEffect, useState } from 'react';
import { api, apiErrorMessage } from '../../api/client';
import type { BackgroundImportResponse } from '../../api/types';

const RUNNING_POLL_MS = 2_000;
const IDLE_POLL_MS = 30_000;
/** A finished import stays listed this long. */
const FINISHED_SHOWN_MS = 10 * 60_000;

/** Zips and backups the server imports in the background (`background=true`): how many
 *  documents it has queued, skipped and failed so far — and Cancel. */
export default function BackgroundImports({
  workspace,
  canCancel,
}: {
  workspace: string;
  canCancel: boolean;
}) {
  const [imports, setImports] = useState<BackgroundImportResponse[]>([]);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      let running = false;
      if (document.visibilityState !== 'hidden') {
        try {
          const response = await api.listBackgroundImports(controller.signal, workspace);
          if (controller.signal.aborted) return;
          const cutoff = Date.now() - FINISHED_SHOWN_MS;
          const shown = response.imports.filter(
            (item) => item.finished_at == null || item.finished_at * 1000 >= cutoff,
          );
          setImports(shown);
          running = shown.some((item) => item.finished_at == null);
        } catch {
          if (controller.signal.aborted) return;
          setImports([]); // an older server, or a refused key
        }
      }
      timer = setTimeout(() => void poll(), running ? RUNNING_POLL_MS : IDLE_POLL_MS);
    };
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [workspace]);

  if (imports.length === 0) return null;
  return (
    <div className="corpus-panel__sitemaps" data-testid="background-imports">
      {imports.map((item) => (
        <p key={item.id} className="corpus-panel__watch-note" data-testid="background-import">
          <span className="mono">{item.name}</span> ({item.kind === 'restore' ? 'backup' : 'zip'}):{' '}
          {item.queued ?? 0} queued
          {item.skipped ? `, ${item.skipped} skipped` : ''}
          {item.failed ? `, ${item.failed} failed` : ''}
          {item.state === 'cancelled'
            ? ' (cancelled)'
            : item.state === 'cancelling'
              ? ' (stopping…)'
              : item.state === 'running'
                ? ' so far'
                : ' — done'}
          {item.state === 'running' && canCancel && (
            <>
              {' '}
              <button
                type="button"
                className="corpus-panel__bulk-button"
                onClick={() => {
                  setError(null);
                  api
                    .cancelBackgroundImport(item.id, workspace)
                    .then((next) =>
                      setImports((prev) => prev.map((each) => (each.id === next.id ? next : each))),
                    )
                    .catch((err) => setError(apiErrorMessage(err, 'Could not cancel it.')));
                }}
                data-testid="background-import-cancel"
              >
                Cancel
              </button>
            </>
          )}
          {item.problems && item.problems.length > 0 && (
            <span className="corpus-panel__sitemap-problems" title={item.problems.join('\n')}>
              {' '}
              · {item.problems[0]}
              {item.problems.length > 1 ? ` (and ${item.problems.length - 1} more)` : ''}
            </span>
          )}
        </p>
      ))}
      {error && <p role="alert">{error}</p>}
    </div>
  );
}
