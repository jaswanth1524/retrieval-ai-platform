import { useEffect, useState } from 'react';
import { api, apiErrorMessage } from '../../api/client';
import type { SitemapImportResponse } from '../../api/types';

/** How often imports are read while one runs, and otherwise. */
export const SITEMAP_RUNNING_POLL_MS = 2_000;
export const SITEMAP_IDLE_POLL_MS = 30_000;
/** Fired (on window) when this tab starts an import: the list is read at once. */
export const SITEMAP_STARTED_EVENT = 'docrag-sitemap-started';
/** A finished import stays listed this long. */
const FINISHED_SHOWN_MS = 10 * 60_000;

/** Sitemap imports going on (and those just finished): pages done of the listed ones,
 *  how many were queued, unchanged, failed — and Cancel. The pages show as jobs too;
 *  this says where the import as a whole is. */
export default function SitemapImports({
  workspace,
  canCancel,
  refreshKey = 0,
}: {
  workspace: string;
  canCancel: boolean;
  /** Bumped when an import was just started here: read again at once. */
  refreshKey?: number;
}) {
  const [imports, setImports] = useState<SitemapImportResponse[]>([]);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      let running = false;
      if (document.visibilityState !== 'hidden') {
        try {
          const response = await api.listSitemapImports(controller.signal, workspace);
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
      timer = setTimeout(() => void poll(), running ? SITEMAP_RUNNING_POLL_MS : SITEMAP_IDLE_POLL_MS);
    };
    const now = () => {
      clearTimeout(timer);
      void poll();
    };
    window.addEventListener(SITEMAP_STARTED_EVENT, now);
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
      window.removeEventListener(SITEMAP_STARTED_EVENT, now);
    };
  }, [workspace, refreshKey]);

  if (imports.length === 0) return null;
  return (
    <div className="corpus-panel__sitemaps" data-testid="sitemap-imports">
      {imports.map((item) => (
        <p key={item.id} className="corpus-panel__watch-note" data-testid="sitemap-import">
          <span className="mono">{host(item.url)}</span>: {item.done} of {item.total} page(s)
          {` — ${item.queued} added, ${item.unchanged} unchanged`}
          {item.failed ? `, ${item.failed} failed` : ''}
          {item.skipped ? `, ${item.skipped} skipped` : ''}
          {item.state === 'cancelled' ? ' (cancelled)' : item.state === 'cancelling' ? ' (stopping…)' : ''}
          {item.finished_at == null && canCancel && item.state === 'running' && (
            <>
              {' '}
              <button
                type="button"
                className="corpus-panel__bulk-button"
                onClick={() => {
                  setError(null);
                  api
                    .cancelSitemapImport(item.id, workspace)
                    .then((next) =>
                      setImports((prev) => prev.map((each) => (each.id === next.id ? next : each))),
                    )
                    .catch((err) => setError(apiErrorMessage(err, 'Could not cancel it.')));
                }}
                data-testid="sitemap-import-cancel"
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

function host(url: string): string {
  try {
    return new URL(url).host || url;
  } catch {
    return url;
  }
}
