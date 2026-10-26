import { useEffect, useRef } from 'react';

/** At most one re-list per this long from the tab coming back into view. */
export const CORPUS_REFRESH_MIN_INTERVAL_MS = 30_000;
/** How often an empty corpus is listed again (its first document may come from the
 *  watched folder or the CLI, and the composer waits for one), and a watched one. */
export const EMPTY_CORPUS_POLL_MS = 15_000;
export const WATCHED_CORPUS_POLL_MS = 60_000;

/**
 * Re-lists the corpus when the tab comes back into view and when the connection comes
 * back. The listing was read at boot and after this tab's own changes only: documents
 * added by the watched folder, the CLI or another tab stayed invisible until a reload,
 * and an empty corpus kept the composer disabled after one arrived.
 */
export function useCorpusAutoRefresh(
  enabled: boolean,
  connectionLost: boolean,
  refresh: () => Promise<unknown>,
  pollMs: number | null = null,
): void {
  const refreshRef = useRef(refresh);
  refreshRef.current = refresh;
  const lastRef = useRef(0);

  useEffect(() => {
    if (!enabled) return;
    // Counted from now: boot has just listed.
    lastRef.current = Date.now();
    const onVisible = () => {
      if (document.visibilityState !== 'visible') return;
      if (Date.now() - lastRef.current < CORPUS_REFRESH_MIN_INTERVAL_MS) return;
      lastRef.current = Date.now();
      refreshRef.current().catch(() => {
        // The listing's own error handling (the banner, the auth state) applies; a
        // background re-list failing is not worth a toast.
      });
    };
    document.addEventListener('visibilitychange', onVisible);
    return () => document.removeEventListener('visibilitychange', onVisible);
  }, [enabled]);

  // Polled while the tab is visible, for documents nobody in this tab added.
  useEffect(() => {
    if (!enabled || pollMs === null || connectionLost) return;
    const timer = setInterval(() => {
      if (document.visibilityState !== 'visible') return;
      lastRef.current = Date.now();
      refreshRef.current().catch(() => {});
    }, pollMs);
    return () => clearInterval(timer);
  }, [enabled, pollMs, connectionLost]);

  const wasLost = useRef(connectionLost);
  useEffect(() => {
    const restored = wasLost.current && !connectionLost;
    wasLost.current = connectionLost;
    if (!enabled || !restored) return;
    lastRef.current = Date.now();
    refreshRef.current().catch(() => {});
  }, [enabled, connectionLost]);
}
