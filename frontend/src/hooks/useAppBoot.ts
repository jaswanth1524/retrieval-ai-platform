import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiClientError, api } from '../api/client';
import type { AccessLevel, ApiStatus, PublicConfigResponse } from '../api/types';
import { BOOT_RETRY_BASE_MS, BOOT_RETRY_MAX_MS, LISTING_RETRY_BASE_MS } from '../appHelpers';
import { useStableHandler } from './useStableHandler';

interface AppBootOptions {
  /** Lists the corpus (useCorpus's refresh); throws on failure. */
  refreshDocuments: (signal?: AbortSignal) => Promise<void>;
  /** Runs once per successful /config, before the status turns 'ok'. */
  onConfig: (config: PublicConfigResponse) => void;
  /** Runs before the corpus is listed, e.g. to drop a remembered workspace that is gone. */
  beforeListing?: (signal: AbortSignal) => Promise<void>;
}

/**
 * What the app knows about the server: reachable and authorized (`apiStatus`), its
 * public configuration, which key this browser holds (`access`), and whether the corpus
 * has been listed. Boots with /config, /access and the listing in parallel, retries a
 * failed boot and a failed listing with backoff, and turns 401/403 responses from any
 * request into the key banner or the read-only UI (`noteAuthFailure`).
 */
export function useAppBoot({ refreshDocuments, onConfig, beforeListing }: AppBootOptions) {
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking');
  // The first corpus listing has come back (or failed): until then "no documents"
  // only means "not loaded yet", and must not tell anyone to upload one.
  const [corpusChecked, setCorpusChecked] = useState(false);
  // The first listing failed with something other than a 401 (a 500, a timeout on a
  // large corpus): unknown, not empty. It used to read as "no documents" and disabled
  // the composer under "Upload a document to start." until a reload.
  const [listingFailed, setListingFailed] = useState(false);
  const [config, setConfig] = useState<PublicConfigResponse | null>(null);
  // Optimistic until /access answers: a full-key (or keyless) server is the common case.
  const [access, setAccess] = useState<AccessLevel>('full');

  /**
   * /health and /config are unauthenticated, so a keyed server answers both and the app
   * looked healthy while every corpus request 401'd — the listing failure left
   * indexedFilenames empty, which disabled the composer under the hint "Upload a
   * document to start.", advice that would itself have 401'd. Returns true when it
   * handled the error, so callers keep their own fallback for everything else.
   */
  const noteAuthFailure = useStableHandler((err: unknown): boolean => {
    if (err instanceof ApiClientError && err.statusCode === 401) {
      setApiStatus('unauthorized');
      return true;
    }
    // A 403 means a read-only key (API_READ_KEY): hide what it can't do from now on.
    // The error itself is still shown by the caller.
    if (err instanceof ApiClientError && err.statusCode === 403) setAccess('read');
    return false;
  });

  /** A request succeeded with the key: the key banner can go. */
  const noteAuthRestored = useCallback(
    () => setApiStatus((prev) => (prev === 'unauthorized' ? 'ok' : prev)),
    [],
  );

  /** Ask the server which key this browser holds. An older server without /access
   *  (404) is treated as full access; a 403 on any write corrects that later. */
  const refreshAccess = useStableHandler(async (signal?: AbortSignal) => {
    try {
      setAccess((await api.access(signal)).access);
    } catch (err) {
      if (err instanceof ApiClientError && err.statusCode === 404) setAccess('full');
      else noteAuthFailure(err);
    }
  });

  const runRefreshDocuments = useStableHandler(refreshDocuments);
  const runOnConfig = useStableHandler(onConfig);
  const runBeforeListing = useStableHandler(async (signal: AbortSignal) => {
    await beforeListing?.(signal);
  });

  // Bumped to boot again: by the backoff below after a failed /config, or by the
  // banner's "Retry now". The API may just be starting (compose boots it after Qdrant),
  // and a one-shot boot left the page on "start the API and refresh" for good.
  const [bootAttempt, setBootAttempt] = useState(0);
  const retryBoot = useCallback(() => setBootAttempt((attempt) => attempt + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;

    // /config, /access and the corpus listing go out together: each round trip used to
    // wait for the one before it (a /health probe first), and the composer said "Upload
    // a document" until the listing, the last of them, came back. Only the listing waits
    // for beforeListing (it is read from a workspace; /config and /access are not).
    async function boot() {
      const listing = runBeforeListing(controller.signal)
        .then(() => (cancelled ? undefined : runRefreshDocuments(controller.signal)))
        .then(() => {
          if (!cancelled) setListingFailed(false);
        })
        .catch((err) => {
          // Anything other than a 401 leaves the corpus unknown; the effect below
          // retries the listing until it answers.
          if (!cancelled && !noteAuthFailure(err)) setListingFailed(true);
        })
        .finally(() => {
          if (!cancelled) setCorpusChecked(true);
        });
      void refreshAccess(controller.signal);
      try {
        const configResult = await api.config(controller.signal);
        if (cancelled) return;
        setConfig(configResult);
        runOnConfig(configResult);
        // The listing or /access may have answered 401 first; that stands.
        setApiStatus((prev) => (prev === 'unauthorized' ? prev : 'ok'));
      } catch {
        if (cancelled) return;
        setApiStatus('error');
        retryTimer = setTimeout(
          () => setBootAttempt((attempt) => attempt + 1),
          Math.min(BOOT_RETRY_BASE_MS * 2 ** bootAttempt, BOOT_RETRY_MAX_MS),
        );
      }
      await listing;
    }

    void boot();
    return () => {
      cancelled = true;
      clearTimeout(retryTimer);
      // Aborting releases the in-flight request instead of just ignoring its result.
      controller.abort();
    };
    // Boots once, and again per retry; everything else it calls is a stable handler.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bootAttempt]);

  /**
   * The corpus again, from scratch: another workspace is in use. Until it answers the
   * corpus is unknown (not empty, which said "Upload a document to start."), and a
   * failure is retried like a failed first listing instead of leaving it empty.
   */
  const relistSequence = useRef(0);
  const relistCorpus = useStableHandler(async () => {
    const sequence = ++relistSequence.current;
    setCorpusChecked(false);
    try {
      await runRefreshDocuments();
      if (sequence === relistSequence.current) setListingFailed(false);
    } catch (err) {
      if (sequence === relistSequence.current && !noteAuthFailure(err)) setListingFailed(true);
    } finally {
      // A newer switch has its own listing on the way.
      if (sequence === relistSequence.current) setCorpusChecked(true);
    }
  });

  // Retry a listing that failed (not a 401) while the page is open, backing off.
  useEffect(() => {
    if (!listingFailed) return;
    let cancelled = false;
    let delay = LISTING_RETRY_BASE_MS;
    let timer: ReturnType<typeof setTimeout>;
    const attempt = () => {
      timer = setTimeout(() => {
        runRefreshDocuments()
          .then(() => {
            if (!cancelled) setListingFailed(false);
          })
          .catch((err) => {
            if (cancelled) return;
            if (noteAuthFailure(err)) {
              setListingFailed(false);
              return;
            }
            delay = Math.min(delay * 2, BOOT_RETRY_MAX_MS);
            attempt();
          });
      }, delay);
    };
    attempt();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [listingFailed, runRefreshDocuments, noteAuthFailure]);

  return {
    apiStatus,
    corpusChecked,
    listingFailed,
    config,
    access,
    noteAuthFailure,
    noteAuthRestored,
    refreshAccess,
    relistCorpus,
    retryBoot,
  };
}
