import { useCallback, useEffect, useRef } from 'react';

/** One request at a time, cancellable: ``start()`` aborts the one before it and returns
 *  the new request's signal; ``cancel()`` aborts it; leaving (unmount) aborts it too.
 *  For the long admin requests (a backup check, a file preview, a re-check): the page
 *  stops waiting, and a result that arrives after a cancel is never shown. */
export function useRequestSignal(): { start: () => AbortSignal; cancel: () => void } {
  const controllerRef = useRef<AbortController | null>(null);
  const cancel = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
  }, []);
  const start = useCallback(() => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    return controller.signal;
  }, []);
  useEffect(() => cancel, [cancel]);
  return { start, cancel };
}
