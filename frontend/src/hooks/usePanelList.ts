import { useEffect, useRef, useState } from 'react';
import { apiErrorMessage } from '../api/client';

interface PanelList<T> {
  /** null while loading (or before the first load). */
  items: T[] | null;
  error: string | null;
  /** Load again (Refresh), aborting a load still in flight. */
  load: () => void;
}

/**
 * A rail panel's list (traces, feedback), loaded each time the panel opens, and again
 * when ``reloadKey`` changes while it is open (the workspace: an open list otherwise
 * kept showing the previous workspace's rows, whose traces then 404'd).
 *
 * One request at a time: a Refresh aborts the load still in flight, whose late answer
 * used to overwrite the newer one, and closing the panel aborts it too. A 401 goes to
 * `onError` (the key banner explains it, like every other list) as well as showing here.
 */
export function usePanelList<T>(
  active: boolean,
  fetchItems: (signal: AbortSignal) => Promise<T[]>,
  onError: (err: unknown) => void,
  fallbackMessage: string,
  reloadKey?: unknown,
): PanelList<T> {
  const [items, setItems] = useState<T[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const requestRef = useRef<AbortController | null>(null);
  // The latest callbacks, so `load` keeps one identity and the effect runs per opening.
  const fetchRef = useRef(fetchItems);
  fetchRef.current = fetchItems;
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;

  const loadRef = useRef(() => {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    const { signal } = controller;
    setItems(null);
    setError(null);
    fetchRef
      .current(signal)
      .then((next) => {
        if (!signal.aborted) setItems(next);
      })
      .catch((err) => {
        if (signal.aborted) return;
        onErrorRef.current(err);
        setError(apiErrorMessage(err, fallbackMessage));
      });
  });

  useEffect(() => {
    if (!active) return;
    loadRef.current();
    return () => requestRef.current?.abort();
  }, [active, reloadKey]);

  return { items, error, load: loadRef.current };
}
