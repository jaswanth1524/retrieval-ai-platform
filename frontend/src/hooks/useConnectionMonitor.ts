import { useEffect, useRef, useState } from 'react';
import { api } from '../api/client';

// /health while the page is up: how often, and how often once it has failed.
const HEALTH_POLL_MS = 30_000;
const HEALTH_RETRY_MS = 5_000;

/** True while the API has stopped answering since boot (polled only once `booted`). */
export function useConnectionMonitor(booted: boolean): boolean {
  // The server can go away after boot (a restart, a laptop's network dropping). The
  // footer said "api ok" for the rest of the session, and the first sign was a failed
  // question. /health is polled while the page is visible — every 30 s, every 5 s once
  // it has failed — and the browser's offline/online events are taken at their word.
  const [connectionLost, setConnectionLost] = useState(false);
  const connectionLostRef = useRef(false);
  connectionLostRef.current = connectionLost;
  useEffect(() => {
    if (!booted) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const schedule = () => {
      clearTimeout(timer);
      timer = setTimeout(
        () => void check(),
        connectionLostRef.current ? HEALTH_RETRY_MS : HEALTH_POLL_MS,
      );
    };
    const markLost = (lost: boolean) => {
      // The ref too, at once: the next check is scheduled before React re-renders.
      connectionLostRef.current = lost;
      setConnectionLost(lost);
    };
    const check = async () => {
      if (document.visibilityState !== 'hidden') {
        try {
          await api.health();
          if (!cancelled) markLost(false);
        } catch {
          if (!cancelled) markLost(true);
        }
      }
      if (!cancelled) schedule();
    };
    const goneOffline = () => {
      markLost(true);
      schedule();
    };
    const backOnline = () => void check();
    window.addEventListener('offline', goneOffline);
    window.addEventListener('online', backOnline);
    schedule();
    return () => {
      cancelled = true;
      clearTimeout(timer);
      window.removeEventListener('offline', goneOffline);
      window.removeEventListener('online', backOnline);
    };
  }, [booted]);
  return connectionLost;
}
