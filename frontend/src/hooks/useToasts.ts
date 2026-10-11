import { useCallback, useEffect, useRef, useState } from 'react';
import { newId } from '../utils/id';

export type ToastTone = 'info' | 'good' | 'warn' | 'bad';

export interface Toast {
  id: string;
  title: string;
  body?: string;
  tone: ToastTone;
}

export type ToastInput = Omit<Toast, 'id'> & { id?: string; sticky?: boolean };

export interface UseToastsResult {
  toasts: Toast[];
  /** Show a toast; returns its id so a caller can dismiss it early. A `sticky` toast
   * stays until dismissed — for warnings that must not vanish while the user looks away. */
  push: (toast: ToastInput) => string;
  dismiss: (id: string) => void;
  /** Pause a toast's auto-dismiss while it is hovered or focused; ``release`` restarts it. */
  hold: (id: string) => void;
  release: (id: string) => void;
}

export const TOAST_TIMEOUT_MS = 4200;
// Warnings and failures carry the reason something didn't happen ("Restore failed: …"),
// which took longer to read than 4.2 s allowed.
export const FAILURE_TOAST_TIMEOUT_MS = 10_000;

export function useToasts(): UseToastsResult {
  const [toasts, setToasts] = useState<Toast[]>([]);
  // Every live auto-dismiss timer, so unmount can clear them all. Without this a
  // pending timer fires into an unmounted tree.
  const timers = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  // How long each auto-dismissing toast lives, for restarting its timer after a hold.
  const durations = useRef(new Map<string, number>());

  useEffect(() => {
    const pending = timers.current;
    return () => {
      pending.forEach(clearTimeout);
      pending.clear();
    };
  }, []);

  const stopTimer = useCallback((id: string) => {
    const timer = timers.current.get(id);
    if (timer !== undefined) {
      clearTimeout(timer);
      timers.current.delete(id);
    }
  }, []);

  const startTimer = useCallback((id: string, ms: number) => {
    timers.current.set(
      id,
      setTimeout(() => {
        timers.current.delete(id);
        durations.current.delete(id);
        setToasts((prev) => prev.filter((candidate) => candidate.id !== id));
      }, ms),
    );
  }, []);

  const dismiss = useCallback((id: string) => {
    stopTimer(id);
    durations.current.delete(id);
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  }, [stopTimer]);

  const hold = useCallback((id: string) => stopTimer(id), [stopTimer]);

  const release = useCallback((id: string) => {
    const ms = durations.current.get(id);
    if (ms === undefined || timers.current.has(id)) return;
    startTimer(id, ms);
  }, [startTimer]);

  const push = useCallback(
    (toast: ToastInput) => {
      // A caller-supplied id makes a toast idempotent: a persistence failure that
      // re-renders repeatedly should replace its own notice, not stack up dozens.
      const id = toast.id ?? newId();
      stopTimer(id);
      durations.current.delete(id);

      setToasts((prev) => [
        ...prev.filter((candidate) => candidate.id !== id),
        { id, title: toast.title, body: toast.body, tone: toast.tone },
      ]);
      if (toast.sticky) return id;
      const ms =
        toast.tone === 'warn' || toast.tone === 'bad' ? FAILURE_TOAST_TIMEOUT_MS : TOAST_TIMEOUT_MS;
      durations.current.set(id, ms);
      startTimer(id, ms);
      return id;
    },
    [stopTimer, startTimer],
  );

  return { toasts, push, dismiss, hold, release };
}
