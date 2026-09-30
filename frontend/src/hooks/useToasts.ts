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
}

export const TOAST_TIMEOUT_MS = 4200;

export function useToasts(): UseToastsResult {
  const [toasts, setToasts] = useState<Toast[]>([]);
  // Every live auto-dismiss timer, so unmount can clear them all. Without this a
  // pending timer fires into an unmounted tree.
  const timers = useRef(new Map<string, ReturnType<typeof setTimeout>>());

  useEffect(() => {
    const pending = timers.current;
    return () => {
      pending.forEach(clearTimeout);
      pending.clear();
    };
  }, []);

  const dismiss = useCallback((id: string) => {
    const timer = timers.current.get(id);
    if (timer !== undefined) {
      clearTimeout(timer);
      timers.current.delete(id);
    }
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  }, []);

  const push = useCallback(
    (toast: ToastInput) => {
      // A caller-supplied id makes a toast idempotent: a persistence failure that
      // re-renders repeatedly should replace its own notice, not stack up dozens.
      const id = toast.id ?? newId();
      const existing = timers.current.get(id);
      if (existing !== undefined) {
        clearTimeout(existing);
        timers.current.delete(id);
      }

      setToasts((prev) => [
        ...prev.filter((candidate) => candidate.id !== id),
        { id, title: toast.title, body: toast.body, tone: toast.tone },
      ]);
      if (toast.sticky) return id;
      timers.current.set(
        id,
        setTimeout(() => {
          timers.current.delete(id);
          setToasts((prev) => prev.filter((candidate) => candidate.id !== id));
        }, TOAST_TIMEOUT_MS),
      );
      return id;
    },
    [],
  );

  return { toasts, push, dismiss };
}
