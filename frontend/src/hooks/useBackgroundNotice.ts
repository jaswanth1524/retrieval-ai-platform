import { useEffect, useRef } from 'react';

const NOTIFY_STORAGE_KEY = 'docrag-notify-answers';

export function notificationsSupported(): boolean {
  return typeof window !== 'undefined' && 'Notification' in window;
}

export { NOTIFY_STORAGE_KEY };

/** Say so when an answer finishes in a tab the user has left: answers take a minute or
 *  two on local models, long enough to switch away. The tab title counts unseen answers
 *  ("(2) DocRAG") until the tab is shown again, and — only when the user turned it on
 *  in Settings and the browser granted it — a system notification is shown too. */
export function useBackgroundNotice(pending: boolean, notify: boolean, question?: string): void {
  const baseTitle = useRef(document.title);
  const unseen = useRef(0);
  const wasPending = useRef(pending);
  const latest = useRef({ notify, question });
  latest.current = { notify, question };

  useEffect(() => {
    const finished = wasPending.current && !pending;
    wasPending.current = pending;
    if (!finished || !document.hidden) return;
    unseen.current += 1;
    document.title = `(${unseen.current}) ${baseTitle.current}`;
    if (latest.current.notify && notificationsSupported() && Notification.permission === 'granted') {
      try {
        new Notification('Answer ready', {
          body: latest.current.question?.slice(0, 120),
          tag: 'docrag-answer',
        });
      } catch {
        // Some browsers only allow notifications from a service worker; the title
        // badge still says it.
      }
    }
  }, [pending]);

  useEffect(() => {
    const onVisibility = () => {
      if (document.hidden) return;
      unseen.current = 0;
      document.title = baseTitle.current;
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
  }, []);
}
