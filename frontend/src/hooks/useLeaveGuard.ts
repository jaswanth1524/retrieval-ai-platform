import { useEffect } from 'react';

/** While ``active`` (an answer being written, a question waiting to be sent), leaving
 *  or reloading the page asks first: the browser's own "Leave site?" prompt. A reload
 *  mid-answer kept only the text streamed so far, marked stopped, and a queued
 *  question was lost without a word. */
export function useLeaveGuard(active: boolean): void {
  useEffect(() => {
    if (!active) return;
    const ask = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      // Older browsers show the prompt only with a returnValue set.
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', ask);
    return () => window.removeEventListener('beforeunload', ask);
  }, [active]);
}
