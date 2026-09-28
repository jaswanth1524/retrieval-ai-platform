import { useEffect, useRef } from 'react';
import type { RefObject } from 'react';

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/** Modal behaviour shared by the command palette and the settings modal.
 *
 * Three things, all of which are the difference between a dialog and a div that looks
 * like one: focus moves into it on open, Tab cannot escape it while it is open, and
 * focus returns to whatever opened it on close. Without the last one a keyboard user
 * lands back at the top of the document every time they close a dialog.
 *
 * Returns a ref to put on the dialog container.
 */
export function useDialog(open: boolean, onClose: () => void): RefObject<HTMLDivElement | null> {
  const containerRef = useRef<HTMLDivElement>(null);
  // Captured at open time, because by close time the opener may no longer be focused.
  const openerRef = useRef<HTMLElement | null>(null);
  // Read through a ref so the effect depends on `open` alone. Callers pass inline
  // arrows; with onClose in the deps every parent re-render (upload polling, streaming,
  // toasts) ran the cleanup — refocusing the opener — then refocused the first control,
  // yanking focus out of whatever field the user was typing in.
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    if (!open) return;
    openerRef.current = document.activeElement as HTMLElement | null;

    const container = containerRef.current;
    const focusables = container?.querySelectorAll<HTMLElement>(FOCUSABLE);
    focusables?.[0]?.focus();

    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onCloseRef.current();
        return;
      }
      if (event.key !== 'Tab' || !containerRef.current) return;

      // Re-query on each Tab: the palette's result list changes as the user filters,
      // so a list captured at open time goes stale immediately.
      const items = Array.from(containerRef.current.querySelectorAll<HTMLElement>(FOCUSABLE));
      if (items.length === 0) return;
      const first = items[0];
      const last = items[items.length - 1];
      const active = document.activeElement;

      if (event.shiftKey && (active === first || !containerRef.current.contains(active))) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('keydown', handleKey);
      // Only restore if the opener is still in the document — it may have been
      // unmounted by whatever the dialog just did.
      const opener = openerRef.current;
      if (opener && document.contains(opener)) opener.focus();
    };
  }, [open]);

  return containerRef;
}
