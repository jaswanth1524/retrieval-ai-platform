import { useEffect, useRef } from 'react';
import type { RefObject } from 'react';

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]';

/** What Tab can actually reach inside ``container``, in order.
 *
 * ``summary`` (a <details> disclosure) is reachable but wasn't listed, so the trap
 * wrapped before it; and a button taken out of the tab order (an inactive tab,
 * tabindex=-1) was listed, so it could be the "last" item the trap waited for. */
function tabbable(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
    (element) => element.getAttribute('tabindex') !== '-1',
  );
}

// Open dialogs, innermost last. Several can be open at once (the document viewer opened
// from the inspector drawer), and only the innermost may react to Escape or trap Tab —
// otherwise one Escape closed both, and the outer trap pulled focus out of the inner.
const openDialogs: symbol[] = [];

/** Modal behaviour shared by the command palette, the settings modal, the document
 *  viewer and the narrow-screen drawers.
 *
 * Three things, all of which are the difference between a dialog and a div that looks
 * like one: focus moves into it on open, Tab cannot escape it while it is open, and
 * focus returns to whatever opened it on close. Without the last one a keyboard user
 * lands back at the top of the document every time they close a dialog.
 *
 * Returns a ref to put on the dialog container.
 */
export function useDialog<T extends HTMLElement = HTMLDivElement>(
  open: boolean,
  onClose: () => void,
): RefObject<T | null> {
  const containerRef = useRef<T>(null);
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
    const id = Symbol('dialog');
    openDialogs.push(id);
    openerRef.current = document.activeElement as HTMLElement | null;

    const container = containerRef.current;
    if (container) tabbable(container)[0]?.focus();

    const handleKey = (event: KeyboardEvent) => {
      if (openDialogs[openDialogs.length - 1] !== id) return;
      if (event.key === 'Escape') {
        event.preventDefault();
        onCloseRef.current();
        return;
      }
      if (event.key !== 'Tab' || !containerRef.current) return;

      // Re-query on each Tab: the palette's result list changes as the user filters,
      // so a list captured at open time goes stale immediately.
      const items = tabbable(containerRef.current);
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
      openDialogs.splice(openDialogs.indexOf(id), 1);
      // Only restore if the opener is still in the document — it may have been
      // unmounted by whatever the dialog just did.
      const opener = openerRef.current;
      if (opener && document.contains(opener)) opener.focus();
    };
  }, [open]);

  return containerRef;
}
