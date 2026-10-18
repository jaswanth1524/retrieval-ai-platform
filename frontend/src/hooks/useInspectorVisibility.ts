import { useCallback, useEffect, useRef, useState } from 'react';
import { useStableHandler } from './useStableHandler';

/**
 * Whether the inspector is open: width-driven until the user takes a position.
 *
 * It opens on a roomy window and closes on a narrow one — until the user opens or
 * closes it themselves; after that the width rule no longer overrides them, or crossing
 * a breakpoint silently undid what they just did. One exception: an inspector opened on
 * a wide window still closes when the window narrows, since below that width it is a
 * modal drawer, which took focus and trapped Tab for an inspector nobody had just asked
 * for.
 */
export function useInspectorVisibility(
  tooNarrow: boolean,
  roomyEnough: boolean,
  /** Called when the toggle opens a closed inspector. */
  onReopen?: () => void,
) {
  const [open, setOpen] = useState(false);
  const [forced, setForced] = useState(false);

  const openedNarrowRef = useRef(false);
  useEffect(() => {
    if (open) openedNarrowRef.current = tooNarrow;
    // Only when it opens: a resize while open is the effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);
  useEffect(() => {
    if (forced) {
      if (tooNarrow && !openedNarrowRef.current) setOpen(false);
      return;
    }
    if (tooNarrow) setOpen(false);
    else if (roomyEnough) setOpen(true);
  }, [forced, tooNarrow, roomyEnough]);

  /** An explicit request to see it (a trace picked): outranks the width rule. */
  const show = useCallback(() => {
    setOpen(true);
    setForced(true);
  }, []);
  const close = useCallback(() => setOpen(false), []);
  const openRef = useRef(open);
  openRef.current = open;
  const reopened = useStableHandler(() => onReopen?.());
  const toggle = useCallback(() => {
    if (!openRef.current) reopened();
    setForced(true);
    setOpen((prev) => !prev);
  }, [reopened]);

  return { inspectorOpen: open, showInspector: show, closeInspector: close, toggleInspector: toggle };
}
