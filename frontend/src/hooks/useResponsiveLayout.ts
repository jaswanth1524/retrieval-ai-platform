import { useEffect, useState } from 'react';

/** Below this the inspector auto-closes; at or above the upper bound it reopens. */
export const INSPECTOR_COLLAPSE_PX = 1120;
export const INSPECTOR_RESTORE_PX = 1320;

export interface ResponsiveLayout {
  /** True while the viewport is too narrow to hold the inspector comfortably. */
  tooNarrowForInspector: boolean;
  /** True once there is room to bring it back on its own. */
  roomyEnoughForInspector: boolean;
}

function listen(query: string, onChange: (matches: boolean) => void): () => void {
  // matchMedia fires only when the breakpoint is actually crossed. A resize listener
  // would re-render on every pixel of a drag to compute the same two booleans.
  const mql = window.matchMedia(query);
  const handler = (event: MediaQueryListEvent) => onChange(event.matches);
  mql.addEventListener('change', handler);
  return () => mql.removeEventListener('change', handler);
}

export function useResponsiveLayout(): ResponsiveLayout {
  const [tooNarrow, setTooNarrow] = useState(
    () => window.matchMedia(`(max-width: ${INSPECTOR_COLLAPSE_PX - 1}px)`).matches,
  );
  const [roomy, setRoomy] = useState(
    () => window.matchMedia(`(min-width: ${INSPECTOR_RESTORE_PX}px)`).matches,
  );

  useEffect(() => {
    const stopNarrow = listen(`(max-width: ${INSPECTOR_COLLAPSE_PX - 1}px)`, setTooNarrow);
    const stopRoomy = listen(`(min-width: ${INSPECTOR_RESTORE_PX}px)`, setRoomy);
    return () => {
      stopNarrow();
      stopRoomy();
    };
  }, []);

  return { tooNarrowForInspector: tooNarrow, roomyEnoughForInspector: roomy };
}
