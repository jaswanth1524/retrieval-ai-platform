import { useEffect, useState } from 'react';

/** Phones and other touch screens (or a very narrow window): a row of small icon
 *  buttons is hard to hit and crowds the name it belongs to, so row actions go into a
 *  "⋯" menu (OverflowMenu) of labelled items. */
export const COMPACT_ACTIONS_QUERY = '(hover: none) and (pointer: coarse), (max-width: 480px)';

export function useCompactActions(): boolean {
  const [compact, setCompact] = useState(() => window.matchMedia(COMPACT_ACTIONS_QUERY).matches);
  useEffect(() => {
    const mql = window.matchMedia(COMPACT_ACTIONS_QUERY);
    const handler = (event: MediaQueryListEvent) => setCompact(event.matches);
    mql.addEventListener('change', handler);
    return () => mql.removeEventListener('change', handler);
  }, []);
  return compact;
}
