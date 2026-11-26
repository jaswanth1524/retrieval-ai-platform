import { useCallback, useEffect, useRef, useState } from 'react';
import type { SourceView } from './useLinks';

// Places kept to go back to (each a passage opened from a citation, a search, a link).
const TRAIL_MAX = 30;

function samePlace(a: SourceView | null, b: SourceView | null): boolean {
  return a?.filename === b?.filename && a?.chunkId === b?.chunkId;
}

/** Back and forward through the passages opened in the document viewer: following
 *  one answer's citations [1], [2], [3] in the docked viewer replaced each with the
 *  next, and the one before was found again only by clicking its citation again. Ends
 *  (and forgets) when the viewer closes. */
export function useViewerTrail(
  view: SourceView | null,
  setView: (view: SourceView | null) => void,
): { back?: () => void; forward?: () => void } {
  // The stacks in a ref (read by back/forward); a state copy only to re-render.
  const trailRef = useRef<{ back: SourceView[]; forward: SourceView[] }>({ back: [], forward: [] });
  const [trail, setTrail] = useState(trailRef.current);
  const keep = (next: { back: SourceView[]; forward: SourceView[] }) => {
    trailRef.current = next;
    setTrail(next);
  };
  const shownRef = useRef<SourceView | null>(view);
  // A move made here (back, forward): the place it went to, not a new one to remember.
  // The place itself, not a flag: a move the parent didn't make (it kept the view) left
  // the flag set, and the next real navigation wasn't remembered.
  const steppingRef = useRef<SourceView | null>(null);

  useEffect(() => {
    const before = shownRef.current;
    shownRef.current = view;
    const stepped = steppingRef.current;
    steppingRef.current = null;
    if (view === null) {
      keep({ back: [], forward: [] });
      return;
    }
    if (stepped !== null && samePlace(stepped, view)) return;
    if (before !== null && !samePlace(before, view)) {
      keep({ back: [...trailRef.current.back, before].slice(-TRAIL_MAX), forward: [] });
    }
  }, [view]);

  const back = useCallback(() => {
    const current = shownRef.current;
    const { back: behind, forward: ahead } = trailRef.current;
    const target = behind[behind.length - 1];
    if (!target || !current) return;
    keep({ back: behind.slice(0, -1), forward: [current, ...ahead] });
    // Now, not after the render: a second Back pressed before it pushed the place
    // already left onto the forward stack again, and lost the one between.
    shownRef.current = target;
    steppingRef.current = target;
    setView(target);
  }, [setView]);

  const forward = useCallback(() => {
    const current = shownRef.current;
    const { back: behind, forward: ahead } = trailRef.current;
    const [target, ...rest] = ahead;
    if (!target || !current) return;
    keep({ back: [...behind, current], forward: rest });
    shownRef.current = target;
    steppingRef.current = target;
    setView(target);
  }, [setView]);

  return {
    back: trail.back.length > 0 ? back : undefined,
    forward: trail.forward.length > 0 ? forward : undefined,
  };
}
