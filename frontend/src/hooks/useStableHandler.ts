import { useCallback, useLayoutEffect, useRef } from 'react';

/** A function whose identity never changes but always calls the latest `handler`.
 *
 *  For event handlers passed to memoized components: a handler rebuilt each render
 *  re-renders every child it is passed to, and App renders on every streamed frame.
 *  Not for anything called during render — it sees the handler from the last commit. */
export function useStableHandler<A extends unknown[], R>(
  handler: (...args: A) => R,
): (...args: A) => R {
  const ref = useRef(handler);
  useLayoutEffect(() => {
    ref.current = handler;
  });
  return useCallback((...args: A) => ref.current(...args), []);
}
