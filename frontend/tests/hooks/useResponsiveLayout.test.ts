import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  INSPECTOR_COLLAPSE_PX,
  INSPECTOR_RESTORE_PX,
  useResponsiveLayout,
} from '../../src/hooks/useResponsiveLayout';

/** A matchMedia stub whose breakpoints can be crossed on demand. */
function installMatchMedia(initialWidth: number) {
  const listeners = new Map<string, Set<(e: MediaQueryListEvent) => void>>();
  let width = initialWidth;

  const evaluate = (query: string) => {
    const max = /max-width:\s*(\d+)px/.exec(query);
    if (max) return width <= Number(max[1]);
    const min = /min-width:\s*(\d+)px/.exec(query);
    if (min) return width >= Number(min[1]);
    return false;
  };

  vi.stubGlobal('matchMedia', (query: string) => ({
    get matches() {
      return evaluate(query);
    },
    media: query,
    onchange: null,
    addEventListener: (_type: string, fn: (e: MediaQueryListEvent) => void) => {
      if (!listeners.has(query)) listeners.set(query, new Set());
      listeners.get(query)!.add(fn);
    },
    removeEventListener: (_type: string, fn: (e: MediaQueryListEvent) => void) => {
      listeners.get(query)?.delete(fn);
    },
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  }));

  return {
    resizeTo(next: number) {
      width = next;
      listeners.forEach((fns, query) => {
        fns.forEach((fn) => fn({ matches: evaluate(query) } as MediaQueryListEvent));
      });
    },
    listenerCount: () => [...listeners.values()].reduce((sum, set) => sum + set.size, 0),
  };
}

describe('useResponsiveLayout', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('reports a narrow viewport on first render, without waiting for a resize', () => {
    installMatchMedia(900);

    const { result } = renderHook(() => useResponsiveLayout());

    expect(result.current.tooNarrowForInspector).toBe(true);
    expect(result.current.roomyEnoughForInspector).toBe(false);
  });

  it('reports a roomy viewport on first render', () => {
    installMatchMedia(1440);

    const { result } = renderHook(() => useResponsiveLayout());

    expect(result.current.tooNarrowForInspector).toBe(false);
    expect(result.current.roomyEnoughForInspector).toBe(true);
  });

  it('leaves a mid-range viewport in neither state, so the inspector is left alone', () => {
    // Between the two breakpoints nothing should be forced either way — that gap is
    // what stops the panel flapping open and shut around a single threshold.
    installMatchMedia((INSPECTOR_COLLAPSE_PX + INSPECTOR_RESTORE_PX) / 2);

    const { result } = renderHook(() => useResponsiveLayout());

    expect(result.current.tooNarrowForInspector).toBe(false);
    expect(result.current.roomyEnoughForInspector).toBe(false);
  });

  it('updates when a breakpoint is crossed', () => {
    const media = installMatchMedia(1440);
    const { result } = renderHook(() => useResponsiveLayout());

    act(() => media.resizeTo(900));

    expect(result.current.tooNarrowForInspector).toBe(true);
    expect(result.current.roomyEnoughForInspector).toBe(false);
  });

  it('removes its listeners on unmount', () => {
    const media = installMatchMedia(1440);
    const { unmount } = renderHook(() => useResponsiveLayout());
    expect(media.listenerCount()).toBe(2);

    unmount();

    expect(media.listenerCount()).toBe(0);
  });
});
