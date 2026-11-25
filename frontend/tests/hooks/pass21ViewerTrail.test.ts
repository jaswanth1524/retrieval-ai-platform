import { act, renderHook } from '@testing-library/react';
import { useState } from 'react';
import { describe, expect, it } from 'vitest';
import type { SourceView } from '../../src/hooks/useLinks';
import { useViewerTrail } from '../../src/hooks/useViewerTrail';

function useHarness() {
  const [view, setView] = useState<SourceView | null>(null);
  const trail = useViewerTrail(view, setView);
  return { view, setView, trail };
}

const place = (chunkId: string): SourceView => ({ filename: 'a.txt', chunkId });

describe('pass 21: back and forward in the viewer', () => {
  it('goes back through the passages opened, and forward again', () => {
    const { result } = renderHook(useHarness);
    act(() => result.current.setView(place('c1')));
    expect(result.current.trail.back).toBeUndefined();
    act(() => result.current.setView(place('c2')));
    act(() => result.current.setView(place('c3')));
    act(() => result.current.trail.back?.());
    expect(result.current.view?.chunkId).toBe('c2');
    act(() => result.current.trail.back?.());
    expect(result.current.view?.chunkId).toBe('c1');
    expect(result.current.trail.back).toBeUndefined();
    act(() => result.current.trail.forward?.());
    expect(result.current.view?.chunkId).toBe('c2');
    // A new passage opened drops what was ahead.
    act(() => result.current.setView(place('c9')));
    expect(result.current.trail.forward).toBeUndefined();
    act(() => result.current.trail.back?.());
    expect(result.current.view?.chunkId).toBe('c2');
  });

  it('forgets the trail when the viewer closes', () => {
    const { result } = renderHook(useHarness);
    act(() => result.current.setView(place('c1')));
    act(() => result.current.setView(place('c2')));
    act(() => result.current.setView(null));
    act(() => result.current.setView(place('c3')));
    expect(result.current.trail.back).toBeUndefined();
  });
});
