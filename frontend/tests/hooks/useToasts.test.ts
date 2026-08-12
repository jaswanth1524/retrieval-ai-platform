import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { TOAST_TIMEOUT_MS, useToasts } from '../../src/hooks/useToasts';

describe('useToasts', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('pushes a toast and auto-dismisses it after the timeout', () => {
    const { result } = renderHook(() => useToasts());

    act(() => {
      result.current.push({ tone: 'good', title: 'Settings saved' });
    });
    expect(result.current.toasts).toHaveLength(1);

    act(() => {
      vi.advanceTimersByTime(TOAST_TIMEOUT_MS - 1);
    });
    expect(result.current.toasts).toHaveLength(1);

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(result.current.toasts).toHaveLength(0);
  });

  it('dismisses immediately on request and cancels the pending timer', () => {
    const { result } = renderHook(() => useToasts());

    let id = '';
    act(() => {
      id = result.current.push({ tone: 'info', title: 'Hello' });
    });
    act(() => {
      result.current.dismiss(id);
    });

    expect(result.current.toasts).toHaveLength(0);
    // The timer must not fire into an already-empty list later.
    expect(vi.getTimerCount()).toBe(0);
  });

  it('replaces rather than stacks when the same id is pushed repeatedly', () => {
    // Persistence failures arrive as a state flag, so the effect that surfaces them
    // re-runs on every render. Without id-keyed replacement that stacks dozens of
    // identical notices.
    const { result } = renderHook(() => useToasts());

    act(() => {
      result.current.push({ id: 'persist-error', tone: 'bad', title: 'first' });
      result.current.push({ id: 'persist-error', tone: 'bad', title: 'second' });
      result.current.push({ id: 'persist-error', tone: 'bad', title: 'third' });
    });

    expect(result.current.toasts).toHaveLength(1);
    expect(result.current.toasts[0].title).toBe('third');
  });

  it('clears every pending timer on unmount', () => {
    const { result, unmount } = renderHook(() => useToasts());

    act(() => {
      result.current.push({ tone: 'info', title: 'a' });
      result.current.push({ tone: 'info', title: 'b' });
    });
    expect(vi.getTimerCount()).toBe(2);

    unmount();

    expect(vi.getTimerCount()).toBe(0);
  });
});
