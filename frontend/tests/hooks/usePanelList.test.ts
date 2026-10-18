import { act, renderHook, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ApiClientError } from '../../src/api/client';
import { usePanelList } from '../../src/hooks/usePanelList';

describe('usePanelList', () => {
  it('loads each time the panel opens, and a Refresh beats the load it replaces', async () => {
    const resolvers: ((items: string[]) => void)[] = [];
    const signals: AbortSignal[] = [];
    const fetchItems = vi.fn(
      (signal: AbortSignal) =>
        new Promise<string[]>((resolve) => {
          signals.push(signal);
          resolvers.push(resolve);
        }),
    );
    const { result, rerender } = renderHook(
      ({ active }) => usePanelList(active, fetchItems, vi.fn(), 'Could not load.'),
      { initialProps: { active: false } },
    );
    expect(fetchItems).not.toHaveBeenCalled();

    rerender({ active: true });
    expect(fetchItems).toHaveBeenCalledTimes(1);
    act(() => result.current.load());
    expect(signals[0].aborted).toBe(true);
    // The first (aborted) load answering late doesn't overwrite the second.
    await act(async () => {
      resolvers[1](['new']);
      resolvers[0](['stale']);
    });
    expect(result.current.items).toEqual(['new']);

    // Closing the panel aborts what is in flight; reopening loads again.
    rerender({ active: false });
    rerender({ active: true });
    expect(fetchItems).toHaveBeenCalledTimes(3);
    rerender({ active: false });
    expect(signals[2].aborted).toBe(true);
  });

  it("shows the server's message and passes the error on", async () => {
    const onError = vi.fn();
    const { result } = renderHook(() =>
      usePanelList(true, () => Promise.reject(new ApiClientError('Missing key.', 401)), onError, 'Could not load.'),
    );
    await waitFor(() => expect(result.current.error).toBe('Missing key.'));
    expect(onError).toHaveBeenCalledWith(expect.any(ApiClientError));
    expect(result.current.items).toBeNull();
  });
});
