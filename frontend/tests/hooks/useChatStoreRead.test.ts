import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';

// IndexedDB stand-in: the test decides what reading the stored history finds.
const read = vi.fn();
const saveChatStore = vi.fn(async (_value: unknown) => true);
vi.mock('../../src/hooks/chatStore', () => ({
  chatStoreAvailable: () => true,
  readChatStore: () => read(),
  loadChatStore: async () => null,
  saveChatStore: (value: unknown) => saveChatStore(value),
  clearChatStore: async () => true,
  resetChatStoreConnection: () => {},
}));

const { useChat } = await import('../../src/hooks/useChat');

afterEach(() => {
  read.mockReset();
  saveChatStore.mockClear();
  localStorage.clear();
  vi.useRealTimers();
});

it('never saves over a stored history it could not read, and says so', async () => {
  read.mockResolvedValue({ status: 'failed' });
  const { result } = renderHook(() => useChat());
  await waitFor(() => expect(result.current.persistPartial).toBe(true));
  // What is loaded is the partial copy: nothing may be pruned against it.
  expect(result.current.historyReady).toBe(false);

  vi.useFakeTimers();
  act(() => {
    result.current.newConversation();
    result.current.renameConversation(result.current.activeConversationId!, 'Renamed');
  });
  act(() => {
    vi.advanceTimersByTime(5000);
    window.dispatchEvent(new Event('pagehide'));
  });

  expect(saveChatStore).not.toHaveBeenCalled();
  expect(result.current.persistPartial).toBe(true);
});

it('saves once the stored history was read, even when it was empty', async () => {
  read.mockResolvedValue({ status: 'empty' });
  const { result } = renderHook(() => useChat());
  await waitFor(() => expect(result.current.historyReady).toBe(true));

  act(() => {
    window.dispatchEvent(new Event('pagehide'));
  });

  expect(saveChatStore).toHaveBeenCalled();
  expect(result.current.persistPartial).toBe(false);
});
