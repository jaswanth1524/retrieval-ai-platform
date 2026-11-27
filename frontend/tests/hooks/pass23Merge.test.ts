import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';

const read = vi.fn();
const save = vi.fn(async (_value: unknown) => true);
vi.mock('../../src/hooks/chatStore', () => ({
  chatStoreAvailable: () => true,
  readChatStore: () => read(),
  loadChatStore: async () => null,
  saveChatStore: (value: unknown) => save(value),
  clearChatStore: async () => true,
  resetChatStoreConnection: () => {},
}));

const { useChat } = await import('../../src/hooks/useChat');

afterEach(() => {
  read.mockReset();
  save.mockClear();
  localStorage.clear();
});

function history(...ids: string[]) {
  return {
    status: 'data',
    value: {
      version: 2,
      activeConversationId: ids[0],
      conversations: ids.map((id, index) => {
        const at = 1_700_000_000_000 + index * 1000;
        return {
          id,
          title: `Chat ${id}`,
          createdAt: at,
          updatedAt: at,
          turns: [
            { id: `u-${id}`, role: 'user', content: 'Q', sources: [], timestamp: at, timings: null, traceId: null },
          ],
        };
      }),
    },
  };
}

// Longer than the hold before an IndexedDB write.
const settle = () => act(() => new Promise<void>((resolve) => setTimeout(resolve, 700)));
const savedText = () => JSON.stringify(save.mock.calls.map((call) => call[0]));

async function loaded() {
  read.mockResolvedValue(history('mine'));
  const rendered = renderHook(() => useChat());
  await waitFor(() => expect(rendered.result.current.historyReady).toBe(true));
  await settle();
  save.mockClear();
  return rendered;
}

function otherTabSaved(): void {
  const channel = new BroadcastChannel('docrag-chat');
  channel.postMessage({ type: 'saved', from: 'another-tab' });
  channel.close();
}

it('a merge from another tab alone is not written back', async () => {
  const { result } = await loaded();
  read.mockResolvedValue(history('mine', 'theirs'));
  otherTabSaved();
  await waitFor(() => expect(result.current.conversations.map((c) => c.id)).toContain('theirs'));
  await settle();
  expect(save).not.toHaveBeenCalled();
});

it('an edit made here as the other tab’s change is merged is still written', async () => {
  const { result } = await loaded();
  let answer: (value: unknown) => void = () => {};
  const reading = new Promise((resolve) => {
    answer = resolve;
  });
  read.mockReturnValue(reading);
  otherTabSaved();
  await waitFor(() => expect(read).toHaveBeenCalledTimes(2));
  // Both before React renders again.
  await act(async () => {
    result.current.renameConversation('mine', 'Renamed here');
    answer(history('mine', 'theirs'));
    await reading;
  });
  await waitFor(() => expect(result.current.conversations.map((c) => c.id)).toContain('theirs'));
  await settle();
  expect(savedText()).toContain('Renamed here');
});
