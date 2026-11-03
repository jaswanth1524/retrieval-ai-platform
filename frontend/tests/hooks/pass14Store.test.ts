import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';

// IndexedDB stand-in whose read the test finishes when it chooses.
let finishRead: (value: unknown) => void = () => {};
vi.mock('../../src/hooks/chatStore', () => ({
  chatStoreAvailable: () => true,
  readChatStore: () =>
    new Promise((resolve) => {
      finishRead = (value) => resolve({ status: 'data', value });
    }),
  loadChatStore: async () => null,
  saveChatStore: async () => true,
  clearChatStore: async () => true,
  resetChatStoreConnection: () => {},
}));

const { useChat } = await import('../../src/hooks/useChat');

afterEach(() => localStorage.clear());

it('a listing before the stored history loads prunes the loaded history, losing nothing', async () => {
  const turns = Array.from({ length: 10 }, (_, n) => ({
    id: `t${n}`,
    role: n % 2 ? 'assistant' : 'user',
    content: `Turn ${n} with its passage text`,
    sources: [],
    timestamp: 1,
    timings: null,
    traceId: null,
  }));
  const stored = {
    version: 2,
    activeConversationId: 'c1',
    conversations: [
      {
        id: 'c1',
        title: 'Policy talk',
        createdAt: 1000,
        updatedAt: 2000,
        turns,
        scope: { filenames: ['gone.txt', 'kept.txt'], tags: [] },
      },
    ],
  };
  const { result } = renderHook(() => useChat());

  // The corpus listing lands first: gone.txt was deleted elsewhere.
  act(() => result.current.pruneScopes(['kept.txt']));
  await act(async () => {
    finishRead(stored);
  });

  await waitFor(() =>
    expect(result.current.conversations.some((conversation) => conversation.id === 'c1')).toBe(
      true,
    ),
  );
  act(() => result.current.switchConversation('c1'));
  await waitFor(() => expect(result.current.scope.filenames).toEqual(['kept.txt']));
  // Every stored turn is still there: the prune didn't make a stub win the merge.
  expect(result.current.turns).toHaveLength(10);
});
