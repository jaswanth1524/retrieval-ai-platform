import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { MAX_CONVERSATIONS } from '../../src/hooks/chat/model';
import { readTrash } from '../../src/utils/trash';

const read = vi.fn();
vi.mock('../../src/hooks/chatStore', () => ({
  chatStoreAvailable: () => true,
  readChatStore: () => read(),
  loadChatStore: async () => null,
  saveChatStore: async () => true,
  clearChatStore: async () => true,
  resetChatStoreConnection: () => {},
}));

const { useChat } = await import('../../src/hooks/useChat');

afterEach(() => {
  read.mockReset();
  localStorage.clear();
});

function conversation(index: number) {
  const at = 1_700_000_000_000 + index * 1000;
  return {
    id: `c${index}`,
    title: `Conversation ${index}`,
    createdAt: at,
    updatedAt: at,
    turns: [
      { id: `u${index}`, role: 'user', content: `Question ${index}`, sources: [], timestamp: at, timings: null, traceId: null },
    ],
  };
}

it('a conversation the history cap lets go is put in Recently deleted', async () => {
  // With the fresh chat a load starts with, the history is exactly full.
  const conversations = Array.from({ length: MAX_CONVERSATIONS - 1 }, (_, index) => conversation(index));
  read.mockResolvedValue({
    status: 'data',
    value: { version: 2, activeConversationId: 'c98', conversations },
  });
  const { result } = renderHook(() => useChat());
  await waitFor(() => expect(result.current.historyReady).toBe(true));
  expect(result.current.conversations).toHaveLength(MAX_CONVERSATIONS);
  const before = result.current.trashedVersion;

  // A new chat past the cap: the least recently used one (c0) goes.
  act(() => result.current.switchConversation('c50'));
  act(() => result.current.newConversation());

  expect(result.current.conversations.some((summary) => summary.id === 'c0')).toBe(false);
  expect(readTrash().map((entry) => entry.conversation.id)).toEqual(['c0']);
  expect(result.current.trashedVersion).toBe(before + 1);

  // A deliberate delete leaves a tombstone: it isn't trashed here a second time.
  act(() => {
    result.current.deleteConversation('c1');
  });
  expect(readTrash().map((entry) => entry.conversation.id)).toEqual(['c0']);
});

it('Clear keeps the turns it removes as a copy in Recently deleted', async () => {
  read.mockResolvedValue({
    status: 'data',
    value: { version: 2, activeConversationId: 'c1', conversations: [conversation(1)] },
  });
  const { result } = renderHook(() => useChat());
  await waitFor(() => expect(result.current.historyReady).toBe(true));

  act(() => result.current.switchConversation('c1'));
  act(() => result.current.clear());

  expect(result.current.turns).toEqual([]);
  const [entry] = readTrash();
  expect(entry.conversation.id).not.toBe('c1');
  expect(entry.conversation.title).toBe('Conversation 1 (cleared)');
  expect(entry.conversation.turns[0].content).toBe('Question 1');
});
