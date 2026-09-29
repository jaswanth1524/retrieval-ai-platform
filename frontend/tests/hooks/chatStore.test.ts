import 'fake-indexeddb/auto';
import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ChatTurn } from '../../src/components/ChatMessage';
import { loadChatStore, resetChatStoreConnection, saveChatStore } from '../../src/hooks/chatStore';
import { useChat } from '../../src/hooks/useChat';

function turn(id: string, role: ChatTurn['role'], content: string, text = ''): ChatTurn {
  return {
    id,
    role,
    content,
    sources: text ? [{ source_number: 1, filename: 'a.pdf', page: 1, section: 'S', chunk_id: 'c', text }] : [],
    timestamp: 1,
    timings: null,
    traceId: null,
  };
}

function history(turns: ChatTurn[]) {
  return {
    version: 2,
    activeConversationId: 'c1',
    conversations: [{ id: 'c1', title: 'Saved', createdAt: 1, updatedAt: 50, turns }],
  };
}

beforeEach(async () => {
  resetChatStoreConnection();
  await saveChatStore(null);
});

afterEach(() => {
  localStorage.clear();
  vi.restoreAllMocks();
});

describe('chatStore', () => {
  it('round-trips the history through IndexedDB', async () => {
    const value = history([turn('q', 'user', 'Q')]);

    expect(await saveChatStore(value)).toBe(true);

    expect(await loadChatStore()).toEqual(value);
  });
});

describe('useChat with IndexedDB', () => {
  it('restores passage text that localStorage dropped, from the full copy in IndexedDB', async () => {
    const full = history([turn('q', 'user', 'Q'), turn('a', 'assistant', 'A', 'the full passage')]);
    await saveChatStore(full);
    // localStorage's compact copy of the same conversation, text stripped.
    localStorage.setItem(
      'docrag-chat-history',
      JSON.stringify(history([turn('q', 'user', 'Q'), turn('a', 'assistant', 'A', '')])),
    );

    const { result } = renderHook(() => useChat());

    await waitFor(() => expect(result.current.turns[1]?.sources[0]?.text).toBe('the full passage'));
  });

  it('keeps everything when localStorage is full, without warning the user', async () => {
    const { result } = renderHook(() => useChat());
    await waitFor(async () => expect(await loadChatStore()).not.toBeNull());
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('quota', 'QuotaExceededError');
    });

    act(() => result.current.importConversation([turn('q', 'user', 'Big Q'), turn('a', 'assistant', 'Big A')]));

    await waitFor(async () => {
      const saved = (await loadChatStore()) as ReturnType<typeof history>;
      expect(saved.conversations.some((c) => c.turns.some((t) => t.content === 'Big A'))).toBe(true);
    });
    await waitFor(() => {
      expect(result.current.persistError).toBe(false);
      expect(result.current.persistPartial).toBe(false);
    });
  });
});
