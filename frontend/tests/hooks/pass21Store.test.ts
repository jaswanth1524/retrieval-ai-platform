import 'fake-indexeddb/auto';
import { beforeEach, describe, expect, it } from 'vitest';
import { loadChatStore, resetChatStoreConnection, saveChatStore } from '../../src/hooks/chatStore';
import { withStoredConversations } from '../../src/hooks/useChatPersistence';

function stored(id: string, content: string) {
  return {
    version: 2,
    activeConversationId: id,
    conversations: [
      {
        id,
        title: id,
        createdAt: 1,
        updatedAt: 50,
        turns: [
          { id: `${id}-q`, role: 'user', content, sources: [], timestamp: 1, timings: null, traceId: null },
        ],
      },
    ],
  };
}

beforeEach(async () => {
  resetChatStoreConnection();
  await saveChatStore(null);
});

describe('pass 21: chat store saves', () => {
  it('writes nothing when the merge throws, instead of this tab’s copy', async () => {
    const theirs = stored('other-tab', 'kept');
    await saveChatStore(theirs);

    const saved = await saveChatStore(stored('mine', 'mine'), () => {
      throw new Error('unreadable');
    });

    expect(saved).toBe(false);
    expect(await loadChatStore()).toEqual(theirs);
  });

  it('removes the history only when a merge says nothing is worth keeping', async () => {
    await saveChatStore(stored('a', 'x'));
    expect(await saveChatStore({}, () => undefined)).toBe(true);
    expect(await loadChatStore()).toBeNull();
  });

  it('keeps another tab’s conversations when this tab holds only an empty one', () => {
    const lonelyEmpty = {
      version: 2 as const,
      activeConversationId: 'e',
      conversations: [{ id: 'e', title: 'New chat', createdAt: 1, updatedAt: 1, turns: [] }],
    };
    const merged = withStoredConversations(lonelyEmpty as never, stored('other-tab', 'kept'), null) as {
      conversations: { id: string }[];
    };
    expect(merged.conversations.map((c) => c.id)).toContain('other-tab');
  });
});

describe('pass 21: another tab’s save', () => {
  it('is announced and heard', async () => {
    const { announceHistorySaved, onHistorySaved } = await import('../../src/utils/chatChannel');
    const heard = new Promise<void>((resolve) => {
      const stop = onHistorySaved(() => {
        stop();
        resolve();
      });
    });
    announceHistorySaved();
    await heard;
  });

  it('brings a conversation only IndexedDB holds into this tab', async () => {
    const { renderHook, waitFor } = await import('@testing-library/react');
    const { useChat } = await import('../../src/hooks/useChat');
    const { announceHistorySaved } = await import('../../src/utils/chatChannel');
    const { result } = renderHook(() => useChat());
    await waitFor(() => expect(result.current.historyReady).toBe(true));
    await saveChatStore(stored('other-tab', 'asked elsewhere'));
    announceHistorySaved();
    await waitFor(() =>
      expect(result.current.conversations.map((c) => c.id)).toContain('other-tab'),
    );
  });
});
