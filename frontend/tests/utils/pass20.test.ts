import 'fake-indexeddb/auto';
import { renderHook } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import type { ChatTurn } from '../../src/components/ChatMessage';
import { loadChatStore, resetChatStoreConnection, saveChatStore } from '../../src/hooks/chatStore';
import { useAnswerQueries } from '../../src/hooks/chat/answers';
import type { ChatStorageV2 } from '../../src/hooks/chatPersistence';
import { mergeRegenerated, withVersionRestored } from '../../src/utils/answerVersions';
import { parsePassageLink, passageLinkUrl } from '../../src/utils/deepLink';

function turn(id: string, role: ChatTurn['role'], content: string, extra: Partial<ChatTurn> = {}): ChatTurn {
  return { id, role, content, sources: [], timestamp: 1, timings: null, traceId: null, ...extra };
}

describe('pass 20: passage links to documents in folders', () => {
  it('open, and names that climb or hide are still refused', () => {
    const url = passageLinkUrl(
      { filename: 'reports/2024/q1.pdf', chunkId: 'abc' },
      { origin: 'https://docs.example', pathname: '/' },
    );
    expect(parsePassageLink(new URL(url).hash)).toEqual({ filename: 'reports/2024/q1.pdf', chunkId: 'abc' });
    for (const bad of ['reports/../x.pdf', '/x.pdf', 'a//b.pdf', 'a/./b.pdf', 'a\\b.pdf', 'a/']) {
      expect(parsePassageLink(`#d=${encodeURIComponent(bad)}`)).toBeNull();
    }
  });
});

describe('pass 20: answer versions', () => {
  it('a regenerated answer keeps the note the reader wrote on it', () => {
    const turns = [
      turn('q1', 'user', 'Why?'),
      turn('a1', 'assistant', 'Because.', { saved: true, note: 'check with HR' }),
      turn('q2', 'user', 'Why?'),
      turn('a2', 'assistant', 'Because, again.'),
    ];
    const merged = mergeRegenerated(turns, 'a1', 'a2');
    expect(merged[1]).toMatchObject({ id: 'a2', saved: true, note: 'check with HR' });
  });

  it("restoring an earlier version drops the replaced answer's checks and notes", () => {
    const answer = turn('a', 'assistant', 'New text', {
      unsupportedClaims: 2,
      unsupportedNumbers: 1,
      promptCut: true,
      searchedAs: 'rewritten',
      cannotContinue: true,
      continueFailed: 'timed out',
      versions: [{ content: 'Old text', sources: [], timestamp: 0 }],
    });
    const restored = withVersionRestored(answer, 0);
    expect(restored.content).toBe('Old text');
    for (const field of [
      'unsupportedClaims',
      'unsupportedNumbers',
      'promptCut',
      'searchedAs',
      'cannotContinue',
      'continueFailed',
    ] as const) {
      expect(restored[field]).toBeUndefined();
    }
  });
});

describe('pass 20: Needs review', () => {
  it("lists a summary as a task and a narrowed question with its narrowing, so Ask again keeps them", () => {
    const storage = {
      version: 2,
      activeConversationId: 'c1',
      conversations: [
        {
          id: 'c1',
          title: 'T',
          createdAt: 1,
          updatedAt: 1,
          turns: [
            turn('q1', 'user', 'Summarize a.pdf', { task: { kind: 'summarize', filenames: ['a.pdf'] } } as unknown as Partial<ChatTurn>),
            turn('a1', 'assistant', 'A summary.'),
            turn('q2', 'user', 'Leave?', { narrowing: { section: 'Leave' } } as Partial<ChatTurn>),
            turn('a2', 'assistant', '28 days.', { timestamp: 2 }),
          ],
        },
      ],
    } as unknown as ChatStorageV2;
    const { result } = renderHook(() => useAnswerQueries(storage.conversations, { current: storage }));
    const [narrowed, summary] = result.current.answersIn('');
    expect(summary).toMatchObject({ question: 'Summarize a.pdf', task: true });
    expect(narrowed).toMatchObject({ question: 'Leave?', narrowing: { section: 'Leave' } });
  });
});

describe('pass 20: IndexedDB writes from two tabs', () => {
  it("keep the other tab's conversations instead of overwriting them", async () => {
    resetChatStoreConnection();
    const conversation = (id: string) => ({
      id,
      title: id,
      createdAt: 1,
      updatedAt: 5,
      turns: [turn(`${id}-q`, 'user', 'Q'), turn(`${id}-a`, 'assistant', 'A')],
    });
    const otherTab = { version: 2, activeConversationId: 'b', conversations: [conversation('b')] };
    await saveChatStore(otherTab);
    const mine = { version: 2, activeConversationId: 'a', conversations: [conversation('a')] };
    await saveChatStore(mine, (stored) => {
      const kept = (stored as typeof otherTab).conversations;
      return { ...mine, conversations: [...mine.conversations, ...kept] };
    });
    const saved = (await loadChatStore()) as typeof otherTab;
    expect(saved.conversations.map((c) => c.id).sort()).toEqual(['a', 'b']);
  });
});

describe('pass 20: what a tab writes to IndexedDB', () => {
  it("adds the other tab's conversations and never brings back one deleted here", async () => {
    const { withStoredConversations } = await import('../../src/hooks/useChatPersistence');
    const conversation = (id: string, updatedAt = 5) => ({
      id,
      title: id,
      createdAt: 1,
      updatedAt,
      turns: [turn(`${id}-q`, 'user', 'Q'), turn(`${id}-a`, 'assistant', 'A')],
    });
    const mine = {
      version: 2,
      activeConversationId: 'a',
      conversations: [conversation('a')],
      deleted: [{ id: 'gone', at: Date.now() }],
    } as unknown as ChatStorageV2;
    const stored = {
      version: 2,
      activeConversationId: 'b',
      conversations: [conversation('b'), conversation('gone', 4)],
    };
    const written = withStoredConversations(mine, stored, null) as ChatStorageV2;
    expect(written.conversations.map((c) => c.id).sort()).toEqual(['a', 'b']);
  });
});
