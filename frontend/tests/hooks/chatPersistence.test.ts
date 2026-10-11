import { describe, expect, it } from 'vitest';
import type { ChatTurn } from '../../src/components/ChatMessage';
import {
  CHAT_STORAGE_MAX_TURNS,
  MAX_CONVERSATIONS,
  MAX_PINNED_CONVERSATIONS,
  type ChatStorageV2,
  type Conversation,
  TOMBSTONE_MAX_AGE_MS,
  capConversations,
  mergeConversations,
  parseConversations,
  sanitizeConversation,
  searchConversations,
} from '../../src/hooks/chatPersistence';

function turn(id: string, role: ChatTurn['role'] = 'user'): ChatTurn {
  return { id, role, content: id, sources: [], timestamp: 1, timings: null, traceId: null };
}

function conversation(overrides: Partial<Conversation> = {}): Conversation {
  return { id: 'c1', title: 'T', createdAt: 1, updatedAt: 5, turns: [turn('t1')], ...overrides };
}

function storage(conversations: Conversation[], extra: Partial<ChatStorageV2> = {}): ChatStorageV2 {
  return { version: 2, activeConversationId: conversations[0]?.id ?? null, conversations, ...extra };
}

describe('mergeConversations across tabs', () => {
  it("keeps this tab's older turns when another tab's copy was cut to the storage cap", () => {
    // The cut copy was merged as-is, and the next IndexedDB save lost the older turns.
    const all = Array.from({ length: 130 }, (_, index) => turn(`t${index}`));
    const local = storage([conversation({ turns: all, updatedAt: 5 })]);
    // The other tab asked one more question, and saved the last 100 of its 131 turns.
    const cut = conversation({ turns: [...all, turn('new')].slice(-CHAT_STORAGE_MAX_TURNS), updatedAt: 9 });

    const merged = mergeConversations(local, [cut]);

    const turns = merged.conversations[0].turns;
    expect(turns).toHaveLength(131);
    expect(turns[0].id).toBe('t0');
    expect(turns.at(-1)?.id).toBe('new');
  });

  it('leaves a copy that is genuinely shorter alone', () => {
    const local = storage([conversation({ turns: [turn('t1'), turn('t2'), turn('t3')] })]);
    const cleared = conversation({ turns: [turn('t3')], updatedAt: 9 });

    expect(mergeConversations(local, [cleared]).conversations[0].turns.map((t) => t.id)).toEqual(['t3']);
  });

  it("takes another tab's rename, though renaming doesn't move updatedAt", () => {
    // The merge only took strictly newer updatedAt, so the rename was overwritten by
    // this tab's next save and gone after a reload.
    const local = storage([conversation({ title: 'Old', updatedAt: 5 })]);
    const renamed = conversation({ title: 'New', updatedAt: 5, editedAt: 8 });

    expect(mergeConversations(local, [renamed]).conversations[0].title).toBe('New');
    // And this tab's own later edit is not undone by an older copy.
    const mine = storage([conversation({ title: 'Mine', updatedAt: 5, editedAt: 9 })]);
    expect(mergeConversations(mine, [renamed]).conversations[0].title).toBe('Mine');
  });

  it('keeps a conversation deleted in another tab deleted', () => {
    // "Losing a delete beats losing a conversation": tab B's next save brought it back.
    const local = storage([conversation({ id: 'keep' }), conversation({ id: 'gone' })]);
    const now = 100_000;

    const merged = mergeConversations(local, [conversation({ id: 'keep' })], false, [{ id: 'gone', at: 50 }], now);

    expect(merged.conversations.map((c) => c.id)).toEqual(['keep']);
    expect(merged.deleted).toEqual([{ id: 'gone', at: 50 }]);
    // Another tab's stale copy of it, arriving later, doesn't resurrect it either.
    const again = mergeConversations(merged, [conversation({ id: 'gone', updatedAt: 5 })], false, undefined, now);
    expect(again.conversations.map((c) => c.id)).toEqual(['keep']);
  });

  it('keeps a deleted conversation that was used again after the delete', () => {
    const local = storage([conversation({ id: 'c1', updatedAt: 90 })]);

    const merged = mergeConversations(local, [], false, [{ id: 'c1', at: 50 }], 100);

    expect(merged.conversations.map((c) => c.id)).toEqual(['c1']);
  });

  it('forgets deletes older than the tombstone age', () => {
    const now = TOMBSTONE_MAX_AGE_MS * 2;
    const local = storage([conversation()], { deleted: [{ id: 'old', at: 1 }] });

    expect(mergeConversations(local, [], false, undefined, now).deleted).toEqual([]);
  });
});

describe('sanitizeConversation', () => {
  it('drops what a hand-edited import could carry that crashes rendering', () => {
    // model: {} was rendered as a React child and a null source dereferenced: once
    // saved, the page crashed on every load.
    const parsed = sanitizeConversation({
      id: 'c1',
      turns: [
        {
          ...turn('a', 'assistant'),
          model: { name: 'x' },
          timings: {},
          feedback: 'great',
          traceId: 7,
          sources: [
            null,
            { source_number: 1, filename: 'a.pdf', page: 1, section: 'S', chunk_id: 'c', text: 't' },
            { filename: 'b.pdf' },
          ],
        },
        { role: 'user' },
      ],
    });

    expect(parsed?.turns).toHaveLength(1);
    const [answer] = parsed!.turns;
    expect(answer.model).toBeUndefined();
    expect(answer.timings).toBeNull();
    expect(answer.feedback).toBeUndefined();
    expect(answer.traceId).toBeNull();
    expect(answer.sources.map((s) => s.filename)).toEqual(['a.pdf']);
  });

  it('keeps every valid field, editedAt included', () => {
    const valid: ChatTurn = {
      ...turn('a', 'assistant'),
      model: 'llama3.1:8b',
      feedback: 'up',
      stopped: true,
      traceId: 'tr',
      timings: { embed_ms: 1, search_ms: 1, rerank_ms: 1, generate_ms: 1, total_ms: 4, condense_ms: 0 },
    };
    const parsed = parseConversations({ version: 2, conversations: [conversation({ turns: [valid], editedAt: 7 })] });

    expect(parsed?.[0].turns[0]).toEqual(valid);
    expect(parsed?.[0].editedAt).toBe(7);
  });

  it('replaces a time a Date cannot hold, which crashed every render', () => {
    const parsed = parseConversations({
      version: 2,
      conversations: [
        conversation({
          createdAt: Number.MAX_VALUE,
          updatedAt: -1e300,
          editedAt: 9e15,
          turns: [{ ...turn('a'), timestamp: 1e300 }],
        }),
      ],
    });

    const [saved] = parsed!;
    expect(() => new Date(saved.turns[0].timestamp).toISOString()).not.toThrow();
    expect(() => new Date(saved.createdAt).toISOString()).not.toThrow();
    expect(() => new Date(saved.updatedAt).toISOString()).not.toThrow();
    expect(saved.editedAt).toBeUndefined();
  });
});

describe('mergeConversations with an answer streaming', () => {
  it('neither drops nor replaces the streaming conversation', () => {
    const local = storage([conversation({ id: 's', turns: [turn('q'), turn('a', 'assistant')], updatedAt: 5 })]);
    const theirs = conversation({ id: 's', title: 'Renamed', turns: [], updatedAt: 5, editedAt: 9 });

    const renamed = mergeConversations(local, [theirs], false, undefined, 10, 's');
    expect(renamed.conversations[0].title).toBe('Renamed');
    expect(renamed.conversations[0].turns.map((t) => t.id)).toEqual(['q', 'a']);

    const deleted = mergeConversations(local, [], false, [{ id: 's', at: 9 }], 10, 's');
    expect(deleted.conversations.map((c) => c.id)).toEqual(['s']);
    expect(deleted.activeConversationId).toBe('s');
  });

  it('keeps the streaming conversation past the conversation cap', () => {
    const local = storage([conversation({ id: 's', updatedAt: 1 })]);
    const incoming = Array.from({ length: MAX_CONVERSATIONS }, (_, index) =>
      conversation({ id: `n${index}`, updatedAt: 10 + index }),
    );

    const merged = mergeConversations(local, incoming, false, undefined, 100, 's');
    expect(merged.conversations).toHaveLength(MAX_CONVERSATIONS);
    expect(merged.conversations.some((c) => c.id === 's')).toBe(true);
    expect(merged.activeConversationId).toBe('s');
  });
});

describe('pinned conversations', () => {
  const many = (count: number, pinnedIds: string[] = []) =>
    Array.from({ length: count }, (_, index) =>
      conversation({
        id: `c${index}`,
        updatedAt: count - index, // c0 is the most recently used
        ...(pinnedIds.includes(`c${index}`) ? { pinned: true } : {}),
      }),
    );

  it('drops the least recently used unpinned conversations past the cap', () => {
    const kept = capConversations(many(MAX_CONVERSATIONS + 3, ['c21', 'c22']));

    expect(kept).toHaveLength(MAX_CONVERSATIONS);
    const ids = kept.map((c) => c.id);
    expect(ids).toContain('c21'); // pinned, though among the oldest
    expect(ids).toContain('c22');
    expect(ids).not.toContain('c20');
    expect(ids).not.toContain('c19');
    // Still most recently used first.
    expect(ids[0]).toBe('c0');
  });

  it('never drops the conversation being created or streamed into', () => {
    const list = many(MAX_CONVERSATIONS + 1);
    const kept = capConversations(list, `c${MAX_CONVERSATIONS}`);
    expect(kept.map((c) => c.id)).toContain(`c${MAX_CONVERSATIONS}`);
    expect(kept).toHaveLength(MAX_CONVERSATIONS);
  });

  it('protects at most the allowed number of pins', () => {
    const pinned = Array.from({ length: MAX_PINNED_CONVERSATIONS + 2 }, (_, i) => `c${i + 5}`);
    const kept = capConversations(many(MAX_CONVERSATIONS + 5, pinned));
    expect(kept).toHaveLength(MAX_CONVERSATIONS);
  });

  it('reads a pin back and keeps it out of unrelated values', () => {
    expect(sanitizeConversation({ ...conversation(), pinned: true })?.pinned).toBe(true);
    expect(sanitizeConversation({ ...conversation(), pinned: 'yes' })?.pinned).toBeUndefined();
  });

  it("takes another tab's pin like a rename, even on the conversation streaming here", () => {
    const local = storage([conversation({ updatedAt: 5 })]);
    const pinnedThere = conversation({ updatedAt: 5, editedAt: 9, pinned: true });

    expect(mergeConversations(local, [pinnedThere]).conversations[0].pinned).toBe(true);
    const streaming = mergeConversations(local, [{ ...pinnedThere, turns: [] }], false, [], 10, 'c1');
    expect(streaming.conversations[0]).toMatchObject({ pinned: true, turns: [turn('t1')] });
  });
});

describe('searchConversations', () => {
  it('finds the first turn whose text matches, with a snippet around it', () => {
    const answer: ChatTurn = {
      ...turn('a1', 'assistant'),
      content: `${'lead in words '.repeat(6)}The travel policy allows economy fares only.`,
    };
    const conversations = [
      conversation({ id: 'c1', turns: [turn('q1'), answer] }),
      conversation({ id: 'c2', turns: [turn('nothing here')] }),
    ];

    const matches = searchConversations(conversations, '  TRAVEL policy ');

    expect([...matches.keys()]).toEqual(['c1']);
    expect(matches.get('c1')?.turnId).toBe('a1');
    expect(matches.get('c1')?.snippet).toMatch(/^….*travel policy allows economy fares only\.$/);
    expect(searchConversations(conversations, '   ').size).toBe(0);
  });

  it('skips error turns', () => {
    const error: ChatTurn = { ...turn('e1', 'error'), content: 'Request failed: timeout' };
    expect(searchConversations([conversation({ turns: [error] })], 'timeout').size).toBe(0);
  });
});
