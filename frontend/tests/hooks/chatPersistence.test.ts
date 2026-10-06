import { describe, expect, it } from 'vitest';
import type { ChatTurn } from '../../src/components/ChatMessage';
import {
  CHAT_STORAGE_MAX_TURNS,
  type ChatStorageV2,
  type Conversation,
  TOMBSTONE_MAX_AGE_MS,
  mergeConversations,
  parseConversations,
  sanitizeConversation,
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
});
