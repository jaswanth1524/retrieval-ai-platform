import { describe, expect, it } from 'vitest';
import type { ChatTurn } from '../../src/components/ChatMessage';
import {
  CHAT_STORAGE_MAX_TURNS,
  FULL_TEXT_CONVERSATIONS,
  LOCAL_STORAGE_MAX_CONVERSATIONS,
  MAX_CONVERSATIONS,
  MAX_PINNED_CONVERSATIONS,
  type ChatStorageV2,
  type Conversation,
  TOMBSTONE_MAX_AGE_MS,
  capConversations,
  compactForStorage,
  conversationsForLocalStorage,
  forChatStore,
  keepPassageText,
  mergeConversations,
  parseConversations,
  conversationWorkspace,
  sanitizeConversation,
  sanitizeTask,
  sanitizeTurn,
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
  it('keeps a valid workspace name and drops anything else', () => {
    const base = { id: 'c1', turns: [turn('q', 'user')] };
    const legal = sanitizeConversation({ ...base, workspace: 'legal' })!;
    expect(legal.workspace).toBe('legal');
    expect(conversationWorkspace(legal)).toBe('legal');
    for (const workspace of ['Legal', '../x', '', 7]) {
      expect(sanitizeConversation({ ...base, workspace })!.workspace).toBeUndefined();
    }
    // No field means the default workspace; no questions means no workspace yet.
    expect(conversationWorkspace(sanitizeConversation(base)!)).toBe('');
    expect(conversationWorkspace(sanitizeConversation({ id: 'c2', turns: [], workspace: 'legal' })!)).toBeNull();
  });

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
    const n = MAX_CONVERSATIONS;
    const kept = capConversations(many(n + 3, [`c${n + 1}`, `c${n + 2}`]));

    expect(kept).toHaveLength(MAX_CONVERSATIONS);
    const ids = kept.map((c) => c.id);
    expect(ids).toContain(`c${n + 1}`); // pinned, though among the oldest
    expect(ids).toContain(`c${n + 2}`);
    expect(ids).not.toContain(`c${n}`);
    expect(ids).not.toContain(`c${n - 1}`);
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

describe('cited passages (spans)', () => {
  const source = {
    source_number: 1,
    filename: 'a.md',
    page: 1,
    section: 'Intro',
    chunk_id: 'c1',
    text: 'Alpha is documented.',
    spans: [{ start: 0, end: 20, quote: 'Alpha is documented.' }],
  };
  const answer = (id: string): ChatTurn => ({ ...turn(id, 'assistant'), sources: [source] });

  it('keeps valid spans and drops spans that would crash rendering, not the citation', () => {
    expect(sanitizeTurn(answer('a'))?.sources[0].spans).toEqual(source.spans);
    const broken = sanitizeTurn({
      ...answer('a'),
      sources: [{ ...source, spans: [{ start: 0, quote: 7 }] }],
    });
    expect(broken?.sources).toHaveLength(1);
    expect(broken?.sources[0]).not.toHaveProperty('spans');
  });

  it('drops spans with the text on older turns, and gets both back from a full copy', () => {
    const conversation: Conversation = {
      ...sanitizeConversation({ id: 'c', title: 't', updatedAt: 1, turns: [] })!,
      turns: Array.from({ length: 8 }, (_, index) => answer(`t${index}`)),
    };

    const stored = compactForStorage(conversation, 100);

    expect(stored.turns[0].sources[0]).not.toHaveProperty('spans');
    expect(stored.turns[0].sources[0].text).toBe('');
    expect(stored.turns[7].sources[0].spans).toEqual(source.spans);
    expect(keepPassageText(stored, conversation).turns[0].sources[0]).toEqual(source);
  });
});


describe('document tasks (summaries)', () => {
  it('keeps a task the server would accept and drops one it would refuse', () => {
    expect(sanitizeTask({ mode: 'summarize', filenames: ['a.pdf'], instruction: 'risks' })).toEqual({
      mode: 'summarize',
      filenames: ['a.pdf'],
      instruction: 'risks',
    });
    expect(sanitizeTask({ mode: 'compare', filenames: ['a.pdf', 'b.pdf'], instruction: '  ' })).toEqual({
      mode: 'compare',
      filenames: ['a.pdf', 'b.pdf'],
    });
    for (const bad of [
      null,
      'summarize',
      { mode: 'outline', filenames: ['a.pdf'] },
      { mode: 'summarize', filenames: ['a.pdf', 'b.pdf'] },
      { mode: 'compare', filenames: ['a.pdf', 'a.pdf'] },
      { mode: 'compare', filenames: ['a.pdf'] },
      { mode: 'summarize', filenames: [7] },
      { mode: 'summarize', filenames: [''] },
    ]) {
      expect(sanitizeTask(bad)).toBeNull();
    }
  });

  it('round-trips a summary turn and its partial coverage, and drops bad values', () => {
    const asked = { ...turn('q'), task: { mode: 'summarize' as const, filenames: ['a.pdf'] } };
    expect(sanitizeTurn(asked)?.task).toEqual({ mode: 'summarize', filenames: ['a.pdf'] });
    const answered = { ...turn('a', 'assistant'), coverage: { read: 40, total: 300 } };
    expect(sanitizeTurn(answered)?.coverage).toEqual({ read: 40, total: 300 });
    expect(sanitizeTurn({ ...answered, coverage: { read: 300, total: 300 } })).not.toHaveProperty('coverage');
    expect(sanitizeTurn({ ...answered, coverage: { read: -1, total: 3 } })).not.toHaveProperty('coverage');
    expect(sanitizeTurn({ ...asked, task: { mode: 'x' } })).not.toHaveProperty('task');
  });

  it('keeps suggested follow-ups that are strings, within the server caps', () => {
    const answered = { ...turn('a', 'assistant'), followups: ['Next?', 3, 'x'.repeat(500), 'More?'] };
    expect(sanitizeTurn(answered)?.followups).toEqual(['Next?', 'More?']);
    expect(sanitizeTurn({ ...answered, followups: [1, 2] })).not.toHaveProperty('followups');
    // Each suggestion is a button's React key: a repeat broke rendering (and the e2e
    // console-error check); an empty one rendered as a blank button.
    expect(
      sanitizeTurn({ ...answered, followups: ['Next?', '', '   ', 'Next?', 'More?'] })?.followups,
    ).toEqual(['Next?', 'More?']);
  });
});

describe('what each store keeps of a long history', () => {
  const many = (count: number) =>
    Array.from({ length: count }, (_, index) =>
      conversation({
        id: `c${index}`,
        updatedAt: count - index, // c0 is the most recently used
        turns: [
          {
            id: `t${index}`,
            role: 'assistant',
            content: 'answer [1]',
            sources: [
              {
                source_number: 1,
                filename: 'a.md',
                page: 1,
                section: 'S',
                chunk_id: 'k',
                text: 'passage text',
              },
            ],
            timestamp: 1,
          } as ChatTurn,
        ],
      }),
    );

  it('localStorage holds the active, pinned and most recent conversations only', () => {
    const list = many(LOCAL_STORAGE_MAX_CONVERSATIONS + 30);
    list[40] = { ...list[40], pinned: true };
    const kept = conversationsForLocalStorage({
      version: 2,
      activeConversationId: 'c45',
      conversations: list,
    }).map((c) => c.id);

    expect(kept).toHaveLength(LOCAL_STORAGE_MAX_CONVERSATIONS);
    expect(kept).toContain('c45');
    expect(kept).toContain('c40');
    expect(kept).toContain('c0');
    expect(kept).not.toContain('c49');
  });

  it('a conversation renamed or pinned just now reaches the copy other tabs read', () => {
    // Other tabs learn of changes only through localStorage; a rename of an old
    // conversation left out of it was overwritten by the next tab's save.
    const list = many(LOCAL_STORAGE_MAX_CONVERSATIONS + 30);
    const oldest = list.length - 1;
    list[oldest] = { ...list[oldest], title: 'Renamed', editedAt: Date.now() + 1000 };
    const kept = conversationsForLocalStorage({
      version: 2,
      activeConversationId: 'c0',
      conversations: list,
    }).map((c) => c.id);

    expect(kept).toHaveLength(LOCAL_STORAGE_MAX_CONVERSATIONS);
    expect(kept).toContain(`c${oldest}`);
  });

  it('a conversation missing from the compact copy is not read as deleted', () => {
    const local = storage(many(3));
    const merged = mergeConversations(local, [local.conversations[0]], true);
    expect(merged.conversations.map((c) => c.id).sort()).toEqual(['c0', 'c1', 'c2']);
  });

  it('IndexedDB keeps every conversation, with passage text on the recent ones only', () => {
    const list = many(FULL_TEXT_CONVERSATIONS + 5);
    const stored = forChatStore({ version: 2, activeConversationId: 'c24', conversations: list });

    expect(stored.conversations).toHaveLength(list.length);
    const text = (id: string) =>
      stored.conversations.find((c) => c.id === id)?.turns[0].sources[0].text;
    expect(text('c0')).toBe('passage text');
    expect(text('c24')).toBe('passage text'); // the active one
    expect(text('c23')).toBe('');
    expect(stored.conversations.find((c) => c.id === 'c23')?.turns).toHaveLength(1);
  });
});

describe('pass 7 turn fields', () => {
  it('keeps valid source marks, scores and usage, and drops anything else', () => {
    const base = {
      id: 't',
      role: 'assistant',
      content: 'x',
      sources: [],
      timestamp: 1,
      timings: null,
      traceId: null,
    };
    const kept = sanitizeTurn({
      ...base,
      irrelevantSources: [2, 2, 3],
      topRerankScore: 0.4,
      usage: { prompt: 10, completion: 2 },
    });
    expect(kept?.irrelevantSources).toEqual([2, 3]);
    expect(kept?.topRerankScore).toBe(0.4);
    expect(kept?.usage).toEqual({ prompt: 10, completion: 2 });

    const dropped = sanitizeTurn({
      ...base,
      irrelevantSources: ['2', -1, 1.5, 5000],
      topRerankScore: 7,
      usage: { prompt: -1, completion: 'x' },
    });
    expect(dropped?.irrelevantSources).toBeUndefined();
    expect(dropped?.topRerankScore).toBeUndefined();
    expect(dropped?.usage).toBeUndefined();
  });
});
