import { describe, expect, it } from 'vitest';
import { narrowingBody } from '../../src/api/client';
import { searchNarrowing } from '../../src/utils/searchFilters';
import { snippetAround } from '../../src/utils/snippet';
import { dateGroup, listedConversations } from '../../src/utils/conversationList';
import { parseUploadTags } from '../../src/utils/tags';
import {
  MAX_CONVERSATIONS,
  capConversations,
  sanitizeNarrowing,
  sanitizeScope,
  sanitizeTurn,
  scopeNarrowing,
  type Conversation,
} from '../../src/hooks/chatPersistence';
import type { ConversationSummary } from '../../src/hooks/useChat';
import { fromDateInput, toDateInput } from '../../src/utils/dates';
import { extensionsLabel } from '../../src/utils/uploadValidation';

describe('scope narrowing', () => {
  it('orders a page range typed back to front and adds the upload date', () => {
    const scope = sanitizeScope({ filenames: ['a.pdf'], tags: [], pageMin: 3, uploadedAfter: 100 })!;
    expect(scopeNarrowing({ ...scope, pageMin: 9, pageMax: 2 })).toEqual({
      page_min: 2,
      page_max: 9,
      uploaded_after: 100,
    });
    // Across several documents only the date applies.
    expect(scopeNarrowing({ ...scope, filenames: ['a', 'b'] })).toEqual({ uploaded_after: 100 });
    expect(scopeNarrowing({ filenames: [], tags: [] })).toBeUndefined();
    expect(sanitizeScope({ filenames: [], tags: [], uploadedAfter: -1 })).toEqual({
      filenames: [],
      tags: [],
    });
  });

  it('keeps left-out passages and upload dates on a saved turn, in order only', () => {
    expect(
      sanitizeNarrowing({ exclude_chunk_ids: ['c1', 3, ''], uploaded_after: 10, uploaded_before: 20 }),
    ).toEqual({ exclude_chunk_ids: ['c1'], uploaded_after: 10, uploaded_before: 20 });
    expect(sanitizeNarrowing({ uploaded_after: 30, uploaded_before: 20 })).toBeNull();
    expect(narrowingBody({ exclude_chunk_ids: ['c1'], uploaded_after: 5 })).toEqual({
      exclude_chunk_ids: ['c1'],
      uploaded_after: 5,
    });
  });
});

describe('saved answers', () => {
  const turn = (id: string, saved?: boolean) => ({
    id,
    role: 'assistant' as const,
    content: 'A',
    sources: [],
    timestamp: 1,
    timings: null,
    traceId: null,
    ...(saved ? { saved } : {}),
  });

  it('keep the flag on an answer only', () => {
    expect(sanitizeTurn(turn('a', true))?.saved).toBe(true);
    expect(sanitizeTurn({ ...turn('b', true), role: 'user' })?.saved).toBeUndefined();
  });

  it('keep their conversation when the history is full', () => {
    const conversation = (index: number, saved = false): Conversation => ({
      id: `c${index}`,
      title: `c${index}`,
      createdAt: index,
      updatedAt: index,
      turns: [turn(`t${index}`, saved)],
    });
    const oldestSaved = conversation(0, true);
    const all = [
      oldestSaved,
      ...Array.from({ length: MAX_CONVERSATIONS + 5 }, (_, i) => conversation(i + 1)),
    ];
    const kept = capConversations(all);
    expect(kept).toHaveLength(MAX_CONVERSATIONS);
    expect(kept.some((c) => c.id === 'c0')).toBe(true);
  });
});

describe('the conversation list', () => {
  const summary = (id: string, extra: Partial<ConversationSummary> = {}): ConversationSummary => ({
    id,
    title: id,
    updatedAt: 0,
    turnCount: 1,
    ...extra,
  });

  it('lists pinned first, and filters by saved answers and workspace', () => {
    const conversations = [
      summary('a'),
      summary('b', { pinned: true }),
      summary('c', { savedCount: 2, workspace: 'legal' }),
    ];
    expect(listedConversations(conversations, 'all', '*').map((c) => c.id)).toEqual(['b', 'a', 'c']);
    expect(listedConversations(conversations, 'saved', '*').map((c) => c.id)).toEqual(['c']);
    expect(listedConversations(conversations, 'all', '').map((c) => c.id)).toEqual(['b', 'a']);
  });

  it('groups by day', () => {
    const now = new Date(2026, 9, 24, 15).getTime();
    expect(dateGroup(new Date(2026, 9, 24, 1).getTime(), now)).toBe('Today');
    expect(dateGroup(new Date(2026, 9, 23, 23).getTime(), now)).toBe('Yesterday');
    expect(dateGroup(new Date(2026, 9, 20).getTime(), now)).toBe('This week');
    expect(dateGroup(new Date(2026, 8, 1).getTime(), now)).toBe('Earlier');
  });
});

describe('small helpers', () => {
  it('round-trips a date input as a local day', () => {
    const seconds = fromDateInput('2026-05-31')!;
    expect(toDateInput(seconds)).toBe('2026-05-31');
    expect(fromDateInput('')).toBeUndefined();
    expect(fromDateInput('31/05/2026')).toBeUndefined();
    expect(toDateInput(undefined)).toBe('');
  });

  it('reads upload tags as the server stores them', () => {
    expect(parseUploadTags(' hr, Policy,  hr ,, a   b ')).toEqual(['hr', 'Policy', 'a b']);
    expect(parseUploadTags(Array.from({ length: 30 }, (_, i) => `t${i}`).join(','))).toHaveLength(20);
  });

  it('says code files as one word in the dropzone', () => {
    expect(extensionsLabel(['.pdf', '.py', '.ts', '.json', '.jpeg', '.jpg'])).toBe('pdf json jpg code');
  });

  it('cuts a long search result around its match', () => {
    const text = `${'filler '.repeat(60)}the parking passes are issued by reception${' tail'.repeat(60)}`;
    const snippet = snippetAround(text, ['parking passes are issued'], 120);
    expect(snippet).toContain('parking passes are issued');
    expect(snippet.startsWith('…')).toBe(true);
    expect(snippet.length).toBeLessThanOrEqual(122);
    expect(snippetAround('short', ['x'], 120)).toBe('short');
  });

  it('builds a search narrowing from the panel filters over the conversation scope', () => {
    const filters = {
      section: ' Leave ',
      exclude: 'a.pdf, ,b.pdf',
      pageMin: '9',
      pageMax: '2',
      uploadedAfter: '',
      count: '',
    };
    expect(searchNarrowing(filters, { uploaded_after: 50, page_min: 1 })).toEqual({
      page_min: 2,
      page_max: 9,
      section: 'Leave',
      exclude_filenames: ['a.pdf', 'b.pdf'],
      uploaded_after: 50,
    });
  });
});
