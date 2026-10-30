import { describe, expect, it } from 'vitest';
import type { QueryLogEntryResponse, CitationResponse } from '../../src/api/types';
import type { ChatTurn } from '../../src/components/ChatMessage';
import {
  LOCAL_STORAGE_MAX_CONVERSATIONS,
  compactForStorage,
  conversationsForLocalStorage,
  forChatStore,
  type Conversation,
} from '../../src/hooks/chatPersistence';
import { UNCOVERED, groupQuestions, mergeQueryLogs, stillUnanswered } from '../../src/utils/questions';
import { NO_FILTERS, searchNarrowing } from '../../src/utils/searchFilters';

function source(text: string): CitationResponse {
  return {
    source_number: 1,
    filename: 'a.pdf',
    page: 1,
    section: 'S',
    chunk_id: 'c1',
    excerpt: text.slice(0, 20),
    text,
    rerank_score: 0.5,
  } as CitationResponse;
}

function turn(id: string, role: ChatTurn['role'], extra: Partial<ChatTurn> = {}): ChatTurn {
  return {
    id,
    role,
    content: id,
    sources: role === 'assistant' ? [source(`passage of ${id}`)] : [],
    timestamp: 1,
    timings: null,
    traceId: null,
    ...extra,
  };
}

function conversation(id: string, updatedAt: number, turns: ChatTurn[]): Conversation {
  return { id, title: id, createdAt: updatedAt, updatedAt, turns };
}

describe('saved answers in the stored copies', () => {
  it('keeps a saved answer’s passages however old it is', () => {
    const turns = [
      turn('q1', 'user'),
      turn('a1', 'assistant', { saved: true }),
      turn('q2', 'user'),
      turn('a2', 'assistant'),
    ];
    const compact = compactForStorage(conversation('c', 1, turns), 100, 0);
    expect(compact.turns[1].sources[0].text).toBe('passage of a1');
    expect(compact.turns[3].sources[0].text).toBe('');
    const stored = forChatStore({
      version: 2,
      activeConversationId: null,
      conversations: Array.from({ length: 30 }, (_, index) =>
        conversation(`c${index}`, index, index === 0 ? turns : [turn(`q${index}`, 'user')]),
      ),
    });
    expect(stored.conversations[0].turns[1].sources[0].text).toBe('passage of a1');
  });

  it('keeps conversations holding a saved answer in localStorage’s copy', () => {
    const recent = Array.from({ length: LOCAL_STORAGE_MAX_CONVERSATIONS + 5 }, (_, index) =>
      conversation(`recent${index}`, 1000 + index, [turn(`q${index}`, 'user')]),
    );
    const old = conversation('old-saved', 1, [turn('q', 'user'), turn('a', 'assistant', { saved: true })]);
    const kept = conversationsForLocalStorage({
      version: 2,
      activeConversationId: null,
      conversations: [...recent, old],
    });
    expect(kept).toHaveLength(LOCAL_STORAGE_MAX_CONVERSATIONS);
    expect(kept.map((c) => c.id)).toContain('old-saved');
  });
});

function entry(question: string, createdAt: number, status = 'ok'): QueryLogEntryResponse {
  return { created_at: createdAt, question, status, top_score: null, mode: 'question', trace_id: null };
}

describe('asked questions', () => {
  it('lists as unanswered only what wasn’t answered the last time it was asked', () => {
    const rows = groupQuestions([
      entry('Leave policy?', 30),
      entry('Parking?', 20, UNCOVERED),
      entry('leave  policy?', 10, UNCOVERED),
    ]);
    const leave = rows.find((row) => row.question === 'Leave policy?')!;
    expect(leave.count).toBe(2);
    expect(leave.unanswered).toBe(1);
    expect(stillUnanswered(leave)).toBe(false);
    expect(rows.filter(stillUnanswered).map((row) => row.question)).toEqual(['Parking?']);
  });

  it('adds older unanswered asks after the latest window, each once', () => {
    const recent = [entry('b', 20, UNCOVERED), entry('a', 10)];
    const uncovered = [entry('b', 20, UNCOVERED), entry('old', 5, UNCOVERED)];
    expect(mergeQueryLogs(recent, uncovered).map((row) => row.question)).toEqual(['b', 'a', 'old']);
  });
});

describe('search filters', () => {
  it('takes the page range from the panel or the conversation, never a mix', () => {
    const scoped = { page_min: 1, page_max: 10 };
    expect(searchNarrowing({ ...NO_FILTERS, pageMin: '50' }, scoped)).toEqual({ page_min: 50 });
    expect(searchNarrowing(NO_FILTERS, scoped)).toEqual({ page_min: 1, page_max: 10 });
    expect(searchNarrowing({ ...NO_FILTERS, pageMin: '9', pageMax: '3' }, scoped)).toEqual({
      page_min: 3,
      page_max: 9,
    });
  });
});

describe('pass 13 helpers', () => {
  it('writes CSV cells quoted, and never as a spreadsheet formula', async () => {
    const { csvCell, toCsv } = await import('../../src/utils/csv');
    expect(csvCell('plain')).toBe('plain');
    expect(csvCell('a, "b"')).toBe('"a, ""b"""');
    expect(csvCell('=HYPERLINK("x")')).toBe(`"'=HYPERLINK(""x"")"`);
    expect(csvCell('-2+3')).toBe("'-2+3");
    expect(csvCell(null)).toBe('');
    expect(toCsv(['q', 'n'], [['line\nbreak', 2]])).toBe('q,n\r\n"line\nbreak",2\r\n');
  });

  it('says when a fetched document was checked and is next due', async () => {
    const { refreshSummary } = await import('../../src/utils/urlRefresh');
    const now = 1_000_000_000_000;
    const seconds = now / 1000;
    expect(
      refreshSummary(
        { last_checked: seconds - 7200, last_outcome: 'unchanged', next_due: seconds + 3 * 3600 },
        now,
      ),
    ).toBe('unchanged 2h ago · next in 3 h');
    expect(
      refreshSummary({ last_checked: seconds - 60, last_outcome: 'failed', failures: 3, next_due: seconds }, now),
    ).toBe('check failed 3× 1m ago · due now');
    expect(refreshSummary({ refresh_minutes: 0, next_due: null }, now)).toBe('not checked automatically');
  });

  it('sends the stricter answer switches only when on', async () => {
    const { questionRequestBody } = await import('../../src/api/client');
    const off = questionRequestBody('q', undefined, {
      rerankTopK: null,
      maxContextChunks: null,
      llmTemperature: null,
    });
    expect(off).toEqual({ question: 'q' });
    const on = questionRequestBody('q', undefined, {
      rerankTopK: null,
      maxContextChunks: null,
      llmTemperature: null,
      passagesOnly: true,
      strongMatchesOnly: true,
    });
    expect(on).toMatchObject({ passages_only: true, strong_matches_only: true });
  });

  it('puts the upload date range in order and keeps saved answers in Markdown', async () => {
    const { scopeNarrowing } = await import('../../src/hooks/chatPersistence');
    expect(
      scopeNarrowing({ filenames: [], tags: [], uploadedAfter: 500, uploadedBefore: 100 }),
    ).toEqual({ uploaded_after: 100, uploaded_before: 500 });
    const { searchNarrowing: narrowing, searchTags } = await import('../../src/utils/searchFilters');
    const filters = { ...NO_FILTERS, uploadedBefore: '2024-05-31', tag: 'hr' };
    expect(narrowing(filters, undefined).uploaded_before).toBeGreaterThan(0);
    expect(searchTags(filters, ['legal'])).toEqual(['hr']);
    expect(searchTags(NO_FILTERS, ['legal'])).toEqual(['legal']);

    const { savedAnswersToMarkdown } = await import('../../src/utils/exportChat');
    const markdown = savedAnswersToMarkdown([
      { conversationTitle: 'Leave', question: 'How long?', turn: turn('a', 'assistant', { content: 'Twenty days [1].' }) },
    ]);
    expect(markdown.startsWith('# Saved answers')).toBe(true);
    expect(markdown).toContain('## How long?');
    expect(markdown).toContain('Twenty days [1].');
    expect(markdown).toContain('[1] a.pdf');
  });
});

describe('saved answers past the stored turn limit', () => {
  it('keeps a saved answer and its question however far back', () => {
    const turns: ChatTurn[] = [
      turn('q0', 'user'),
      turn('a0', 'assistant', { saved: true }),
      ...Array.from({ length: 10 }, (_, index) =>
        turn(`t${index}`, index % 2 === 0 ? 'user' : 'assistant'),
      ),
    ];
    const compact = compactForStorage(conversation('c', 1, turns), 4);
    expect(compact.turns.map((item) => item.id)).toEqual(['q0', 'a0', 't6', 't7', 't8', 't9']);
    expect(compact.turns[1].sources[0].text).toBe('passage of a0');
  });
});
