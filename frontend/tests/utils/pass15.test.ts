import { describe, expect, it } from 'vitest';
import type { CitationResponse } from '../../src/api/types';
import type { ChatTurn } from '../../src/components/ChatMessage';
import { cleanLabels, emptyStorage, withConversationRestored, withoutConversation } from '../../src/hooks/chatPersistence';
import { sanitizeConversation, sanitizeTask, sanitizeTurn } from '../../src/hooks/chatPersistence';
import { MAX_ANSWER_VERSIONS, mergeRegenerated } from '../../src/utils/answerVersions';
import { listedConversations } from '../../src/utils/conversationList';
import { deepLinkHash, deepLinkUrl, parseDeepLink, usageTotal } from '../../src/utils/deepLink';
import { matchRanges } from '../../src/utils/find';
import { reviewReasons } from '../../src/utils/review';
import { isArchive, uploadExtensions } from '../../src/utils/uploadValidation';

const source = (n: number, filename = 'a.md'): CitationResponse => ({
  source_number: n,
  filename,
  page: 1,
  section: 'S',
  chunk_id: `c${n}`,
  text: 'the passage',
});

const turn = (id: string, role: ChatTurn['role'], content: string, extra: Partial<ChatTurn> = {}): ChatTurn => ({
  id,
  role,
  content,
  sources: [],
  timestamp: 1_700_000_000_000,
  timings: null,
  traceId: null,
  ...extra,
});

describe('pass 15 utils', () => {
  it('a regenerated answer keeps the one it replaced as an earlier version', () => {
    const turns = [
      turn('q1', 'user', 'What?'),
      turn('a1', 'assistant', 'First.', { sources: [source(1)], model: 'm1' }),
      turn('q2', 'user', 'What?'),
      turn('a2', 'assistant', 'Second.'),
    ];

    const merged = mergeRegenerated(turns, 'a1', 'a2');

    expect(merged.map((t) => t.id)).toEqual(['q2', 'a2']);
    expect(merged[1].versions).toEqual([
      { content: 'First.', sources: [{ ...source(1), text: '' }], timestamp: turns[1].timestamp, model: 'm1' },
    ]);
    // Something in between, or another question: left as it is.
    expect(mergeRegenerated([...turns.slice(0, 2), turn('x', 'user', 'Other?'), turns[3]], 'a1', 'a2')).toHaveLength(4);
    const many = Array.from({ length: 8 }, (_, n) => ({ content: `v${n}`, sources: [], timestamp: n }));
    const capped = mergeRegenerated([turns[0], { ...turns[1], versions: many }, turns[2], turns[3]], 'a1', 'a2');
    expect(capped[1].versions).toHaveLength(MAX_ANSWER_VERSIONS);
    expect(capped[1].versions?.at(-1)?.content).toBe('First.');
  });

  it('find matches every occurrence regardless of case', () => {
    expect(matchRanges('Leave and LEAVE and leaves', 'leave')).toEqual([
      [0, 5],
      [10, 15],
      [20, 25],
    ]);
    expect(matchRanges('anything', '  ')).toEqual([]);
  });

  it('deep links round-trip and refuse odd ids', () => {
    const hash = deepLinkHash({ conversationId: 'abc-1', turnId: 't_2' });
    expect(parseDeepLink(hash)).toEqual({ conversationId: 'abc-1', turnId: 't_2' });
    expect(parseDeepLink('#c=abc')).toEqual({ conversationId: 'abc' });
    expect(parseDeepLink('#c=<script>')).toBeNull();
    expect(parseDeepLink('')).toBeNull();
    expect(deepLinkUrl({ conversationId: 'x' }, { origin: 'http://h', pathname: '/' })).toBe('http://h/#c=x');
  });

  it('usage adds up tokens and the cost of priced answers only', () => {
    expect(usageTotal([turn('a', 'assistant', 'x')])).toBeNull();
    expect(
      usageTotal([
        turn('a', 'assistant', 'x', { usage: { prompt: 10, completion: 5, costUsd: 0.01 } }),
        turn('b', 'assistant', 'y', { usage: { prompt: 1, completion: 1 } }),
      ]),
    ).toEqual({ tokens: 17, costUsd: 0.01 });
    expect(usageTotal([turn('b', 'assistant', 'y', { usage: { prompt: 1, completion: 1 } })])).toEqual({ tokens: 2 });
  });

  it('review reasons name what to check', () => {
    const answer = turn('a', 'assistant', 'Yes [1].', {
      weakMatch: true,
      truncated: true,
      sources: [source(1, 'gone.md'), source(2, 'new.md')],
    });
    expect(
      reviewReasons(answer, (filename) => (filename === 'gone.md' ? 'deleted' : 'changed')),
    ).toEqual(['weak match', 'cut off', 'a source changed since', 'a source was deleted']);
    expect(reviewReasons(turn('a', 'assistant', 'Fine [1].'))).toEqual([]);
    expect(reviewReasons(turn('q', 'user', 'not found'))).toEqual([]);
  });

  it('labels are cleaned, capped and filter the list', () => {
    expect(cleanLabels([' HR ', 'hr', 'audit  2026', 7, '', 'a'.repeat(40), 'x', 'y', 'z'])).toEqual([
      'HR',
      'audit 2026',
      'a'.repeat(24),
      'x',
      'y',
    ]);
    const list = [
      { id: '1', title: 'a', updatedAt: 1, turnCount: 1, labels: ['HR'] },
      { id: '2', title: 'b', updatedAt: 2, turnCount: 1 },
    ];
    expect(listedConversations(list, 'all', '*', 'hr').map((c) => c.id)).toEqual(['1']);
    expect(listedConversations(list, 'all', '*').map((c) => c.id)).toEqual(['1', '2']);
  });

  it('stored conversations keep their labels, versions and a summary page range', () => {
    const raw = {
      id: 'c1',
      title: 'T',
      createdAt: 1,
      updatedAt: 2,
      labels: ['hr', 'HR', 3],
      turns: [
        turn('a', 'assistant', 'Now.', {
          versions: [{ content: 'Before.', sources: [source(1)], timestamp: 5 }, { bogus: true }] as never,
        }),
      ],
    };
    const conversation = sanitizeConversation(raw);
    expect(conversation?.labels).toEqual(['hr']);
    expect(conversation?.turns[0].versions).toHaveLength(1);
    expect(sanitizeTurn(turn('q', 'user', 'x', { versions: [] }))?.versions).toBeUndefined();
    expect(sanitizeTask({ mode: 'summarize', filenames: ['a.pdf'], pageMin: 2, pageMax: 0 })).toEqual({
      mode: 'summarize',
      filenames: ['a.pdf'],
      pageMin: 2,
    });
    expect(sanitizeTask({ mode: 'compare', filenames: ['a', 'b'], pageMin: 2 })).toEqual({
      mode: 'compare',
      filenames: ['a', 'b'],
    });
  });

  it('a deleted conversation comes back with its tombstone gone', () => {
    const conversation = {
      id: 'c1',
      title: 'T',
      createdAt: 1,
      updatedAt: 2,
      turns: [turn('q', 'user', 'x')],
    };
    const deleted = withoutConversation({ ...emptyStorage(), conversations: [conversation] }, 'c1', 10);
    expect(deleted.deleted?.map((entry) => entry.id)).toEqual(['c1']);
    const back = withConversationRestored(deleted, conversation);
    expect(back.deleted).toEqual([]);
    expect(back.conversations.map((c) => c.id)).toContain('c1');
    expect(back.activeConversationId).toBe('c1');
  });

  it('uploads take a zip of documents', () => {
    expect(uploadExtensions(['.pdf'])).toEqual(['.pdf', '.zip']);
    expect(isArchive(new File(['x'], 'Docs.ZIP'))).toBe(true);
    expect(isArchive(new File(['x'], 'a.pdf'))).toBe(false);
  });
});
