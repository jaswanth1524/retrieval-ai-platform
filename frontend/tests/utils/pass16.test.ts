import { describe, expect, it } from 'vitest';
import type { ChatTurn } from '../../src/components/ChatMessage';
import { sanitizeTurn } from '../../src/hooks/chatPersistence';
import { findAnswer, mergeRegenerated } from '../../src/utils/answerVersions';

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

describe('pass 16: answer versions', () => {
  it('a regenerated saved answer stays saved and remembers what it replaced', () => {
    const turns = [
      turn('q1', 'user', 'What?'),
      turn('a1', 'assistant', 'First.', { saved: true, traceId: 't1', style: { answerFormat: 'table' } }),
      turn('q2', 'user', 'What?'),
      turn('a2', 'assistant', 'Second.', { traceId: 't2' }),
    ];
    const [, merged] = mergeRegenerated(turns, 'a1', 'a2');
    expect(merged.saved).toBe(true);
    expect(merged.previousIds).toEqual(['a1']);
    expect(merged.versions?.[0]).toMatchObject({ traceId: 't1', style: { answerFormat: 'table' } });
    expect(findAnswer([turns[2], merged], 'a1')?.id).toBe('a2');
  });

  it('regenerating an earlier answer replaces it where it is', () => {
    const turns = [
      turn('q1', 'user', 'First question?'),
      turn('a1', 'assistant', 'Old one.'),
      turn('q2', 'user', 'Second question?'),
      turn('a2', 'assistant', 'Two.'),
      turn('q3', 'user', 'First question?'),
      turn('a3', 'assistant', 'New one.'),
    ];
    const merged = mergeRegenerated(turns, 'a1', 'a3');
    expect(merged.map((t) => t.id)).toEqual(['q3', 'a3', 'q2', 'a2']);
    expect(merged[1].versions?.map((v) => v.content)).toEqual(['Old one.']);
  });

  it('a saved style, version trace and previous ids survive storage', () => {
    const saved = sanitizeTurn(
      turn('a', 'assistant', 'x', {
        style: { answerFormat: 'bullets', answerLanguage: 'German' },
        previousIds: ['old', 7 as unknown as string],
        versions: [{ content: 'v', sources: [], timestamp: 1, traceId: 't', style: { answerFormat: 'nope' as never } }],
      }),
    );
    expect(saved?.style).toEqual({ answerFormat: 'bullets', answerLanguage: 'German' });
    expect(saved?.previousIds).toEqual(['old']);
    expect(saved?.versions?.[0]).toEqual({ content: 'v', sources: [], timestamp: 1, traceId: 't' });
  });
});

describe('pass 16: labels', () => {
  it('labels set in another tab during an answer are kept', async () => {
    const { mergeConversations } = await import('../../src/hooks/chatPersistence');
    const mine = {
      id: 'c',
      title: 'Plans',
      createdAt: 1,
      updatedAt: 5,
      editedAt: 5,
      turns: [turn('q', 'user', 'Hi?'), turn('a', 'assistant', 'streaming…')],
    };
    const theirs = { ...mine, editedAt: 9, updatedAt: 9, labels: ['hr'], turns: [mine.turns[0]] };
    const merged = mergeConversations(
      { version: 2, activeConversationId: 'c', conversations: [mine] },
      [theirs],
      false,
      undefined,
      10,
      'c',
    );
    const [kept] = merged.conversations;
    expect(kept.labels).toEqual(['hr']);
    // The streaming answer is not replaced by the other tab's copy.
    expect(kept.turns).toHaveLength(2);
  });
});

describe('pass 16: ingest warnings', () => {
  it('what indexing left out is said when the document is indexed', async () => {
    const { ingestNote } = await import('../../src/utils/ingestNotes');
    const base = {
      filename: 'scan.pdf',
      sections_parsed: 3,
      chunks_ingested: 4,
      collection_name: 'docs',
    };
    expect(ingestNote({ ...base, warnings: ['OCR failed on 2 scanned page(s): not read.'] })).toBe(
      'OCR failed on 2 scanned page(s): not read.',
    );
    expect(ingestNote({ ...base, warnings: [] })).toBeNull();
  });
});

describe('pass 16: versions compared and restored', () => {
  it('a word diff marks what was removed and added', async () => {
    const { wordDiff } = await import('../../src/utils/textDiff');
    expect(wordDiff('Leave is ten days.', 'Leave is twelve days.')).toEqual([
      { kind: 'same', text: 'Leave is ' },
      { kind: 'removed', text: 'ten ' },
      { kind: 'added', text: 'twelve ' },
      { kind: 'same', text: 'days.' },
    ]);
    expect(wordDiff('a '.repeat(2000), 'b')).toBeNull();
  });

  it('an earlier version becomes the answer, and the answer a version', async () => {
    const { withVersionRestored } = await import('../../src/utils/answerVersions');
    const answer = turn('a', 'assistant', 'New.', {
      traceId: 't2',
      weakMatch: true,
      feedback: 'up',
      saved: true,
      versions: [{ content: 'Old.', sources: [], timestamp: 5, traceId: 't1', style: { answerFormat: 'table' } }],
    });
    const restored = withVersionRestored(answer, 0);
    expect(restored).toMatchObject({ content: 'Old.', traceId: 't1', style: { answerFormat: 'table' }, saved: true });
    expect(restored.weakMatch).toBeUndefined();
    expect(restored.feedback).toBeUndefined();
    expect(restored.versions?.map((v) => v.content)).toEqual(['New.']);
    expect(withVersionRestored(answer, 3)).toBe(answer);
  });
});

describe('pass 16: passage links', () => {
  it('round-trip, and refuse names and ids that could be anything', async () => {
    const { parsePassageLink, passageLinkUrl } = await import('../../src/utils/deepLink');
    const url = passageLinkUrl(
      { filename: 'leave policy.pdf', chunkId: 'abc123', workspace: 'legal' },
      { origin: 'https://docs.example', pathname: '/' },
    );
    expect(url).toBe('https://docs.example/#d=leave+policy.pdf&k=abc123&w=legal');
    expect(parsePassageLink(new URL(url).hash)).toEqual({
      filename: 'leave policy.pdf',
      chunkId: 'abc123',
      workspace: 'legal',
    });
    expect(parsePassageLink('#d=a.txt&k=<x>&w=Bad Name')).toEqual({ filename: 'a.txt' });
    expect(parsePassageLink('#d=../etc/passwd')).toBeNull();
    expect(parsePassageLink('#c=conv1')).toBeNull();
  });
});
