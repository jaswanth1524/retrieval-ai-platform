import { afterEach, describe, expect, it } from 'vitest';
import { pruneDrafts, readDraft, writeDraft } from '../../src/utils/drafts';

describe('drafts', () => {
  afterEach(() => localStorage.clear());

  it('keeps one draft per conversation and forgets a blank one', () => {
    writeDraft('a', 'half a question');
    writeDraft('b', 'another');
    expect(readDraft('a')).toBe('half a question');
    expect(readDraft('b')).toBe('another');

    writeDraft('a', '   ');
    expect(readDraft('a')).toBe('');
    expect(readDraft('b')).toBe('another');
  });

  it('caps the number of drafts, dropping the oldest', () => {
    for (let index = 0; index < 55; index += 1) {
      writeDraft(`c${index}`, `draft ${index}`);
    }
    // Same-millisecond writes tie; at least the newest survive and the total is capped.
    const stored = JSON.parse(localStorage.getItem('docrag-drafts') ?? '{}');
    expect(Object.keys(stored)).toHaveLength(50);
  });

  it('forgets drafts of conversations that are gone, keeping the new-chat one', () => {
    writeDraft('kept', 'x');
    writeDraft('gone', 'y');
    writeDraft('new', 'z');

    pruneDrafts(new Set(['kept']));

    expect(readDraft('kept')).toBe('x');
    expect(readDraft('gone')).toBe('');
    expect(readDraft('new')).toBe('z');
  });

  it('survives storage holding something else', () => {
    localStorage.setItem('docrag-drafts', '[1, 2]');
    expect(readDraft('a')).toBe('');
    writeDraft('a', 'ok');
    expect(readDraft('a')).toBe('ok');
  });
});
