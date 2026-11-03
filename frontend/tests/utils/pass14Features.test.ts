import { afterEach, describe, expect, it } from 'vitest';
import type { ChatTurn } from '../../src/components/ChatMessage';
import type { CitationResponse } from '../../src/api/types';
import { sourceChange } from '../../src/utils/citationFreshness';
import { citedSourcesCsv } from '../../src/utils/exportChat';
import { ingestNote } from '../../src/utils/ingestNotes';
import {
  MAX_SCOPE_PRESETS,
  SCOPE_PRESETS_STORAGE_KEY,
  prunePresets,
  readScopePresets,
  withPreset,
  writeScopePresets,
} from '../../src/utils/scopePresets';

afterEach(() => localStorage.clear());

const source = (overrides: Partial<CitationResponse> = {}): CitationResponse => ({
  source_number: 1,
  filename: 'guide.md',
  page: 2,
  section: 'Intro',
  chunk_id: 'c1',
  text: 'Alpha.',
  ...overrides,
});

describe('citation freshness', () => {
  const answeredAt = Date.UTC(2026, 0, 2);
  it('flags a deleted or re-uploaded document, never before the listing loaded', () => {
    const documents = { 'guide.md': { uploadedAt: answeredAt / 1000 - 60 } };
    expect(sourceChange('guide.md', answeredAt, documents, true)).toBeNull();
    expect(sourceChange('gone.md', answeredAt, documents, true)).toBe('deleted');
    expect(sourceChange('gone.md', answeredAt, documents, false)).toBeNull();
    // Seconds in the listing, milliseconds on the turn.
    expect(
      sourceChange('guide.md', answeredAt, { 'guide.md': { uploadedAt: answeredAt / 1000 + 5 } }, true),
    ).toBe('changed');
    expect(sourceChange('guide.md', answeredAt, { 'guide.md': {} }, true)).toBeNull();
  });
});

describe('cited sources CSV', () => {
  it('lists each source under its answer and question, quotes included, formulas defused', () => {
    const turns: ChatTurn[] = [
      { id: 'q', role: 'user', content: '=SUM(A1)', sources: [], timestamp: 1, timings: null, traceId: null },
      {
        id: 'a',
        role: 'assistant',
        content: 'Alpha [1].',
        sources: [source({ spans: [{ start: 0, end: 6, quote: 'Alpha.' }] }), source({ source_number: 2, page_end: 3 })],
        timestamp: 2,
        timings: null,
        traceId: null,
      },
    ];
    const lines = citedSourcesCsv(turns, 'Guide talk').trimEnd().split('\r\n');
    expect(lines[0]).toBe('conversation,question,answer,source,filename,page,section,chunk_id,quote');
    expect(lines[1]).toBe("Guide talk,'=SUM(A1),1,1,guide.md,p.2,Intro,c1,Alpha.");
    expect(lines).toHaveLength(3);
    expect(lines[2]).toContain('guide.md,p.2–3,Intro,c1,');
  });
});

describe('ingest notes', () => {
  it('says what a re-upload changed and names a duplicate', () => {
    const base = { filename: 'a.txt', sections_parsed: 1, chunks_ingested: 10, collection_name: 'c' };
    expect(ingestNote(base)).toBeNull();
    expect(ingestNote({ ...base, unchanged: true, duplicate_of: 'b.txt' })).toBeNull();
    expect(ingestNote({ ...base, chunks_reused: 8, chunks_removed: 1 })).toBe(
      '2 passages new or changed, 8 kept as they were, 1 removed',
    );
    expect(ingestNote({ ...base, duplicate_of: 'b.txt' })).toBe('same bytes as b.txt');
  });
});

describe('scope presets', () => {
  it('are saved per workspace, replaced by name, capped and sanitized', () => {
    let presets = withPreset([], 'Policies', { filenames: ['a.md'], tags: [] });
    presets = withPreset(presets, 'HR', { filenames: [], tags: ['hr'] });
    presets = withPreset(presets, 'Policies', { filenames: ['b.md'], tags: [] });
    expect(presets.map((preset) => preset.name)).toEqual(['Policies', 'HR']);
    expect(presets[0].scope.filenames).toEqual(['b.md']);
    // A scope that narrows nothing isn't saved.
    expect(withPreset(presets, 'Everything', { filenames: [], tags: [] })).toBe(presets);

    writeScopePresets('legal', presets);
    expect(readScopePresets('legal')).toEqual(presets);
    expect(readScopePresets('')).toEqual([]);

    let many = presets;
    for (let n = 0; n < MAX_SCOPE_PRESETS + 5; n += 1) {
      many = withPreset(many, `p${n}`, { filenames: [`${n}.md`], tags: [] });
    }
    expect(many).toHaveLength(MAX_SCOPE_PRESETS);

    localStorage.setItem(
      SCOPE_PRESETS_STORAGE_KEY,
      JSON.stringify({ '': [{ name: 'bad', scope: 'x' }, { name: ' ok ', scope: { filenames: ['a.md'], tags: [] } }] }),
    );
    expect(readScopePresets('')).toEqual([{ name: 'ok', scope: { filenames: ['a.md'], tags: [] } }]);
  });

  it('lose deleted files, and a preset left narrowing nothing goes', () => {
    const presets = [
      { name: 'One', scope: { filenames: ['gone.md'], tags: [], pageMin: 2 } },
      { name: 'Two', scope: { filenames: ['a.md', 'gone.md'], tags: [] } },
      { name: 'Tags', scope: { filenames: ['gone.md'], tags: ['hr'] } },
    ];
    const pruned = prunePresets(presets, new Set(['a.md']));
    expect(pruned).toEqual([
      { name: 'Two', scope: { filenames: ['a.md'], tags: [] } },
      { name: 'Tags', scope: { filenames: [], tags: ['hr'] } },
    ]);
    expect(prunePresets(pruned, new Set(['a.md']))).toBe(pruned);
  });
});
