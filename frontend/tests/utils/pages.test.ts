import { pageLabel } from '../../src/utils/pages';
import { sanitizeTurn } from '../../src/hooks/chatPersistence';

describe('pageLabel', () => {
  it('shows one page, or the range a passage crosses', () => {
    expect(pageLabel(3)).toBe('p.3');
    expect(pageLabel(3, null)).toBe('p.3');
    expect(pageLabel(3, 3)).toBe('p.3');
    expect(pageLabel(3, 4)).toBe('p.3–4');
  });
});

describe('saved citations', () => {
  it('keep a page range and drop one that is not a page number', () => {
    const source = { source_number: 1, filename: 'a.pdf', page: 3, section: 'S', chunk_id: 'c', text: 't' };
    const turn = (page_end: unknown) =>
      sanitizeTurn({ id: 't', role: 'assistant', content: 'x', timestamp: 1, sources: [{ ...source, page_end }] });

    expect(turn(4)?.sources[0].page_end).toBe(4);
    expect(turn('4; drop table')?.sources[0]).not.toHaveProperty('page_end');
  });
});
