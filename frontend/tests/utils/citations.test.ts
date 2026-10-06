import { describe, expect, it } from 'vitest';
import { citationNumberFromHref, citedNumbers, linkCitations } from '../../src/utils/citations';

describe('linkCitations', () => {
  it('turns single, list and range markers into one link per cited source', () => {
    expect(linkCitations('A [1]. B [1, 2]. C [2-3].', 3)).toBe(
      'A [1](#docrag-cite-1). B [1](#docrag-cite-1)[2](#docrag-cite-2). ' +
        'C [2](#docrag-cite-2)[3](#docrag-cite-3).',
    );
  });

  it('leaves code, links, references, images and uncited numbers alone', () => {
    const text = 'See `arr[1]` and\n```\nx[2]\n```\n[1](http://a) ![1](i.png) [9]\n\n[2]: http://b';
    expect(linkCitations(text, 3)).toBe(text);
  });

  it('does nothing when there are no sources', () => {
    expect(linkCitations('A [1].', 0)).toBe('A [1].');
  });
});

describe('citedNumbers', () => {
  it('expands, clamps and ignores backwards ranges', () => {
    expect(citedNumbers('1, 3-4', 5)).toEqual([1, 3, 4]);
    expect(citedNumbers('1-100000', 3)).toEqual([1, 2, 3]);
    expect(citedNumbers('3-1', 5)).toEqual([]);
    // Sources start at 1: [0] links nothing (it became a link to the app itself).
    expect(citedNumbers('0', 5)).toEqual([]);
    expect(citedNumbers('0-2', 5)).toEqual([1, 2]);
  });
});

describe('citationNumberFromHref', () => {
  it('reads only rewritten citation links', () => {
    expect(citationNumberFromHref('#docrag-cite-2')).toBe(2);
    expect(citationNumberFromHref('https://example.com')).toBeNull();
    expect(citationNumberFromHref(undefined)).toBeNull();
  });
});
