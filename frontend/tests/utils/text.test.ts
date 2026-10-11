import { describe, expect, it } from 'vitest';
import { sliceWhole } from '../../src/utils/text';

describe('sliceWhole', () => {
  it('never ends inside a surrogate pair', () => {
    // '😀' is two code units; a cut after the first left a lone high surrogate.
    expect(sliceWhole('ab😀c', 3)).toBe('ab');
    expect(sliceWhole('ab😀c', 4)).toBe('ab😀');
  });

  it('leaves short text and plain cuts alone', () => {
    expect(sliceWhole('abc', 10)).toBe('abc');
    expect(sliceWhole('abcdef', 3)).toBe('abc');
  });
});
