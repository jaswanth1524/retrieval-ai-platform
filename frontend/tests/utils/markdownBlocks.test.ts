import { describe, expect, it } from 'vitest';
import { splitStreamingMarkdown } from '../../src/utils/markdownBlocks';

describe('splitStreamingMarkdown', () => {
  it('splits finished paragraphs from the one still streaming', () => {
    expect(splitStreamingMarkdown('First para.\n\nSecond para.\n\nThird is grow')).toEqual({
      done: ['First para.\n', 'Second para.\n'],
      tail: 'Third is grow',
    });
  });

  it('never splits inside a fenced code block', () => {
    const text = 'Intro.\n\n```\nline one\n\nline two\n```\n\nAfter';
    expect(splitStreamingMarkdown(text)).toEqual({
      done: ['Intro.\n', '```\nline one\n\nline two\n```\n'],
      tail: 'After',
    });
  });

  it('keeps everything in the tail until a block is finished', () => {
    expect(splitStreamingMarkdown('Only one paragraph so far')).toEqual({
      done: [],
      tail: 'Only one paragraph so far',
    });
    expect(splitStreamingMarkdown('')).toEqual({ done: [], tail: '' });
  });

  it('rejoins to the same text minus the separating blank lines', () => {
    const text = '# Title\n\n- a\n- b\n\nEnd [1].';
    const { done, tail } = splitStreamingMarkdown(text);
    expect([...done, tail].join('\n')).toBe(text);
  });
});
