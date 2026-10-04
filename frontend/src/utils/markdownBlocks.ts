/** Split streamed markdown into the blocks finished so far, plus the one still growing.
 *
 * A block ends at a blank line outside a fenced code block. Finished blocks don't change
 * as more tokens arrive, so each can be parsed once instead of the whole answer being
 * re-parsed on every token. Only for rendering *while streaming*: a finished answer is
 * rendered whole, since splitting can change how loose lists and reference-style links
 * render. */
export function splitStreamingMarkdown(text: string): { done: string[]; tail: string } {
  const done: string[] = [];
  let blockStart = 0;
  let inFence = false;
  let lineStart = 0;
  while (lineStart <= text.length) {
    const newline = text.indexOf('\n', lineStart);
    const lineEnd = newline === -1 ? text.length : newline;
    const line = text.slice(lineStart, lineEnd);
    if (/^\s{0,3}(```|~~~)/.test(line)) inFence = !inFence;
    // A blank line that isn't the last line (the last may still be followed by more text
    // of the same block) closes the block before it.
    if (!inFence && newline !== -1 && line.trim() === '' && lineStart > blockStart) {
      const block = text.slice(blockStart, lineStart);
      if (block.trim()) done.push(block);
      blockStart = newline + 1;
    }
    if (newline === -1) break;
    lineStart = newline + 1;
  }
  return { done, tail: text.slice(blockStart) };
}
