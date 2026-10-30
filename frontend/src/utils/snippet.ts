/** A cut-down passage and the parts of the quotes it still shows, to mark in it. */
export interface Snippet {
  text: string;
  quotes: string[];
}

// Never cut between the two halves of a surrogate pair (an emoji, some CJK).
function safeStart(text: string, at: number): number {
  const code = text.charCodeAt(at);
  return at > 0 && code >= 0xdc00 && code <= 0xdfff ? at - 1 : at;
}

function safeEnd(text: string, at: number): number {
  const code = text.charCodeAt(at - 1);
  return at < text.length && code >= 0xd800 && code <= 0xdbff ? at + 1 : at;
}

/** At most about ``max`` characters of ``text`` around the first of ``quotes`` found in
 *  it (its start when none is), with ellipses where it was cut, and each quote clipped
 *  to what the snippet shows. A matched sentence longer than the snippet starts it, so
 *  part of it is still marked rather than none. */
export function snippetAround(text: string, quotes: string[], max: number): Snippet {
  if (text.length <= max) return { text, quotes };
  const found = quotes
    .map((quote) => ({ quote, at: quote ? text.indexOf(quote) : -1 }))
    .filter((item) => item.at >= 0);
  const first = found.length > 0 ? Math.min(...found.map((item) => item.at)) : undefined;
  if (first === undefined) {
    return { text: `${text.slice(0, safeEnd(text, max)).trimEnd()}…`, quotes: [] };
  }
  // A little of what comes before, so the mark reads in context — unless the quote is
  // too long to show whole, when it starts the snippet.
  const firstLength = found.find((item) => item.at === first)?.quote.length ?? 0;
  const lead = firstLength + 40 > max ? 0 : 40;
  const start = safeStart(text, Math.max(0, Math.min(first - lead, text.length - max)));
  const end = safeEnd(text, Math.min(text.length, start + max));
  const visible = text.slice(start, end);
  const clipped = found
    .map(({ quote, at }) => {
      const from = Math.max(at, start);
      const to = Math.min(at + quote.length, end);
      return to > from ? text.slice(from, to).trim() : '';
    })
    .filter(Boolean);
  const body = visible.trim();
  return {
    text: `${start > 0 ? '…' : ''}${body}${end < text.length ? '…' : ''}`,
    quotes: clipped,
  };
}
