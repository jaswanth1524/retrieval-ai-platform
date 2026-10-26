/** At most ``max`` characters of ``text`` around the first of ``quotes`` found in it
 *  (its start when none is), with ellipses where it was cut. */
export function snippetAround(text: string, quotes: string[], max: number): string {
  if (text.length <= max) return text;
  const first = quotes.map((quote) => (quote ? text.indexOf(quote) : -1)).find((at) => at >= 0);
  if (first === undefined) return `${text.slice(0, max).trimEnd()}…`;
  // A little of what comes before, so the mark reads in context.
  const start = Math.max(0, Math.min(first - 40, text.length - max));
  const end = Math.min(text.length, start + max);
  return `${start > 0 ? '…' : ''}${text.slice(start, end).trim()}${end < text.length ? '…' : ''}`;
}
