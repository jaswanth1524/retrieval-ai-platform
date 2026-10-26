import type { ReactNode } from 'react';

/** ``text`` with each of ``quotes`` found in it marked (found by search, not offsets:
 *  the server counts code points, JavaScript UTF-16 units). Overlapping quotes after the
 *  first are left unmarked. */
export default function MarkedText({
  text,
  quotes,
  className,
  testId,
}: {
  text: string;
  quotes: string[];
  className: string;
  testId?: string;
}) {
  const ranges = quotes
    .map((quote) => ({ start: quote ? text.indexOf(quote) : -1, length: quote.length }))
    .filter((range) => range.start >= 0)
    .sort((a, b) => a.start - b.start);
  const parts: ReactNode[] = [];
  let position = 0;
  for (const { start, length } of ranges) {
    if (start < position) continue;
    parts.push(text.slice(position, start));
    parts.push(
      <mark key={start} className={className} data-testid={testId}>
        {text.slice(start, start + length)}
      </mark>,
    );
    position = start + length;
  }
  parts.push(text.slice(position));
  return <>{parts}</>;
}
