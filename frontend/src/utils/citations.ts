// Same marker grammar the server reads citations with (api/citations.py): [3], [1, 2],
// [1-3], [1–3], [1, 3-4]. Not when it is a link's text ([1](url)), a reference
// definition ([1]: url) or an image (![1]).
const MARKER = /(?<![!\]\\])\[(\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*)\](?![(:[])/g;
const RANGE_SEP = /\s*[-–]\s*/;
// Fenced code blocks and inline code spans: markers there are code, not citations.
const CODE = /(```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`[^`\n]*`)/g;

export const CITATION_HREF_PREFIX = '#docrag-cite-';

/** The source numbers one marker's contents cite, ranges expanded and clamped to the
 *  sources on offer (a stray [1-100000] costs nothing; [3-1] and [0] are ignored). */
export function citedNumbers(marker: string, maxNumber: number): number[] {
  const numbers: number[] = [];
  for (const part of marker.split(',')) {
    const bounds = part.trim().split(RANGE_SEP).map(Number);
    // Sources are numbered from 1: an "[0]" became a link to no source, which the
    // renderer then treated as an ordinary link to the app itself.
    const start = Math.max(bounds[0], 1);
    const end = bounds[bounds.length - 1];
    for (let n = start; n <= Math.min(end, maxNumber); n += 1) {
      if (!numbers.includes(n)) numbers.push(n);
    }
  }
  return numbers;
}

/** Rewrite citation markers as links the answer renderer turns into buttons. A marker
 *  citing nothing on offer is left as written. */
export function linkCitations(markdown: string, sourceCount: number): string {
  if (sourceCount <= 0) return markdown;
  return markdown
    .split(CODE)
    .map((part, index) =>
      // split() with a capture group puts the code spans at odd indexes.
      index % 2 === 1
        ? part
        : part.replace(MARKER, (whole, inner: string) => {
            const numbers = citedNumbers(inner, sourceCount);
            if (numbers.length === 0) return whole;
            return numbers.map((n) => `[${n}](${CITATION_HREF_PREFIX}${n})`).join('');
          }),
    )
    .join('');
}

/** The source number a rewritten citation link points at, or null for any other link. */
export function citationNumberFromHref(href: string | undefined): number | null {
  if (!href?.startsWith(CITATION_HREF_PREFIX)) return null;
  const n = Number(href.slice(CITATION_HREF_PREFIX.length));
  return Number.isInteger(n) && n > 0 ? n : null;
}
