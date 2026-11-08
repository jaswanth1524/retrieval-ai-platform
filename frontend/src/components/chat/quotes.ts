import type { CitationResponse } from '../../api/types';

/** The quotes a citation's spans mark in its passage (for the viewer's highlights). */
export function quotesOf(source: CitationResponse): string[] | undefined {
  return source.spans?.length ? source.spans.map((span) => span.quote) : undefined;
}
