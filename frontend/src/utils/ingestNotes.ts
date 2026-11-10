import type { DocumentIngestResponse } from '../api/types';

/** What a finished ingest changed, in one line, or null when there is nothing to add to
 *  "indexed": how many passages kept their vectors and how many of the previous version
 *  went (a re-upload or refresh), and another document holding the same bytes. */
export function ingestNote(result: DocumentIngestResponse | null | undefined): string | null {
  if (!result || result.unchanged) return null;
  const parts: string[] = [];
  const reused = result.chunks_reused ?? 0;
  const removed = result.chunks_removed ?? 0;
  if (reused > 0 || removed > 0) {
    const fresh = Math.max(result.chunks_ingested - reused, 0);
    parts.push(
      `${fresh} passage${fresh === 1 ? '' : 's'} new or changed, ${reused} kept as ${
        reused === 1 ? 'it was' : 'they were'
      }, ${removed} removed`,
    );
  }
  if (result.duplicate_of) parts.push(`same bytes as ${result.duplicate_of}`);
  // What the server left out or couldn't read (a page limit, OCR failing on a page).
  for (const warning of result.warnings ?? []) parts.push(warning);
  return parts.length ? parts.join(' · ') : null;
}
