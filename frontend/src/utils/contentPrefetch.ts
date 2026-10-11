import { api } from '../api/client';
import type { DocumentContentResponse } from '../api/types';

// Chunks either side of a cited chunk the viewer opens on (DocumentViewer).
export const CITATION_WINDOW_RADIUS = 30;
// Hovered citations kept; a hover that is never clicked costs one small request.
const MAX_ENTRIES = 8;
// A document re-indexed meanwhile has new chunks; an old prefetch must not stand in.
const MAX_AGE_MS = 30_000;

interface Entry {
  at: number;
  content: Promise<DocumentContentResponse>;
}

const entries = new Map<string, Entry>();

const keyFor = (filename: string, chunkId: string) => `${filename}\n${chunkId}`;

/** Start loading the viewer's first page for a citation the pointer or focus is on, so
 *  a click opens it straight away instead of after a round trip. */
export function prefetchCitationContent(filename: string, chunkId: string): void {
  const key = keyFor(filename, chunkId);
  const existing = entries.get(key);
  if (existing && Date.now() - existing.at < MAX_AGE_MS) return;
  const content = api.getDocumentContent(filename, undefined, {
    around: chunkId,
    radius: CITATION_WINDOW_RADIUS,
  });
  // A failed prefetch is forgotten; the viewer then asks again and shows the error.
  content.catch(() => {
    if (entries.get(key)?.content === content) entries.delete(key);
  });
  entries.delete(key);
  entries.set(key, { at: Date.now(), content });
  while (entries.size > MAX_ENTRIES) {
    const oldest = entries.keys().next().value;
    if (oldest === undefined) break;
    entries.delete(oldest);
  }
}

/** The prefetched first page for this citation, if one is fresh; it is used once. */
export function takePrefetchedContent(
  filename: string,
  chunkId: string,
): Promise<DocumentContentResponse> | undefined {
  const key = keyFor(filename, chunkId);
  const entry = entries.get(key);
  entries.delete(key);
  if (!entry || Date.now() - entry.at >= MAX_AGE_MS) return undefined;
  return entry.content;
}

/** For tests. */
export function clearPrefetchedContent(): void {
  entries.clear();
}
