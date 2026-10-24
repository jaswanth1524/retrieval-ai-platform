import { formatBytes } from '../../utils/uploadValidation';
import type { DocumentMeta, UploadProgress } from './types';

export function extensionOf(filename: string): string {
  const dot = filename.lastIndexOf('.');
  return dot === -1 ? '' : filename.slice(dot + 1).toUpperCase();
}

export function progressPercent(progress?: UploadProgress): number {
  if (!progress) return 0;
  if (progress.chunksTotal > 0) return Math.round((progress.chunksDone / progress.chunksTotal) * 100);
  return progress.state === 'embedding' ? 50 : 0;
}

/** "12 documents · 340 passages · 1,204 pages · 18.2 MB · 3 with older chunking", or
 *  null for an empty corpus. Pages and sizes add up only the documents that have them. */
export function corpusTotals(
  filenames: string[],
  documents: Record<string, DocumentMeta>,
): string | null {
  if (filenames.length === 0) return null;
  let passages = 0;
  let pages = 0;
  let bytes = 0;
  let stale = 0;
  for (const name of filenames) {
    const meta = documents[name];
    if (!meta) continue;
    passages += meta.chunks;
    pages += meta.pages ?? 0;
    bytes += meta.bytes ?? 0;
    if (meta.stale) stale += 1;
  }
  const plural = (count: number, word: string) =>
    `${count.toLocaleString()} ${word}${count === 1 ? '' : 's'}`;
  return [
    plural(filenames.length, 'document'),
    plural(passages, 'passage'),
    ...(pages > 0 ? [plural(pages, 'page')] : []),
    ...(bytes > 0 ? [formatBytes(bytes)] : []),
    ...(stale > 0 ? [`${stale.toLocaleString()} with older chunking`] : []),
  ].join(' · ');
}
