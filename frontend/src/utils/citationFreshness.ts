// Whether the document a citation points at is still the one the answer was drawn from.

/** Why a cited source may no longer say what the answer quoted. */
export type SourceChange = 'deleted' | 'changed';

/** What the corpus listing knows of a document: its upload time in Unix seconds. */
interface ListedDocument {
  uploadedAt?: number;
}

/** A citation's document as it is now, against an answer given at ``answeredAtMs``:
 *  "deleted" when the listing no longer has it, "changed" when it was uploaded again
 *  (or re-indexed, or refreshed from its URL) after the answer. ``null`` when it is the
 *  same, or when the listing hasn't loaded yet (absent then means nothing). */
export function sourceChange(
  filename: string,
  answeredAtMs: number,
  documents: Readonly<Record<string, ListedDocument>>,
  listed: boolean,
): SourceChange | null {
  if (!listed) return null;
  const document = documents[filename];
  if (!document) return 'deleted';
  // Seconds in the listing, milliseconds on the turn.
  if (typeof document.uploadedAt === 'number' && document.uploadedAt * 1000 > answeredAtMs) {
    return 'changed';
  }
  return null;
}

export const SOURCE_CHANGE_LABELS: Record<SourceChange, string> = {
  deleted: 'document deleted',
  changed: 'document changed since',
};

export const SOURCE_CHANGE_HINTS: Record<SourceChange, string> = {
  deleted: 'This document was deleted after the answer was given.',
  changed:
    'This document was uploaded again after the answer was given: the passage may read differently now.',
};
