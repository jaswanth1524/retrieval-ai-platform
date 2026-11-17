/** Where the reader was in each document (the first passage in view when they left),
 *  so "Read" opens it there again. Kept in this browser, the newest MAX_DOCUMENTS. */
import { readStored, writeStored } from './safeStorage';

export const READING_STORAGE_KEY = 'docrag-reading';
const MAX_DOCUMENTS = 200;

type Places = Record<string, { chunkId: string; at: number }>;

function key(workspace: string, filename: string): string {
  return `${workspace}\u0000${filename}`;
}

function read(): Places {
  try {
    const raw: unknown = JSON.parse(readStored(READING_STORAGE_KEY) ?? '{}');
    return raw && typeof raw === 'object' && !Array.isArray(raw) ? (raw as Places) : {};
  } catch {
    return {};
  }
}

export function rememberReading(workspace: string, filename: string, chunkId: string): void {
  const places = read();
  places[key(workspace, filename)] = { chunkId: chunkId.slice(0, 64), at: Date.now() };
  const kept = Object.entries(places)
    .filter(([, place]) => place && typeof place.chunkId === 'string' && typeof place.at === 'number')
    .sort((a, b) => b[1].at - a[1].at)
    .slice(0, MAX_DOCUMENTS);
  writeStored(READING_STORAGE_KEY, JSON.stringify(Object.fromEntries(kept)));
}

/** The passage to reopen ``filename`` at, or null to start from the top. */
export function recallReading(workspace: string, filename: string): string | null {
  const place = read()[key(workspace, filename)];
  return place && typeof place.chunkId === 'string' && place.chunkId ? place.chunkId : null;
}
