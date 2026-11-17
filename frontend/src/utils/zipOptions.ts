/** How zips are uploaded, remembered in this browser: named by their path inside the
 *  zip (reports/2024/q1.pdf) or their base name, and imported now (the indexing queue's
 *  worth, the rest deferred) or in the background (every document, as room frees). */
import { readStored, writeStored } from './safeStorage';

export const ZIP_OPTIONS_STORAGE_KEY = 'docrag-zip-options';
/** Fired when an import was started here, so the import list looks again at once. */
export const BACKGROUND_IMPORT_STARTED_EVENT = 'docrag-background-import-started';

export interface ZipOptions {
  keepPaths: boolean;
  background: boolean;
}

export function readZipOptions(): ZipOptions {
  try {
    const raw: unknown = JSON.parse(readStored(ZIP_OPTIONS_STORAGE_KEY) ?? '{}');
    const value = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {};
    return { keepPaths: value.keepPaths === true, background: value.background === true };
  } catch {
    return { keepPaths: false, background: false };
  }
}

export function writeZipOptions(options: ZipOptions): void {
  writeStored(ZIP_OPTIONS_STORAGE_KEY, JSON.stringify(options));
}
