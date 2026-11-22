/** A reminder to export the chat history now and then: it lives only in this browser,
 *  and clearing site data (or another device) loses it. Shown at most once per
 *  REMIND_EVERY_MS, only to someone with enough conversations to miss, and not after a
 *  recent export. */
import { readStored, writeStored } from './safeStorage';

export const HISTORY_EXPORTED_KEY = 'docrag-history-exported-at';
export const HISTORY_REMINDED_KEY = 'docrag-history-reminded-at';
export const REMIND_EVERY_MS = 30 * 24 * 60 * 60 * 1000;
export const REMIND_FROM_CONVERSATIONS = 10;

function storedTime(key: string): number | null {
  const value = Number(readStored(key));
  return Number.isFinite(value) && value > 0 ? value : null;
}

/** The whole history was exported now. */
export function noteHistoryExported(now = Date.now()): void {
  writeStored(HISTORY_EXPORTED_KEY, String(now));
}

/** Whether to remind now (and, when so, that it was shown: once per period). */
export function takeHistoryReminder(conversations: number, now = Date.now()): boolean {
  if (conversations < REMIND_FROM_CONVERSATIONS) return false;
  const exported = storedTime(HISTORY_EXPORTED_KEY);
  const reminded = storedTime(HISTORY_REMINDED_KEY);
  if (exported !== null && now - exported < REMIND_EVERY_MS) return false;
  if (reminded !== null && now - reminded < REMIND_EVERY_MS) return false;
  writeStored(HISTORY_REMINDED_KEY, String(now));
  return true;
}
