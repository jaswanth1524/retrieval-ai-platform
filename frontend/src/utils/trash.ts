/** Recently deleted conversations, kept in this browser so a delete can be taken back
 *  after its Undo toast is gone: the newest TRASH_MAX_CONVERSATIONS, each for as long
 *  as the delete's tombstone lasts (TOMBSTONE_MAX_AGE_MS). Passage text is dropped (as
 *  for the localStorage copy): a restored answer opens its sources from the server. */
import { compactForStorage } from '../hooks/chat/persist';
import { sanitizeConversation } from '../hooks/chat/sanitize';
import { CHAT_STORAGE_MAX_TURNS, TOMBSTONE_MAX_AGE_MS, type Conversation } from '../hooks/chat/model';

export const TRASH_STORAGE_KEY = 'docrag-trash';
export const TRASH_MAX_CONVERSATIONS = 20;

export interface TrashedConversation {
  conversation: Conversation;
  deletedAt: number;
}

export function readTrash(now = Date.now()): TrashedConversation[] {
  let raw: unknown;
  try {
    raw = JSON.parse(localStorage.getItem(TRASH_STORAGE_KEY) ?? '[]');
  } catch {
    return [];
  }
  if (!Array.isArray(raw)) return [];
  const kept: TrashedConversation[] = [];
  for (const entry of raw) {
    if (!entry || typeof entry !== 'object') continue;
    const { conversation, deletedAt } = entry as { conversation?: unknown; deletedAt?: unknown };
    const parsed = sanitizeConversation(conversation);
    if (!parsed || typeof deletedAt !== 'number' || now - deletedAt > TOMBSTONE_MAX_AGE_MS) continue;
    kept.push({ conversation: parsed, deletedAt });
  }
  return kept.sort((a, b) => b.deletedAt - a.deletedAt);
}

function writeTrash(entries: TrashedConversation[]): void {
  try {
    localStorage.setItem(TRASH_STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // Full or private: the oldest half goes, then nothing more is kept.
    try {
      localStorage.setItem(
        TRASH_STORAGE_KEY,
        JSON.stringify(entries.slice(0, Math.floor(entries.length / 2))),
      );
    } catch {
      // Kept only as long as the Undo toast.
    }
  }
}

/** ``conversation`` (just deleted) into the trash. */
export function addToTrash(conversation: Conversation, now = Date.now()): void {
  const compact = compactForStorage(conversation, CHAT_STORAGE_MAX_TURNS, 0);
  const others = readTrash(now).filter((entry) => entry.conversation.id !== conversation.id);
  writeTrash([{ conversation: compact, deletedAt: now }, ...others].slice(0, TRASH_MAX_CONVERSATIONS));
}

/** Out of the trash: restored, or deleted for good. */
export function removeFromTrash(id: string): void {
  writeTrash(readTrash().filter((entry) => entry.conversation.id !== id));
}
