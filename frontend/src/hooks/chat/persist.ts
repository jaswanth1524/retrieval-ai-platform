/** Writing history: the compact localStorage copy, the full IndexedDB one, loading. */
import type { ChatTurn } from '../../components/ChatMessage';
import { newId } from '../../utils/id';
import {
  CHAT_STORAGE_FALLBACK_TURNS,
  CHAT_STORAGE_KEY,
  CHAT_STORAGE_MAX_TURNS,
  CHAT_STORAGE_VERSION,
  type ChatStorageV2,
  type Conversation,
  FULL_TEXT_CONVERSATIONS,
  LOCAL_STORAGE_MAX_CONVERSATIONS,
  PERSISTED_SOURCE_TEXT_TURNS,
  byRecentUse,
  deriveTitle,
  emptyConversation,
  lastEdit,
} from './model';
import { parseDeleted, sanitizeConversation, sanitizeTurn } from './sanitize';

/** A conversation as written to disk: the last `maxTurns` turns, with source text kept
 * only on the most recent few (see PERSISTED_SOURCE_TEXT_TURNS). */
export function compactForStorage(
  conversation: Conversation,
  maxTurns: number,
  textTurns = PERSISTED_SOURCE_TEXT_TURNS,
): Conversation {
  // The latest turns, and every saved answer with the question before it however far
  // back: a saved answer is kept by choice, even where this is the only copy (no
  // IndexedDB).
  const all = conversation.turns;
  const keep = new Set<number>();
  for (let index = Math.max(0, all.length - maxTurns); index < all.length; index += 1) keep.add(index);
  all.forEach((turn, index) => {
    if (!turn.saved) return;
    keep.add(index);
    if (index > 0 && all[index - 1].role === 'user') keep.add(index - 1);
  });
  const turns = keep.size === all.length ? all : all.filter((_, index) => keep.has(index));
  const keepTextFrom = turns.length - textTurns;
  return {
    ...conversation,
    turns: turns.map((turn, index) =>
      // A saved answer keeps its passages wherever it is: it was saved to be reread.
      index >= keepTextFrom || turn.saved || turn.sources.every((source) => !source.text)
        ? turn
        : {
            ...turn,
            // The quoted passages go with the text: both are only for reading the source.
            sources: turn.sources.map(({ spans: _spans, ...source }) => ({ ...source, text: '' })),
          },
    ),
  };
}
export type PersistOutcome = 'saved' | 'partial' | 'failed';
/** The conversations localStorage's copy holds: the active one, the pinned ones, those
 *  with a saved answer and the most recently changed, LOCAL_STORAGE_MAX_CONVERSATIONS in
 *  all. A conversation
 *  left out is never read as deleted (only a tombstone deletes; see mergeConversations).
 *
 *  "Changed" includes a rename or a pin (``lastEdit``), not only a new turn: this copy
 *  is how other tabs hear of a change, and an old conversation renamed here and left
 *  out was overwritten by the next save of a tab that never saw the rename. */
export function conversationsForLocalStorage(storage: ChatStorageV2): Conversation[] {
  const { conversations, activeConversationId } = storage;
  if (conversations.length <= LOCAL_STORAGE_MAX_CONVERSATIONS) return conversations;
  const kept = new Set<string>();
  const sorted = [...conversations].sort((a, b) => lastEdit(b) - lastEdit(a));
  for (const conversation of sorted) {
    if (conversation.id === activeConversationId) kept.add(conversation.id);
  }
  // Pinned ones, then ones holding a saved answer: both are kept by choice.
  for (const conversation of sorted) {
    if (conversation.pinned && kept.size < LOCAL_STORAGE_MAX_CONVERSATIONS) kept.add(conversation.id);
  }
  for (const conversation of sorted) {
    if (kept.size >= LOCAL_STORAGE_MAX_CONVERSATIONS) break;
    if (conversation.turns.some((turn) => turn.saved)) kept.add(conversation.id);
  }
  for (const conversation of sorted) {
    if (kept.size >= LOCAL_STORAGE_MAX_CONVERSATIONS) break;
    kept.add(conversation.id);
  }
  return conversations.filter((conversation) => kept.has(conversation.id));
}
/** The history as IndexedDB stores it: every conversation and turn, with passage text
 *  only on the FULL_TEXT_CONVERSATIONS most recently used (and the active one). */
export function forChatStore(storage: ChatStorageV2): ChatStorageV2 {
  if (storage.conversations.length <= FULL_TEXT_CONVERSATIONS) return storage;
  const full = new Set(
    [...storage.conversations]
      .sort(byRecentUse)
      .slice(0, FULL_TEXT_CONVERSATIONS)
      .map((conversation) => conversation.id),
  );
  if (storage.activeConversationId) full.add(storage.activeConversationId);
  return {
    ...storage,
    conversations: storage.conversations.map((conversation) =>
      full.has(conversation.id)
        ? conversation
        : compactForStorage(conversation, Number.MAX_SAFE_INTEGER, 0),
    ),
  };
}
/** Write the store to localStorage, falling back to the active conversation alone. */
export function isOnlyEmpty(storage: ChatStorageV2): boolean {
  return (
    storage.conversations.length === 1 &&
    storage.conversations[0].turns.length === 0 &&
    // A conversation emptied by Clear (or renamed) is an edit the other tabs must see:
    // removing the key told them nothing (a storage event with no value), and their
    // next save wrote the cleared turns back.
    !storage.conversations[0].editedAt &&
    // A recorded delete still has to reach the other tabs.
    !storage.deleted?.length
  );
}
export function persistStorage(storage: ChatStorageV2): PersistOutcome {
  const { conversations, activeConversationId } = storage;
  if (isOnlyEmpty(storage)) {
    try {
      localStorage.removeItem(CHAT_STORAGE_KEY);
    } catch {
      // Nothing more to do if even a removeItem fails.
    }
    return 'saved';
  }
  const local = conversationsForLocalStorage(storage);
  const payloadFor = (convos: Conversation[], maxTurns: number) =>
    JSON.stringify({
      version: CHAT_STORAGE_VERSION,
      activeConversationId,
      conversations: convos.map((conversation) => compactForStorage(conversation, maxTurns)),
      deleted: storage.deleted,
    });
  try {
    localStorage.setItem(CHAT_STORAGE_KEY, payloadFor(local, CHAT_STORAGE_MAX_TURNS));
    return 'saved';
  } catch {
    // Quota most likely exceeded. Retry once with only the active conversation and a
    // shorter tail rather than freezing at the last successful write with no signal —
    // this still drops older conversations from disk but keeps the current one saved.
    const reduced = conversations.filter((c) => c.id === activeConversationId);
    if (reduced.length === 0) {
      // No active match to fall back to. Writing the reduced payload here would
      // persist an empty conversations array, which loadStorage reads back as
      // "nothing saved" — a silent wipe reported as success. Fail loudly instead.
      return 'failed';
    }
    try {
      localStorage.setItem(CHAT_STORAGE_KEY, payloadFor(reduced, CHAT_STORAGE_FALLBACK_TURNS));
      // The write succeeded but every other conversation, and all but the last 20
      // turns of this one, are now absent from disk — its own signal, not a clean save.
      return 'partial';
    } catch {
      return 'failed';
    }
  }
}
export function emptyStorage(): ChatStorageV2 {
  const conversation = emptyConversation();
  return { version: 2, activeConversationId: conversation.id, conversations: [conversation] };
}
// Logs an unreadable or unrecognized-shape persisted value before it's replaced with an
// empty conversation, so the corruption is diagnosable from the console instead of
// vanishing silently. Deliberately does *not* copy the blob to a sibling storage key:
// nothing in the app ever reads such a copy back, so it would double the footprint of a
// value that may well have been what exceeded the quota, and pay that cost forever.
// Recovery would need a real UI to surface and discard it — a feature, not a side effect.
export function reportUnreadableStorage(raw: string): void {
  console.warn(
    `Discarding unreadable chat history in localStorage["${CHAT_STORAGE_KEY}"] ` +
      `(${raw.length} chars); starting a fresh conversation.`,
  );
}
export function loadStorage(): ChatStorageV2 {
  let raw: string | null = null;
  try {
    raw = localStorage.getItem(CHAT_STORAGE_KEY);
    if (!raw) return emptyStorage();
    const parsed = JSON.parse(raw) as {
      version?: number;
      turns?: unknown;
      conversations?: unknown;
      activeConversationId?: unknown;
    };

    if (parsed.version === 2 && Array.isArray(parsed.conversations)) {
      const conversations = parsed.conversations
        .map(sanitizeConversation)
        .filter((conversation): conversation is Conversation => conversation !== null);
      if (conversations.length === 0) return { ...emptyStorage(), deleted: parseDeleted(parsed) };
      const activeId =
        typeof parsed.activeConversationId === 'string' &&
        conversations.some((conversation) => conversation.id === parsed.activeConversationId)
          ? parsed.activeConversationId
          : conversations[0].id;
      return { version: 2, activeConversationId: activeId, conversations, deleted: parseDeleted(parsed) };
    }

    // v1 migration: a single flat thread becomes the first conversation. Read-only
    // until the first successful v2 write, so a crash mid-migration never loses v1.
    if (parsed.version === 1 && Array.isArray(parsed.turns)) {
      const turns = parsed.turns.map(sanitizeTurn).filter((turn): turn is ChatTurn => turn !== null);
      if (turns.length === 0) return emptyStorage();
      const conversation: Conversation = {
        id: newId(),
        title: deriveTitle(turns),
        createdAt: turns[0].timestamp,
        updatedAt: turns[turns.length - 1].timestamp,
        turns,
      };
      return { version: 2, activeConversationId: conversation.id, conversations: [conversation] };
    }

    reportUnreadableStorage(raw);
    return emptyStorage();
  } catch {
    if (raw) reportUnreadableStorage(raw);
    return emptyStorage();
  }
}
