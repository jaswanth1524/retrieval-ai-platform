/** Saving and restoring chat history.
 *
 * localStorage holds a compact copy (instant first paint, other tabs' `storage` events);
 * IndexedDB (chatStore.ts) holds the full one. This module is the format both share:
 * versioned payloads, sanitizing what is read back, compacting what is written, and
 * merging two copies of the same history.
 */
import type { ChatTurn } from '../components/ChatMessage';
import { newId } from '../utils/id';

/** Which documents a conversation's questions search. Empty lists mean "everything". */
export interface ConversationScope {
  filenames: string[];
  tags: string[];
}

export const EMPTY_SCOPE: ConversationScope = { filenames: [], tags: [] };

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  turns: ChatTurn[];
  // Per conversation, so a follow-up in another conversation can't silently search
  // the documents this one was scoped to. Absent on conversations saved before it.
  scope?: ConversationScope;
}

export interface ConversationSummary {
  id: string;
  title: string;
  updatedAt: number;
  turnCount: number;
}

export const CHAT_STORAGE_KEY = 'docrag-chat-history';
export const CHAT_STORAGE_VERSION = 2;
// Cap persisted turns per conversation so a very long session doesn't grow
// localStorage unboundedly; older turns drop from persistence only, never from the
// in-memory session. Conversations beyond the cap are pruned oldest-updated first.
export const CHAT_STORAGE_MAX_TURNS = 100;
export const MAX_CONVERSATIONS = 20;
export const TITLE_MAX_CHARS = 60;
// Each source carries its full chunk text (~2 KB), up to 8 per answer, and it was the
// bulk of what filled the ~5 MB quota. Only the most recent turns keep it on disk; older
// ones keep the citation (file, page, section, chunk id), which is what opens the viewer.
export const PERSISTED_SOURCE_TEXT_TURNS = 6;

export interface ChatStorageV2 {
  version: 2;
  activeConversationId: string | null;
  conversations: Conversation[];
}

export function isChatTurn(value: unknown): value is ChatTurn {
  if (!value || typeof value !== 'object') return false;
  const turn = value as Partial<ChatTurn>;
  return (
    typeof turn.id === 'string' &&
    (turn.role === 'user' || turn.role === 'assistant' || turn.role === 'error') &&
    typeof turn.content === 'string' &&
    Array.isArray(turn.sources) &&
    typeof turn.timestamp === 'number'
  );
}

export function deriveTitle(turns: ChatTurn[]): string {
  const firstUser = turns.find((turn) => turn.role === 'user' && turn.content.trim());
  if (!firstUser) return 'New chat';
  const collapsed = firstUser.content.replace(/\s+/g, ' ').trim();
  return collapsed.slice(0, TITLE_MAX_CHARS) || 'New chat';
}

export function emptyConversation(): Conversation {
  const now = Date.now();
  return { id: newId(), title: 'New chat', createdAt: now, updatedAt: now, turns: [] };
}

export function byRecentUse(a: Conversation, b: Conversation): number {
  return b.updatedAt - a.updatedAt;
}

/** A conversation as written to disk: the last `maxTurns` turns, with source text kept
 * only on the most recent few (see PERSISTED_SOURCE_TEXT_TURNS). */
export function compactForStorage(conversation: Conversation, maxTurns: number): Conversation {
  const turns = conversation.turns.slice(-maxTurns);
  const keepTextFrom = turns.length - PERSISTED_SOURCE_TEXT_TURNS;
  return {
    ...conversation,
    turns: turns.map((turn, index) =>
      index >= keepTextFrom || turn.sources.every((source) => !source.text)
        ? turn
        : { ...turn, sources: turn.sources.map((source) => ({ ...source, text: '' })) },
    ),
  };
}

export type PersistOutcome = 'saved' | 'partial' | 'failed';

/** Write the store to localStorage, falling back to the active conversation alone. */
export function isOnlyEmpty(storage: ChatStorageV2): boolean {
  return storage.conversations.length === 1 && storage.conversations[0].turns.length === 0;
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
  const payloadFor = (convos: Conversation[], maxTurns: number) =>
    JSON.stringify({
      version: CHAT_STORAGE_VERSION,
      activeConversationId,
      conversations: convos.map((conversation) => compactForStorage(conversation, maxTurns)),
    });
  try {
    localStorage.setItem(CHAT_STORAGE_KEY, payloadFor(conversations, CHAT_STORAGE_MAX_TURNS));
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
      localStorage.setItem(CHAT_STORAGE_KEY, payloadFor(reduced, 20));
      // The write succeeded but every other conversation, and all but the last 20
      // turns of this one, are now absent from disk — its own signal, not a clean save.
      return 'partial';
    } catch {
      return 'failed';
    }
  }
}

/** ``incoming`` with passage text put back from ``mine`` where it was stripped.
 *
 * Another tab's localStorage copy drops old passage text; merged over this tab's full
 * copy as-is, the next IndexedDB save would have stored the stripped version. */
export function keepPassageText(incoming: Conversation, mine: Conversation): Conversation {
  const mineById = new Map(mine.turns.map((turn) => [turn.id, turn]));
  let restored = false;
  const turns = incoming.turns.map((turn) => {
    const own = mineById.get(turn.id);
    if (!own || turn.sources.length !== own.sources.length) return turn;
    if (!turn.sources.some((source, index) => !source.text && own.sources[index].text)) return turn;
    restored = true;
    return {
      ...turn,
      sources: turn.sources.map((source, index) =>
        source.text ? source : { ...source, text: own.sources[index].text },
      ),
    };
  });
  return restored ? { ...incoming, turns } : incoming;
}

/** Fold another tab's saved conversations into this tab's, newest copy of each winning.
 *
 * Every tab used to rewrite the whole blob from its own memory, so asking in tab B
 * erased what tab A had just saved. A conversation deleted in the other tab survives
 * here if this tab still has it — losing a delete beats losing a conversation. */
export function mergeConversations(
  local: ChatStorageV2,
  incoming: Conversation[],
  // IndexedDB's copy of a conversation is the complete one (localStorage's drops old
  // passage text), so on equal timestamps it wins.
  preferIncomingOnTie = false,
): ChatStorageV2 {
  const byId = new Map(local.conversations.map((conversation) => [conversation.id, conversation]));
  let changed = false;
  for (const conversation of incoming) {
    const mine = byId.get(conversation.id);
    if (
      !mine ||
      conversation.updatedAt > mine.updatedAt ||
      (preferIncomingOnTie && conversation.updatedAt === mine.updatedAt)
    ) {
      byId.set(conversation.id, mine ? keepPassageText(conversation, mine) : conversation);
      changed = true;
    }
  }
  if (!changed) return local;
  const conversations = [...byId.values()].sort(byRecentUse).slice(0, MAX_CONVERSATIONS);
  const activeConversationId = conversations.some((c) => c.id === local.activeConversationId)
    ? local.activeConversationId
    : conversations[0].id;
  return { ...local, activeConversationId, conversations };
}

/** The conversations in a saved v2 payload (either store), or null when unreadable. */
export function parseConversations(value: unknown): Conversation[] | null {
  if (!value || typeof value !== 'object') return null;
  const parsed = value as { version?: number; conversations?: unknown };
  if (parsed.version !== 2 || !Array.isArray(parsed.conversations)) return null;
  return parsed.conversations
    .map(sanitizeConversation)
    .filter((conversation): conversation is Conversation => conversation !== null);
}

export function emptyStorage(): ChatStorageV2 {
  const conversation = emptyConversation();
  return { version: 2, activeConversationId: conversation.id, conversations: [conversation] };
}

export function sanitizeConversation(value: unknown): Conversation | null {
  if (!value || typeof value !== 'object') return null;
  const raw = value as Partial<Conversation>;
  if (typeof raw.id !== 'string' || !Array.isArray(raw.turns)) return null;
  const turns = raw.turns.filter(isChatTurn);
  return {
    id: raw.id,
    title: typeof raw.title === 'string' && raw.title ? raw.title : deriveTitle(turns),
    createdAt: typeof raw.createdAt === 'number' ? raw.createdAt : Date.now(),
    updatedAt: typeof raw.updatedAt === 'number' ? raw.updatedAt : Date.now(),
    turns,
    scope: sanitizeScope(raw.scope),
  };
}

export function sanitizeScope(value: unknown): ConversationScope | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const raw = value as Partial<ConversationScope>;
  const strings = (list: unknown) =>
    Array.isArray(list) ? list.filter((item): item is string => typeof item === 'string') : [];
  return { filenames: strings(raw.filenames), tags: strings(raw.tags) };
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
      if (conversations.length === 0) return emptyStorage();
      const activeId =
        typeof parsed.activeConversationId === 'string' &&
        conversations.some((conversation) => conversation.id === parsed.activeConversationId)
          ? parsed.activeConversationId
          : conversations[0].id;
      return { version: 2, activeConversationId: activeId, conversations };
    }

    // v1 migration: a single flat thread becomes the first conversation. Read-only
    // until the first successful v2 write, so a crash mid-migration never loses v1.
    if (parsed.version === 1 && Array.isArray(parsed.turns)) {
      const turns = parsed.turns.filter(isChatTurn);
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
