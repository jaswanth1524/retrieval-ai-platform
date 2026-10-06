/** Saving and restoring chat history.
 *
 * localStorage holds a compact copy (instant first paint, other tabs' `storage` events);
 * IndexedDB (chatStore.ts) holds the full one. This module is the format both share:
 * versioned payloads, sanitizing what is read back, compacting what is written, and
 * merging two copies of the same history.
 */
import type { CitationResponse } from '../api/types';
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
  // Last change of any kind — a rename, a scope or rating change — which `updatedAt`
  // deliberately ignores so the list order doesn't jump. Merging two tabs' copies
  // compares this, or a rename in one tab was overwritten by the other's next save.
  editedAt?: number;
}

/** A conversation deleted in some tab, so another tab's copy doesn't bring it back. */
export interface DeletedConversation {
  id: string;
  at: number;
}

// Long enough for any tab left open to have seen the delete.
export const TOMBSTONE_MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000;

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
// The tail kept when even that doesn't fit the localStorage quota.
export const CHAT_STORAGE_FALLBACK_TURNS = 20;
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
  // Absent in payloads from before deletes were recorded; old readers ignore it.
  deleted?: DeletedConversation[];
}

/** When a conversation last changed in any way: what a cross-tab merge compares. */
export function lastEdit(conversation: Conversation): number {
  return Math.max(conversation.updatedAt, conversation.editedAt ?? 0);
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

function isCitation(value: unknown): value is CitationResponse {
  if (!value || typeof value !== 'object') return false;
  const source = value as Partial<CitationResponse>;
  return (
    typeof source.source_number === 'number' &&
    typeof source.filename === 'string' &&
    typeof source.page === 'number' &&
    typeof source.section === 'string' &&
    typeof source.chunk_id === 'string' &&
    typeof source.text === 'string'
  );
}

/** A turn read from disk or an imported file, with every field the UI renders checked.
 *
 * Only the top-level shape used to be checked, so an imported file with `model: {}` or
 * a `null` source was saved — and crashed the page on every load once it rendered. */
export function sanitizeTurn(value: unknown): ChatTurn | null {
  if (!isChatTurn(value)) return null;
  const turn = value as ChatTurn & Record<string, unknown>;
  const timings = turn.timings as unknown;
  const sanitized: ChatTurn = {
    id: turn.id,
    role: turn.role,
    content: turn.content,
    timestamp: turn.timestamp,
    sources: turn.sources.filter(isCitation),
    timings:
      timings && typeof timings === 'object' && typeof (timings as { total_ms?: unknown }).total_ms === 'number'
        ? (timings as ChatTurn['timings'])
        : null,
    traceId: typeof turn.traceId === 'string' ? turn.traceId : null,
  };
  if (typeof turn.question === 'string') sanitized.question = turn.question;
  if (turn.feedback === 'up' || turn.feedback === 'down') sanitized.feedback = turn.feedback;
  if (typeof turn.model === 'string') sanitized.model = turn.model;
  for (const flag of ['stopped', 'incomplete', 'cached'] as const) {
    if (turn[flag] === true) sanitized[flag] = true;
  }
  return sanitized;
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
  return (
    storage.conversations.length === 1 &&
    storage.conversations[0].turns.length === 0 &&
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
  const payloadFor = (convos: Conversation[], maxTurns: number) =>
    JSON.stringify({
      version: CHAT_STORAGE_VERSION,
      activeConversationId,
      conversations: convos.map((conversation) => compactForStorage(conversation, maxTurns)),
      deleted: storage.deleted,
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
      localStorage.setItem(CHAT_STORAGE_KEY, payloadFor(reduced, CHAT_STORAGE_FALLBACK_TURNS));
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

/** ``incoming`` with this tab's older turns put back where a storage cap cut them.
 *
 * localStorage keeps the last CHAT_STORAGE_MAX_TURNS turns (fewer on a quota fallback).
 * Merged over this tab's longer copy as-is, the next IndexedDB save stored the cut copy
 * and the older turns were gone for good. Only a copy of exactly a cap's length, whose
 * first turn this tab has further in, is treated as cut. */
export function keepOlderTurns(incoming: Conversation, mine: Conversation): Conversation {
  const length = incoming.turns.length;
  if (length !== CHAT_STORAGE_MAX_TURNS && length !== CHAT_STORAGE_FALLBACK_TURNS) return incoming;
  const start = mine.turns.findIndex((turn) => turn.id === incoming.turns[0]?.id);
  if (start <= 0) return incoming;
  return { ...incoming, turns: [...mine.turns.slice(0, start), ...incoming.turns] };
}

/** Both tabs' deletes, newest per id, without the ones old enough to forget. */
function mergeTombstones(
  mine: DeletedConversation[] | undefined,
  theirs: DeletedConversation[] | undefined,
  now: number,
): DeletedConversation[] {
  const byId = new Map<string, number>();
  for (const { id, at } of [...(mine ?? []), ...(theirs ?? [])]) {
    if (now - at <= TOMBSTONE_MAX_AGE_MS) byId.set(id, Math.max(at, byId.get(id) ?? 0));
  }
  return [...byId].map(([id, at]) => ({ id, at }));
}

/** Fold another tab's saved conversations into this tab's, newest copy of each winning.
 *
 * Every tab used to rewrite the whole blob from its own memory, so asking in tab B
 * erased what tab A had just saved. "Newest" is the last edit of any kind (renames
 * included). A conversation deleted in either tab stays deleted unless it was changed
 * after the delete — losing a delete beats losing a conversation still in use. */
export function mergeConversations(
  local: ChatStorageV2,
  incoming: Conversation[],
  // IndexedDB's copy of a conversation is the complete one (localStorage's drops old
  // passage text), so on equal timestamps it wins.
  preferIncomingOnTie = false,
  incomingDeleted?: DeletedConversation[],
  now = Date.now(),
): ChatStorageV2 {
  const deleted = mergeTombstones(local.deleted, incomingDeleted, now);
  const deletedAt = new Map(deleted.map(({ id, at }) => [id, at]));
  const isDeleted = (conversation: Conversation) =>
    (deletedAt.get(conversation.id) ?? -Infinity) >= lastEdit(conversation);
  const byId = new Map(local.conversations.map((conversation) => [conversation.id, conversation]));
  let changed = false;
  for (const conversation of local.conversations) {
    if (isDeleted(conversation)) {
      byId.delete(conversation.id);
      changed = true;
    }
  }
  for (const conversation of incoming) {
    if (isDeleted(conversation)) continue;
    const mine = byId.get(conversation.id);
    if (
      !mine ||
      lastEdit(conversation) > lastEdit(mine) ||
      (preferIncomingOnTie && lastEdit(conversation) === lastEdit(mine))
    ) {
      byId.set(
        conversation.id,
        mine ? keepOlderTurns(keepPassageText(conversation, mine), mine) : conversation,
      );
      changed = true;
    }
  }
  const tombstonesChanged =
    deleted.length !== (local.deleted?.length ?? 0) ||
    deleted.some(({ id, at }) => !local.deleted?.some((d) => d.id === id && d.at === at));
  if (!changed && !tombstonesChanged) return local;
  if (!changed) return { ...local, deleted };
  const conversations = [...byId.values()].sort(byRecentUse).slice(0, MAX_CONVERSATIONS);
  if (conversations.length === 0) return { ...emptyStorage(), deleted };
  const activeConversationId = conversations.some((c) => c.id === local.activeConversationId)
    ? local.activeConversationId
    : conversations[0].id;
  return { ...local, activeConversationId, conversations, deleted };
}

/** The recorded deletes in a saved v2 payload (either store). */
export function parseDeleted(value: unknown): DeletedConversation[] | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const deleted = (value as { deleted?: unknown }).deleted;
  if (!Array.isArray(deleted)) return undefined;
  return deleted.filter(
    (entry): entry is DeletedConversation =>
      !!entry &&
      typeof entry === 'object' &&
      typeof (entry as DeletedConversation).id === 'string' &&
      typeof (entry as DeletedConversation).at === 'number',
  );
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
  const turns = raw.turns.map(sanitizeTurn).filter((turn): turn is ChatTurn => turn !== null);
  const conversation: Conversation = {
    id: raw.id,
    title: typeof raw.title === 'string' && raw.title ? raw.title : deriveTitle(turns),
    createdAt: typeof raw.createdAt === 'number' ? raw.createdAt : Date.now(),
    updatedAt: typeof raw.updatedAt === 'number' ? raw.updatedAt : Date.now(),
    turns,
    scope: sanitizeScope(raw.scope),
  };
  if (typeof raw.editedAt === 'number') conversation.editedAt = raw.editedAt;
  return conversation;
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
