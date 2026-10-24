/** Saving and restoring chat history.
 *
 * localStorage holds a compact copy (instant first paint, other tabs' `storage` events);
 * IndexedDB (chatStore.ts) holds the full one. This module is the format both share:
 * versioned payloads, sanitizing what is read back, compacting what is written, and
 * merging two copies of the same history.
 */
import type { CitationResponse, DocumentTask } from '../api/types';
import type { ChatTurn } from '../components/ChatMessage';
import { newId } from '../utils/id';
import { WORKSPACE_NAME_PATTERN } from '../utils/limits';

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
  // Kept past MAX_CONVERSATIONS and listed first. A pin is an edit (editedAt).
  pinned?: boolean;
  // The workspace its questions went to (WORKSPACES_ENABLED), set by the first question;
  // absent = the default one, which is also what every conversation from before
  // workspaces was asked in. See conversationWorkspace.
  workspace?: string;
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
  pinned?: boolean;
  // Only when not the default workspace.
  workspace?: string;
}

/** The workspace a conversation belongs to ('' = the default), or null for one with no
 *  questions yet, which belongs to whichever workspace asks first. */
export function conversationWorkspace(conversation: Conversation): string | null {
  return conversation.turns.length > 0 ? (conversation.workspace ?? '') : null;
}

/** Where a search matched inside a conversation's turns. */
export interface ConversationMatch {
  turnId: string;
  snippet: string;
}

export const CHAT_STORAGE_KEY = 'docrag-chat-history';
export const CHAT_STORAGE_VERSION = 2;
// Cap persisted turns per conversation so a very long session doesn't grow
// localStorage unboundedly; older turns drop from persistence only, never from the
// in-memory session. Conversations beyond the cap are pruned oldest-updated first.
export const CHAT_STORAGE_MAX_TURNS = 100;
// The tail kept when even that doesn't fit the localStorage quota.
export const CHAT_STORAGE_FALLBACK_TURNS = 20;
// IndexedDB holds the history now (chatStore.ts); 20 dated from when localStorage's
// ~5 MB quota held all of it, and the 21st conversation silently dropped the oldest.
export const MAX_CONVERSATIONS = 100;
// localStorage keeps only a copy for first paint and the other tabs' storage events:
// the active conversation, the pinned ones, and the most recently used, up to this.
export const LOCAL_STORAGE_MAX_CONVERSATIONS = 20;
// IndexedDB keeps passage text on this many most recently used conversations; older
// ones keep their citations (file, page, section, chunk id), which open the viewer.
// The whole history is one record, rewritten on every save.
export const FULL_TEXT_CONVERSATIONS = 20;
// Pinned conversations are never the ones dropped past MAX_CONVERSATIONS; at most this
// many, so there is always room left for the conversations in use.
export const MAX_PINNED_CONVERSATIONS = 10;
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

// The range a Date can hold. A number outside it (an import with `"timestamp": 1e300`)
// passed the type check, and then every render threw "Invalid time value" on it.
const MAX_TIME_MS = 8.64e15;

/** ``value`` when it is a time a Date can format, else ``fallback``. */
export function validTime(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) && Math.abs(value) <= MAX_TIME_MS
    ? value
    : fallback;
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

/** The citation, without its spans when any of them isn't one: the citation still opens
 *  its source, just with nothing marked. */
function withValidSpans(source: CitationResponse): CitationResponse {
  // A page range is shown as text: anything but a whole number is dropped.
  if (source.page_end != null && !Number.isInteger(source.page_end)) {
    const { page_end: _pageEnd, ...rest } = source;
    return withValidSpans(rest);
  }
  if (source.spans === undefined) return source;
  const valid =
    Array.isArray(source.spans) &&
    source.spans.every(
      (span: unknown) =>
        !!span &&
        typeof span === 'object' &&
        typeof (span as { start?: unknown }).start === 'number' &&
        typeof (span as { end?: unknown }).end === 'number' &&
        typeof (span as { quote?: unknown }).quote === 'string',
    );
  if (valid) return source;
  const { spans: _spans, ...rest } = source;
  return rest;
}

/** A turn read from disk or an imported file, with every field the UI renders checked.
 *
 * Only the top-level shape used to be checked, so an imported file with `model: {}` or
 * a `null` source was saved — and crashed the page on every load once it rendered. */
// The server sends at most 5 follow-ups of at most 200 characters (api/prompting.py).
const MAX_FOLLOWUPS = 5;
const FOLLOWUP_MAX_CHARS = 200;

/** Suggested follow-ups as the thread can render them: strings with text, each once —
 *  they are the buttons' React keys, so a repeated one (a model can suggest the same
 *  question twice) broke rendering. From the stream and from disk alike. */
export function cleanFollowups(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  const followups = (value as unknown[]).filter(
    (item): item is string =>
      typeof item === 'string' && item.trim().length > 0 && item.length <= FOLLOWUP_MAX_CHARS,
  );
  return [...new Set(followups)].slice(0, MAX_FOLLOWUPS);
}

/** ``value`` as a document task when it is one the server would accept, else null. */
export function sanitizeTask(value: unknown): DocumentTask | null {
  if (!value || typeof value !== 'object') return null;
  const task = value as Record<string, unknown>;
  const mode = task.mode;
  if (mode !== 'summarize' && mode !== 'compare' && mode !== 'brief') return null;
  const names = task.filenames;
  if (!Array.isArray(names) || !names.every((name) => typeof name === 'string' && name !== '')) {
    return null;
  }
  const countOk =
    mode === 'brief' ? names.length >= 2 && names.length <= 50 : names.length === (mode === 'compare' ? 2 : 1);
  if (!countOk || new Set(names).size !== names.length) {
    return null;
  }
  const sanitized: DocumentTask = { mode, filenames: [...(names as string[])] };
  if (typeof task.instruction === 'string' && task.instruction.trim()) {
    sanitized.instruction = task.instruction;
  }
  return sanitized;
}

function isCount(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0;
}

export function sanitizeTurn(value: unknown): ChatTurn | null {
  if (!isChatTurn(value)) return null;
  const turn = value as ChatTurn & Record<string, unknown>;
  const timings = turn.timings as unknown;
  const sanitized: ChatTurn = {
    id: turn.id,
    role: turn.role,
    content: turn.content,
    timestamp: validTime(turn.timestamp, Date.now()),
    sources: turn.sources.filter(isCitation).map(withValidSpans),
    timings:
      timings && typeof timings === 'object' && typeof (timings as { total_ms?: unknown }).total_ms === 'number'
        ? (timings as ChatTurn['timings'])
        : null,
    traceId: typeof turn.traceId === 'string' ? turn.traceId : null,
  };
  if (typeof turn.question === 'string') sanitized.question = turn.question;
  if (turn.feedback === 'up' || turn.feedback === 'down') sanitized.feedback = turn.feedback;
  if (typeof turn.model === 'string') sanitized.model = turn.model;
  const flags = [
    'stopped',
    'incomplete',
    'cached',
    'notFound',
    'uncited',
    'truncated',
    'followUp',
    'weakMatch',
    'autoCited',
    'scoped',
  ] as const;
  for (const flag of flags) {
    if (turn[flag] === true) sanitized[flag] = true;
  }
  if (Array.isArray(turn.contextSources)) {
    const context = turn.contextSources.filter(isCitation).map(withValidSpans).slice(0, 20);
    if (context.length > 0) sanitized.contextSources = context;
  }
  const task = sanitizeTask(turn.task);
  if (task) sanitized.task = task;
  const followups = cleanFollowups(turn.followups);
  if (followups.length > 0) sanitized.followups = followups;
  if (Array.isArray(turn.irrelevantSources)) {
    const numbers = turn.irrelevantSources.filter(
      (value): value is number => Number.isInteger(value) && value >= 1 && value <= 1000,
    );
    if (numbers.length > 0) sanitized.irrelevantSources = [...new Set(numbers)].slice(0, 50);
  }
  if (typeof turn.topRerankScore === 'number' && turn.topRerankScore >= 0 && turn.topRerankScore <= 1) {
    sanitized.topRerankScore = turn.topRerankScore;
  }
  const usage = turn.usage as unknown;
  if (usage && typeof usage === 'object') {
    const { prompt, completion } = usage as Record<string, unknown>;
    if (isCount(prompt) && isCount(completion)) sanitized.usage = { prompt, completion };
  }
  const coverage = turn.coverage as unknown;
  if (coverage && typeof coverage === 'object') {
    const { read, total } = coverage as Record<string, unknown>;
    if (isCount(read) && isCount(total) && read < total) sanitized.coverage = { read, total };
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

/** ``conversations`` cut to MAX_CONVERSATIONS, most recently used first. The ones
 *  dropped are the least recently used unpinned ones; ``keepId`` (a conversation being
 *  created, or streamed into) is never dropped. */
export function capConversations(
  conversations: readonly Conversation[],
  keepId: string | null = null,
): Conversation[] {
  const sorted = [...conversations].sort(byRecentUse);
  if (sorted.length <= MAX_CONVERSATIONS) return sorted;
  const protectedIds = new Set(
    sorted
      .filter((conversation) => conversation.pinned)
      .slice(0, MAX_PINNED_CONVERSATIONS)
      .map((conversation) => conversation.id),
  );
  if (keepId !== null) protectedIds.add(keepId);
  let room = MAX_CONVERSATIONS - protectedIds.size;
  return sorted.filter((conversation) => {
    if (protectedIds.has(conversation.id)) return true;
    room -= 1;
    return room >= 0;
  });
}

const SNIPPET_CONTEXT_CHARS = 40;

/** The first turn of each conversation whose text contains ``query`` (any case), with
 *  a snippet around the match. Titles are matched by the list itself. */
export function searchConversations(
  conversations: readonly Conversation[],
  query: string,
): Map<string, ConversationMatch> {
  const needle = query.trim().toLowerCase();
  const matches = new Map<string, ConversationMatch>();
  if (!needle) return matches;
  for (const conversation of conversations) {
    for (const turn of conversation.turns) {
      if (turn.role === 'error') continue;
      const text = turn.content.replace(/\s+/g, ' ');
      const at = text.toLowerCase().indexOf(needle);
      if (at < 0) continue;
      const start = Math.max(0, at - SNIPPET_CONTEXT_CHARS);
      const end = Math.min(text.length, at + needle.length + SNIPPET_CONTEXT_CHARS);
      matches.set(conversation.id, {
        turnId: turn.id,
        snippet: `${start > 0 ? '…' : ''}${text.slice(start, end).trim()}${end < text.length ? '…' : ''}`,
      });
      break;
    }
  }
  return matches;
}

/** A conversation as written to disk: the last `maxTurns` turns, with source text kept
 * only on the most recent few (see PERSISTED_SOURCE_TEXT_TURNS). */
export function compactForStorage(
  conversation: Conversation,
  maxTurns: number,
  textTurns = PERSISTED_SOURCE_TEXT_TURNS,
): Conversation {
  const turns = conversation.turns.slice(-maxTurns);
  const keepTextFrom = turns.length - textTurns;
  return {
    ...conversation,
    turns: turns.map((turn, index) =>
      index >= keepTextFrom || turn.sources.every((source) => !source.text)
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

/** The conversations localStorage's copy holds: the active one, the pinned ones and
 *  the most recently changed, LOCAL_STORAGE_MAX_CONVERSATIONS in all. A conversation
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
  for (const conversation of sorted) {
    if (conversation.pinned && kept.size < LOCAL_STORAGE_MAX_CONVERSATIONS) kept.add(conversation.id);
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

/** ``incoming`` with passage text (and quoted spans) put back from ``mine`` where it
 *  was stripped.
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
        source.text
          ? source
          : // The quoted passages were stripped with the text (compactForStorage).
            { ...source, text: own.sources[index].text, spans: own.sources[index].spans },
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
 * after the delete — losing a delete beats losing a conversation still in use.
 *
 * ``streamingId`` names the conversation an answer is streaming into in this tab. The
 * other tab's copy of it lacks the question and the partial answer, so it is never
 * dropped or replaced here: a rename or scope change from the other tab is taken, its
 * turns are not. A delete recorded meanwhile is kept in ``deleted`` and applied when the
 * stream ends (``applyDeleteHeldDuringStream``). */
export function mergeConversations(
  local: ChatStorageV2,
  incoming: Conversation[],
  // IndexedDB's copy of a conversation is the complete one (localStorage's drops old
  // passage text), so on equal timestamps it wins.
  preferIncomingOnTie = false,
  incomingDeleted?: DeletedConversation[],
  now = Date.now(),
  streamingId: string | null = null,
): ChatStorageV2 {
  const deleted = mergeTombstones(local.deleted, incomingDeleted, now);
  const deletedAt = new Map(deleted.map(({ id, at }) => [id, at]));
  const isDeleted = (conversation: Conversation) =>
    conversation.id !== streamingId &&
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
      if (mine && conversation.id === streamingId) {
        const scope = conversation.scope ?? mine.scope;
        if (
          conversation.title === mine.title &&
          Boolean(conversation.pinned) === Boolean(mine.pinned) &&
          JSON.stringify(scope) === JSON.stringify(mine.scope)
        ) {
          continue;
        }
        byId.set(conversation.id, {
          ...mine,
          title: conversation.title,
          scope,
          pinned: conversation.pinned,
          editedAt: Math.max(lastEdit(mine), conversation.editedAt ?? 0),
        });
        changed = true;
        continue;
      }
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
  // Past the cap the streaming conversation stays, in place of the least recently used.
  const conversations = capConversations([...byId.values()], streamingId);
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

/** ``storage`` without conversation ``id``, with a tombstone at ``at`` so another tab's
 *  copy of it doesn't bring it back on its next save. */
export function withoutConversation(
  storage: ChatStorageV2,
  id: string,
  at: number,
): ChatStorageV2 {
  const deleted = [...(storage.deleted ?? []).filter((entry) => entry.id !== id), { id, at }];
  const remaining = storage.conversations.filter((conversation) => conversation.id !== id);
  if (remaining.length === 0) return { ...emptyStorage(), deleted };
  const activeConversationId =
    storage.activeConversationId === id
      ? [...remaining].sort(byRecentUse)[0].id
      : storage.activeConversationId;
  return { ...storage, activeConversationId, conversations: remaining, deleted };
}

/** Another tab deleted conversation ``id`` while an answer streamed into it here (at or
 *  after ``since``): ``mergeConversations`` held the delete back so the stream had
 *  somewhere to go, and this applies it once the stream is over. The tombstone is
 *  re-stamped at ``now``, after the answer's own last edit — left at its original time,
 *  this tab's save of the finished answer counted as an edit after the delete and
 *  brought the conversation back in the tab that deleted it. */
export function applyDeleteHeldDuringStream(
  storage: ChatStorageV2,
  id: string,
  since: number,
  now = Date.now(),
): ChatStorageV2 {
  const held = storage.deleted?.some((entry) => entry.id === id && entry.at >= since);
  return held ? withoutConversation(storage, id, now) : storage;
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
    createdAt: validTime(raw.createdAt, Date.now()),
    updatedAt: validTime(raw.updatedAt, Date.now()),
    turns,
    scope: sanitizeScope(raw.scope),
  };
  if (validTime(raw.editedAt, NaN) === raw.editedAt) conversation.editedAt = raw.editedAt;
  if (raw.pinned === true) conversation.pinned = true;
  if (typeof raw.workspace === 'string' && WORKSPACE_NAME_PATTERN.test(raw.workspace)) {
    conversation.workspace = raw.workspace;
  }
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
