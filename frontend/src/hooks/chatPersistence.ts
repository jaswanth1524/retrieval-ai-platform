/** Saving and restoring chat history.
 *
 * localStorage holds a compact copy (instant first paint, other tabs' `storage` events);
 * IndexedDB (chatStore.ts) holds the full one. This module is the format both share:
 * versioned payloads, sanitizing what is read back, compacting what is written, and
 * merging two copies of the same history.
 */
import type { PassageNarrowing } from '../api/types';
import type { ChatTurn } from '../components/ChatMessage';
import {
  type ChatStorageV2,
  type Conversation,
  type ConversationMatch,
  type ConversationScope,
  byRecentUse,
} from './chat/model';
import { emptyStorage } from './chat/persist';
export {
  EMPTY_SCOPE,
  isEmptyScope,
  TOMBSTONE_MAX_AGE_MS,
  conversationWorkspace,
  CHAT_STORAGE_KEY,
  CHAT_STORAGE_VERSION,
  CHAT_STORAGE_MAX_TURNS,
  CHAT_STORAGE_FALLBACK_TURNS,
  MAX_CONVERSATIONS,
  LOCAL_STORAGE_MAX_CONVERSATIONS,
  FULL_TEXT_CONVERSATIONS,
  MAX_PINNED_CONVERSATIONS,
  MAX_SAVED_CONVERSATIONS,
  TITLE_MAX_CHARS,
  PERSISTED_SOURCE_TEXT_TURNS,
  lastEdit,
  deriveTitle,
  emptyConversation,
  byRecentUse,
  capConversations,
  cleanLabels,
  MAX_LABELS,
  LABEL_MAX_CHARS,
} from './chat/model';
export type {
  ConversationScope,
  Conversation,
  DeletedConversation,
  ConversationSummary,
  ConversationMatch,
  ChatStorageV2,
} from './chat/model';
export {
  isChatTurn,
  validTime,
  cleanFollowups,
  sanitizeTask,
  sanitizeNarrowing,
  coverageWorthShowing,
  sanitizeTurn,
  parseDeleted,
  parseConversations,
  sanitizeConversation,
  sanitizeScope,
} from './chat/sanitize';
export {
  compactForStorage,
  conversationsForLocalStorage,
  forChatStore,
  isOnlyEmpty,
  persistStorage,
  emptyStorage,
  reportUnreadableStorage,
  loadStorage,
} from './chat/persist';
export type { PersistOutcome } from './chat/persist';
export { keepPassageText, keepOlderTurns, mergeConversations } from './chat/merge';

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
/** ``storage`` without conversation ``id``, with a tombstone at ``at`` so another tab's
 *  copy of it doesn't bring it back on its next save. */
/** ``storage`` with turn ``turnId`` replaced by ``change(turn)``: in conversation
 *  ``conversationId``, or in every conversation holding it when that is undefined (an
 *  edited question's copy shares the earlier turns' ids). Each conversation changed is
 *  stamped edited at ``now``, so other tabs keep the change. */
export function updateTurn(
  storage: ChatStorageV2,
  conversationId: string | undefined,
  turnId: string,
  change: (turn: ChatTurn) => ChatTurn,
  now: number,
): ChatStorageV2 {
  return {
    ...storage,
    conversations: storage.conversations.map((conversation) =>
      (conversationId === undefined || conversation.id === conversationId) &&
      conversation.turns.some((turn) => turn.id === turnId)
        ? {
            ...conversation,
            turns: conversation.turns.map((turn) => (turn.id === turnId ? change(turn) : turn)),
            editedAt: now,
          }
        : conversation,
    ),
  };
}
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
/** ``conversation`` (just deleted) put back, its tombstone taken away: an Undo. */
export function withConversationRestored(
  storage: ChatStorageV2,
  conversation: Conversation,
): ChatStorageV2 {
  const deleted = (storage.deleted ?? []).filter((entry) => entry.id !== conversation.id);
  const others = storage.conversations.filter((candidate) => candidate.id !== conversation.id);
  // Re-stamped as edited now: a merge with another tab must take it over the
  // tombstone that tab may already have.
  const restored = { ...conversation, editedAt: Date.now() };
  return {
    ...storage,
    conversations: [...others, restored],
    activeConversationId: conversation.id,
    deleted,
  };
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
/** The scope as question narrowing: its page range — only while exactly one document is
 *  scoped (a range across several documents means nothing), and in order whichever end
 *  was typed first — and its upload date. */
export function scopeNarrowing(scope: ConversationScope): PassageNarrowing | undefined {
  const narrowing: PassageNarrowing = {};
  if (scope.filenames.length === 1) {
    let { pageMin: min, pageMax: max } = scope;
    if (min !== undefined && max !== undefined && min > max) [min, max] = [max, min];
    if (min !== undefined) narrowing.page_min = min;
    if (max !== undefined) narrowing.page_max = max;
  }
  // In order, whichever end was set first (the server refuses after > before).
  let after = scope.uploadedAfter;
  let before = scope.uploadedBefore;
  if (after !== undefined && before !== undefined && after > before) [after, before] = [before, after];
  if (after !== undefined) narrowing.uploaded_after = after;
  if (before !== undefined) narrowing.uploaded_before = before;
  return Object.keys(narrowing).length > 0 ? narrowing : undefined;
}
