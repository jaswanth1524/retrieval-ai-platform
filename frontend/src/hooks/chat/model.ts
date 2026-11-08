/** The chat history format: conversations, their scope, the storage limits, and the
 *  order they are kept and capped in. */
import type { ChatTurn } from '../../components/ChatMessage';
import { newId } from '../../utils/id';

// Far past any upload; a bound so a damaged value can't reach the server as garbage.
export const MAX_TIMESTAMP_SECONDS = 32_503_680_000; // year 3000
/** Which documents a conversation's questions search. Empty lists mean "everything". */
export interface ConversationScope {
  filenames: string[];
  tags: string[];
  /** A page range within the one scoped document (used only while exactly one is). */
  pageMin?: number;
  pageMax?: number;
  /** Only documents uploaded since this time (Unix seconds). */
  uploadedAfter?: number;
  /** Only documents uploaded up to this time (Unix seconds; the end of a chosen day). */
  uploadedBefore?: number;
}
export const EMPTY_SCOPE: ConversationScope = { filenames: [], tags: [] };
/** A scope that narrows nothing: no files, tags, pages or dates. */
export function isEmptyScope(scope: ConversationScope): boolean {
  return (
    scope.filenames.length === 0 &&
    scope.tags.length === 0 &&
    scope.pageMin === undefined &&
    scope.pageMax === undefined &&
    scope.uploadedAfter === undefined &&
    scope.uploadedBefore === undefined
  );
}
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
  // The reader's own labels ("hr", "audit 2026"), to filter the list by. An edit.
  labels?: string[];
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
  labels?: string[];
  // Only when not the default workspace.
  workspace?: string;
  // Answers saved in it, and the first of them (to scroll to).
  savedCount?: number;
  firstSavedTurnId?: string;
}
/** The workspace a conversation belongs to ('' = the default), or null for one with no
 *  questions yet, which belongs to whichever workspace asks first. */
/** Labels a conversation may carry, and how long each. */
export const MAX_LABELS = 5;
export const LABEL_MAX_CHARS = 24;

/** ``raw`` as labels: trimmed, spaces collapsed, case-insensitively distinct, at most
 *  MAX_LABELS of at most LABEL_MAX_CHARS. */
export function cleanLabels(raw: unknown): string[] {
  if (!Array.isArray(raw)) return [];
  const seen = new Set<string>();
  const labels: string[] = [];
  for (const value of raw) {
    if (typeof value !== 'string') continue;
    const label = value.replace(/\s+/g, ' ').trim().slice(0, LABEL_MAX_CHARS).trim();
    const key = label.toLocaleLowerCase();
    if (!label || seen.has(key)) continue;
    seen.add(key);
    labels.push(label);
    if (labels.length === MAX_LABELS) break;
  }
  return labels;
}

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
/** Conversations with a saved answer kept past MAX_CONVERSATIONS (beyond pinned ones);
 *  past this many, the least recently used of them can go like any other. */
export const MAX_SAVED_CONVERSATIONS = 20;
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
  const protectedIds = new Set([
    ...sorted
      .filter((conversation) => conversation.pinned)
      .slice(0, MAX_PINNED_CONVERSATIONS)
      .map((conversation) => conversation.id),
    // Saved answers are kept like pinned conversations, the most recent first.
    ...sorted
      .filter((conversation) => conversation.turns.some((turn) => turn.saved))
      .slice(0, MAX_SAVED_CONVERSATIONS)
      .map((conversation) => conversation.id),
  ]);
  if (keepId !== null) protectedIds.add(keepId);
  let room = MAX_CONVERSATIONS - protectedIds.size;
  return sorted.filter((conversation) => {
    if (protectedIds.has(conversation.id)) return true;
    room -= 1;
    return room >= 0;
  });
}
