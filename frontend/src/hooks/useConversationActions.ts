import { useCallback } from 'react';
import type { Dispatch, MutableRefObject, SetStateAction } from 'react';
import { getWorkspace } from '../api/client';
import type { ChatTurn } from '../components/ChatMessage';
import { newId } from '../utils/id';
import {
  EMPTY_SCOPE,
  MAX_CONVERSATIONS,
  MAX_PINNED_CONVERSATIONS,
  TITLE_MAX_CHARS,
  byRecentUse,
  capConversations,
  deriveTitle,
  emptyConversation,
  sanitizeTurn,
  validTime,
} from './chatPersistence';
import type { ChatStorageV2, Conversation } from './chatPersistence';

/** One conversation of an "Export all conversations" file. */
export interface ImportItem {
  raw: unknown;
  title?: string;
}

/** A conversation built from an export (a conversation object, or its turns alone), or
 *  null when it holds no valid turn. Fresh ids, so importing a file twice can't clash;
 *  ``keepDates`` keeps the export's own dates (else it is new, as of ``now``). An
 *  export carries no workspace: it joins the one it is imported into. */
function importedConversation(
  raw: unknown,
  title: string | undefined,
  now: number,
  keepDates: boolean,
): Conversation | null {
  const candidate = Array.isArray(raw) ? { turns: raw } : raw;
  if (!candidate || typeof candidate !== 'object') return null;
  const record = candidate as Partial<Conversation>;
  if (!Array.isArray(record.turns)) return null;
  const turns = record.turns
    .map(sanitizeTurn)
    .filter((turn): turn is ChatTurn => turn !== null)
    .map((turn) => ({ ...turn, id: newId() }));
  if (turns.length === 0) return null;
  const createdAt = keepDates ? Math.min(validTime(record.createdAt, now), now) : now;
  const updatedAt = keepDates
    ? Math.min(Math.max(validTime(record.updatedAt, createdAt), createdAt), now)
    : now;
  const conversation: Conversation = {
    id: newId(),
    title: (title ?? '').trim().slice(0, TITLE_MAX_CHARS) || deriveTitle(turns),
    createdAt,
    updatedAt,
    turns,
  };
  // Kept in its place by its own date, but changed here now: that puts it in the copy
  // other tabs read (conversationsForLocalStorage), so their next save keeps it.
  if (keepDates) conversation.editedAt = now;
  const workspace = getWorkspace();
  if (workspace) conversation.workspace = workspace;
  return conversation;
}

/** The conversation list's own actions (``useChat`` exposes them): import, start,
 *  switch, rename and pin. Each changes ``storage`` through ``setStorage`` and reads the
 *  latest through ``storageRef``, so every one keeps a stable identity. */
export function useConversationActions(
  setStorage: Dispatch<SetStateAction<ChatStorageV2>>,
  storageRef: MutableRefObject<ChatStorageV2>,
  streamingConversationIdRef: MutableRefObject<string | null>,
) {
  const importConversation = useCallback((raw: unknown, title?: string) => {
    // Fine mid-answer: the answer is written to the conversation it was asked in, by id.
    const conversation = importedConversation(raw, title, Date.now(), false);
    if (conversation === null) return false;
    setStorage((prev) => ({
      ...prev,
      activeConversationId: conversation.id,
      conversations: capConversations([conversation, ...prev.conversations], conversation.id),
    }));
    return true;
  }, [setStorage]);

  const importPreview = useCallback((items: readonly ImportItem[]) => {
    const now = Date.now();
    const valid = items.filter(
      (item) => importedConversation(item.raw, item.title, now, true) !== null,
    ).length;
    const room = Math.max(MAX_CONVERSATIONS - storageRef.current.conversations.length, 0);
    return { valid, room };
  }, [storageRef]);

  const importConversations = useCallback(
    (items: readonly ImportItem[], options?: { makeRoom?: boolean }) => {
      const now = Date.now();
      const built = items
        .map((item) => importedConversation(item.raw, item.title, now, true))
        .filter((conversation): conversation is Conversation => conversation !== null)
        .sort(byRecentUse);
      // One "Import all" of 100 conversations used to push every older conversation out
      // of a full history without a word (each import was stamped "now", and the cap
      // drops the least recently used). Now the room left is filled, and older ones go
      // only when the reader agreed to it (makeRoom).
      const current = storageRef.current;
      const room = Math.max(MAX_CONVERSATIONS - current.conversations.length, 0);
      const removable = options?.makeRoom
        ? [...current.conversations]
            .filter(
              (conversation) =>
                !conversation.pinned &&
                conversation.id !== current.activeConversationId &&
                conversation.id !== streamingConversationIdRef.current,
            )
            .sort(byRecentUse)
            .reverse()
            .slice(0, Math.max(built.length - room, 0))
        : [];
      const chosen = built.slice(0, room + removable.length);
      if (chosen.length > 0) {
        const gone = new Set(removable.map((conversation) => conversation.id));
        setStorage((prev) => ({
          ...prev,
          activeConversationId: chosen[0].id,
          conversations: [
            ...chosen,
            ...prev.conversations.filter((conversation) => !gone.has(conversation.id)),
          ].sort(byRecentUse),
          // Tombstoned, so another tab's copy doesn't bring them back.
          deleted: [...(prev.deleted ?? []), ...removable.map((c) => ({ id: c.id, at: now }))],
        }));
      }
      return {
        imported: chosen.length,
        skipped: built.length - chosen.length,
        removed: removable.length,
      };
    },
    [setStorage, storageRef, streamingConversationIdRef],
  );

  const newConversation = useCallback(() => {
    setStorage((prev) => {
      // Reuse an already-empty active conversation instead of stacking blanks — but as a
      // new chat: a document scope set on it ("Ask about these") went on to scope the
      // first question of what the user had just asked to be a fresh conversation.
      const active = prev.conversations.find((c) => c.id === prev.activeConversationId);
      if (active && active.turns.length === 0) {
        if (!active.scope?.filenames.length && !active.scope?.tags.length) return prev;
        return {
          ...prev,
          conversations: prev.conversations.map((c) =>
            c.id === active.id ? { ...c, scope: EMPTY_SCOPE, editedAt: Date.now() } : c,
          ),
        };
      }
      const conversation = emptyConversation();
      // Past the cap, the least recently *used* unpinned conversation goes — not the
      // oldest created, which dropped a conversation still in daily use.
      const conversations = capConversations([conversation, ...prev.conversations], conversation.id);
      return { ...prev, activeConversationId: conversation.id, conversations };
    });
  }, [setStorage]);

  const switchConversation = useCallback((id: string) => {
    setStorage((prev) =>
      prev.conversations.some((conversation) => conversation.id === id)
        ? { ...prev, activeConversationId: id }
        : prev,
    );
  }, [setStorage]);

  const renameConversation = useCallback((id: string, title: string) => {
    const trimmed = title.replace(/\s+/g, ' ').trim().slice(0, TITLE_MAX_CHARS);
    if (!trimmed) return;
    setStorage((prev) => ({
      ...prev,
      conversations: prev.conversations.map((conversation) =>
        conversation.id === id ? { ...conversation, title: trimmed, editedAt: Date.now() } : conversation,
      ),
    }));
  }, [setStorage]);

  const setPinned = useCallback((id: string, pinned: boolean) => {
    const current = storageRef.current.conversations;
    const target = current.find((conversation) => conversation.id === id);
    if (!target || Boolean(target.pinned) === pinned) return Boolean(target);
    if (pinned && current.filter((c) => c.pinned).length >= MAX_PINNED_CONVERSATIONS) return false;
    setStorage((prev) => ({
      ...prev,
      conversations: prev.conversations.map((conversation) =>
        conversation.id === id
          ? { ...conversation, pinned: pinned || undefined, editedAt: Date.now() }
          : conversation,
      ),
    }));
    return true;
  }, [setStorage, storageRef]);

  return {
    importConversation,
    importPreview,
    importConversations,
    newConversation,
    switchConversation,
    renameConversation,
    setPinned,
  };
}
