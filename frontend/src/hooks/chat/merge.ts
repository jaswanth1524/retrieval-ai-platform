/** Two copies of the same history (another tab, IndexedDB vs localStorage) merged. */
import {
  CHAT_STORAGE_FALLBACK_TURNS,
  CHAT_STORAGE_MAX_TURNS,
  type ChatStorageV2,
  type Conversation,
  type DeletedConversation,
  TOMBSTONE_MAX_AGE_MS,
  capConversations,
  lastEdit,
} from './model';
import { emptyStorage } from './persist';

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
