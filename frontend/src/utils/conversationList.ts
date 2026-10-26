import type { ConversationSummary } from '../hooks/chatPersistence';

export type ListFilter = 'all' | 'saved';
// The workspace filter's value for "every workspace".
export const ALL_WORKSPACES = '*';

/** Today / Yesterday / Earlier, by when a conversation was last used. */
export function dateGroup(updatedAt: number, now = Date.now()): string {
  const startOfToday = new Date(now);
  startOfToday.setHours(0, 0, 0, 0);
  const today = startOfToday.getTime();
  if (updatedAt >= today) return 'Today';
  if (updatedAt >= today - 86_400_000) return 'Yesterday';
  if (updatedAt >= today - 6 * 86_400_000) return 'This week';
  return 'Earlier';
}

/** The conversations the list shows, in its order: pinned first, then by recent use;
 *  only those with saved answers, or of one workspace, when filtered. */
export function listedConversations(
  conversations: readonly ConversationSummary[],
  filter: ListFilter,
  workspace: string,
): ConversationSummary[] {
  const shown = conversations.filter(
    (conversation) =>
      (filter === 'all' || (conversation.savedCount ?? 0) > 0) &&
      (workspace === ALL_WORKSPACES || (conversation.workspace ?? '') === workspace),
  );
  return [
    ...shown.filter((conversation) => conversation.pinned),
    ...shown.filter((conversation) => !conversation.pinned),
  ];
}
