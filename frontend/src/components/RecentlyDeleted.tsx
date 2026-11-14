import { memo, useState } from 'react';
import type { Conversation } from '../hooks/chat/model';
import { formatRelativeTime } from '../utils/relativeTime';
import { readTrash, removeFromTrash, type TrashedConversation } from '../utils/trash';

/** Conversations deleted in this browser in the last 30 days (utils/trash): each can be
 *  put back, or deleted for good. Read when opened; ``version`` changes after a delete. */
function RecentlyDeleted({
  version,
  onRestore,
}: {
  version: number;
  onRestore: (conversation: Conversation) => void;
}) {
  const [open, setOpen] = useState(false);
  const [entries, setEntries] = useState<TrashedConversation[]>(() => readTrash());
  // Re-read when a delete happened elsewhere in the app (the list's own changes are
  // applied here directly).
  const [seenVersion, setSeenVersion] = useState(version);
  if (version !== seenVersion) {
    setSeenVersion(version);
    setEntries(readTrash());
  }
  if (entries.length === 0) return null;
  return (
    <details
      className="conversation-list__trash"
      open={open}
      onToggle={(event) => {
        const nowOpen = (event.currentTarget as HTMLDetailsElement).open;
        setOpen(nowOpen);
        if (nowOpen) setEntries(readTrash());
      }}
      data-testid="recently-deleted"
    >
      <summary>Recently deleted ({entries.length})</summary>
      <ul className="conversation-list__trash-list">
        {entries.map(({ conversation, deletedAt }) => (
          <li key={conversation.id} data-testid="recently-deleted-item">
            <span className="conversation-list__trash-title" dir="auto">
              {conversation.title}
            </span>{' '}
            <span className="mono">{formatRelativeTime(deletedAt)}</span>{' '}
            <button
              type="button"
              onClick={() => {
                removeFromTrash(conversation.id);
                setEntries(readTrash());
                onRestore(conversation);
              }}
              aria-label={`Restore “${conversation.title}”`}
              data-testid="recently-deleted-restore"
            >
              Restore
            </button>{' '}
            <button
              type="button"
              onClick={() => {
                removeFromTrash(conversation.id);
                setEntries(readTrash());
              }}
              aria-label={`Delete “${conversation.title}” for good`}
              data-testid="recently-deleted-purge"
            >
              Delete now
            </button>
          </li>
        ))}
      </ul>
    </details>
  );
}

export default memo(RecentlyDeleted);
