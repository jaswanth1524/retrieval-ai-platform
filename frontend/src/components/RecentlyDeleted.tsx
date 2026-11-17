import { memo, useEffect, useState } from 'react';
import type { Conversation } from '../hooks/chat/model';
import { formatRelativeTime } from '../utils/relativeTime';
import {
  readTrash,
  removeFromTrash,
  TRASH_STORAGE_KEY,
  type TrashedConversation,
} from '../utils/trash';
import ConfirmInline from './ConfirmInline';

/** Conversations deleted in this browser in the last 30 days (utils/trash): each can be
 *  put back, or deleted for good. Read when opened, when another tab changes the trash,
 *  and when ``version`` changes (a delete here). One that is in the history again
 *  (``liveIds``: restored in another tab, or edited there after the delete) isn't
 *  offered: restoring the older copy would replace the newer one. */
function RecentlyDeleted({
  version,
  liveIds,
  onRestore,
}: {
  version: number;
  liveIds: ReadonlySet<string>;
  onRestore: (conversation: Conversation) => void;
}) {
  const [open, setOpen] = useState(false);
  const [stored, setStored] = useState<TrashedConversation[]>(() => readTrash());
  const [purging, setPurging] = useState<string | null>(null);
  // Re-read when a delete happened elsewhere in the app (the list's own changes are
  // applied here directly).
  const [seenVersion, setSeenVersion] = useState(version);
  if (version !== seenVersion) {
    setSeenVersion(version);
    setStored(readTrash());
  }
  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.key === TRASH_STORAGE_KEY) setStored(readTrash());
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);
  const entries = stored.filter((entry) => !liveIds.has(entry.conversation.id));
  if (entries.length === 0) return null;
  return (
    <details
      className="conversation-list__trash"
      open={open}
      onToggle={(event) => {
        const nowOpen = (event.currentTarget as HTMLDetailsElement).open;
        setOpen(nowOpen);
        if (nowOpen) setStored(readTrash());
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
            {purging === conversation.id ? (
              <ConfirmInline
                prompt="Delete for good?"
                confirmLabel="Delete"
                onCancel={() => setPurging(null)}
                onConfirm={() => {
                  removeFromTrash(conversation.id);
                  setPurging(null);
                  setStored(readTrash());
                }}
                confirmTestId="recently-deleted-purge-confirm"
              />
            ) : (
              <>
                <button
                  type="button"
                  onClick={() => {
                    removeFromTrash(conversation.id);
                    setStored(readTrash());
                    onRestore(conversation);
                  }}
                  aria-label={`Restore “${conversation.title}”`}
                  data-testid="recently-deleted-restore"
                >
                  Restore
                </button>{' '}
                <button
                  type="button"
                  onClick={() => setPurging(conversation.id)}
                  aria-label={`Delete “${conversation.title}” for good`}
                  data-testid="recently-deleted-purge"
                >
                  Delete now
                </button>
              </>
            )}
          </li>
        ))}
      </ul>
    </details>
  );
}

export default memo(RecentlyDeleted);
