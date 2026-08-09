import { useEffect, useState } from 'react';
import type { KeyboardEvent } from 'react';
import type { ConversationSummary } from '../hooks/useChat';
import { formatRelativeTimeAgo } from '../utils/relativeTime';
import './ConversationList.css';

interface ConversationListProps {
  conversations: ConversationSummary[];
  activeId: string | null;
  onSwitch: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
  disabled?: boolean;
}

function ConversationList({
  conversations,
  activeId,
  onSwitch,
  onRename,
  onDelete,
  disabled,
}: ConversationListProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState('');
  const [confirmingDeleteId, setConfirmingDeleteId] = useState<string | null>(null);

  // A confirm dropped mid-flow (switched conversation, or the row it belonged to
  // vanished) should never linger — deliberately not keyed on `conversations` itself,
  // since that array's identity changes on every appended turn.
  useEffect(() => {
    setConfirmingDeleteId(null);
  }, [activeId]);

  useEffect(() => {
    if (confirmingDeleteId && !conversations.some((c) => c.id === confirmingDeleteId)) {
      setConfirmingDeleteId(null);
    }
  }, [conversations, confirmingDeleteId]);

  const startEditing = (conversation: ConversationSummary) => {
    setEditingId(conversation.id);
    setDraftTitle(conversation.title);
  };

  const commitEditing = () => {
    if (editingId && draftTitle.trim()) onRename(editingId, draftTitle);
    setEditingId(null);
    setDraftTitle('');
  };

  const handleConfirmKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') setConfirmingDeleteId(null);
  };

  return (
    <>
      {conversations.map((conversation) => (
        <div key={conversation.id} className="conversation-list__row-wrap" data-testid="conversation-item">
          {confirmingDeleteId === conversation.id ? (
            <div className="conversation-list__confirm" onKeyDown={handleConfirmKeyDown}>
              <span className="conversation-list__confirm-text">Delete this conversation?</span>
              <div className="conversation-list__confirm-actions">
                <button
                  type="button"
                  className="conversation-list__confirm-cancel"
                  autoFocus
                  onClick={() => setConfirmingDeleteId(null)}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="conversation-list__confirm-delete"
                  onClick={() => {
                    onDelete(conversation.id);
                    setConfirmingDeleteId(null);
                  }}
                  data-testid="conversation-delete-confirm"
                >
                  Delete
                </button>
              </div>
            </div>
          ) : editingId === conversation.id ? (
            <input
              className="conversation-list__edit"
              value={draftTitle}
              autoFocus
              onChange={(event) => setDraftTitle(event.target.value)}
              onBlur={commitEditing}
              onKeyDown={(event) => {
                if (event.key === 'Enter') commitEditing();
                if (event.key === 'Escape') {
                  setEditingId(null);
                  setDraftTitle('');
                }
              }}
              aria-label={`Rename ${conversation.title}`}
            />
          ) : (
            <div
              className={`conversation-list__row${
                conversation.id === activeId ? ' conversation-list__row--active' : ''
              }`}
            >
              <button
                type="button"
                className="conversation-list__select"
                onClick={() => onSwitch(conversation.id)}
                onDoubleClick={() => startEditing(conversation)}
                disabled={disabled}
                title={conversation.title}
                data-testid="conversation-select"
              >
                <span className="conversation-list__title">{conversation.title}</span>
                <span className="conversation-list__meta mono">
                  {conversation.turnCount} {conversation.turnCount === 1 ? 'turn' : 'turns'} &middot;{' '}
                  {formatRelativeTimeAgo(conversation.updatedAt)}
                </span>
              </button>
              <span className="conversation-list__row-actions">
                <button
                  type="button"
                  className="conversation-list__row-action"
                  onClick={() => startEditing(conversation)}
                  aria-label={`Rename ${conversation.title}`}
                >
                  ✎
                </button>
                <button
                  type="button"
                  className="conversation-list__row-action"
                  onClick={() => setConfirmingDeleteId(conversation.id)}
                  disabled={disabled}
                  aria-label={`Delete ${conversation.title}`}
                >
                  ✕
                </button>
              </span>
            </div>
          )}
        </div>
      ))}
    </>
  );
}

export default ConversationList;
