import { useEffect, useState } from 'react';
import type { ConversationSummary } from '../hooks/useChat';
import './ConversationList.css';

interface ConversationListProps {
  conversations: ConversationSummary[];
  activeId: string | null;
  onNew: () => void;
  onSwitch: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
  disabled?: boolean;
}

function ConversationList({
  conversations,
  activeId,
  onNew,
  onSwitch,
  onRename,
  onDelete,
  disabled,
}: ConversationListProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState('');
  const [confirmingDeleteId, setConfirmingDeleteId] = useState<string | null>(null);

  // Nothing else clears the pending confirmation, so without this a row left armed keeps
  // its "Delete?" prompt — and stays unselectable, since the confirm block replaces the
  // row's own switch button — until the user comes back and answers it. Keyed on the
  // active conversation and on the row still existing, deliberately *not* on the
  // `conversations` array: that identity changes on every appended turn and every rename,
  // which would yank the prompt away mid-decision during unrelated activity.
  const confirmingStillListed = conversations.some(
    (conversation) => conversation.id === confirmingDeleteId,
  );
  useEffect(() => {
    if (confirmingDeleteId !== null && !confirmingStillListed) setConfirmingDeleteId(null);
  }, [confirmingDeleteId, confirmingStillListed]);
  useEffect(() => {
    setConfirmingDeleteId(null);
  }, [activeId]);

  const startEditing = (conversation: ConversationSummary) => {
    setEditingId(conversation.id);
    setDraftTitle(conversation.title);
  };

  const commitEditing = () => {
    if (editingId && draftTitle.trim()) onRename(editingId, draftTitle);
    setEditingId(null);
    setDraftTitle('');
  };

  return (
    <section className="conversation-list" aria-label="Conversations">
      <div className="conversation-list__header">
        <span className="conversation-list__title">Chats</span>
        <button
          type="button"
          className="conversation-list__new"
          onClick={onNew}
          disabled={disabled}
          data-testid="conversation-new"
        >
          + New chat
        </button>
      </div>
      <ul className="conversation-list__items">
        {conversations.map((conversation) => (
          <li
            key={conversation.id}
            className={`conversation-list__item${
              conversation.id === activeId ? ' conversation-list__item--active' : ''
            }`}
            data-testid="conversation-item"
          >
            {editingId === conversation.id ? (
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
            ) : confirmingDeleteId === conversation.id ? (
              // Escape dismisses, matching the rename input's own handler — otherwise the
              // only way out is a mouse click on Cancel.
              <div
                className="conversation-list__confirm"
                onKeyDown={(event) => {
                  if (event.key === 'Escape') setConfirmingDeleteId(null);
                }}
              >
                <span className="conversation-list__confirm-text">Delete?</span>
                <button
                  type="button"
                  className="conversation-list__confirm-yes"
                  onClick={() => {
                    onDelete(conversation.id);
                    setConfirmingDeleteId(null);
                  }}
                  disabled={disabled}
                  aria-label={`Confirm delete ${conversation.title}`}
                >
                  Confirm
                </button>
                <button
                  type="button"
                  className="conversation-list__confirm-no"
                  onClick={() => setConfirmingDeleteId(null)}
                  disabled={disabled}
                  aria-label={`Cancel delete ${conversation.title}`}
                  // Clicking Delete unmounts the button that had focus, dropping it to
                  // <body> — where the Escape handler above would never see a keypress.
                  // Cancel takes it rather than Confirm so a stray Enter dismisses the
                  // prompt instead of destroying the conversation.
                  autoFocus
                >
                  Cancel
                </button>
              </div>
            ) : (
              <>
                <button
                  type="button"
                  className="conversation-list__select"
                  onClick={() => onSwitch(conversation.id)}
                  disabled={disabled}
                  title={conversation.title}
                >
                  {conversation.title}
                </button>
                <span className="conversation-list__actions">
                  <button
                    type="button"
                    className="conversation-list__action"
                    onClick={() => startEditing(conversation)}
                    aria-label={`Rename ${conversation.title}`}
                  >
                    ✎
                  </button>
                  <button
                    type="button"
                    className="conversation-list__action"
                    onClick={() => setConfirmingDeleteId(conversation.id)}
                    disabled={disabled}
                    aria-label={`Delete ${conversation.title}`}
                  >
                    ✕
                  </button>
                </span>
              </>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}

export default ConversationList;
