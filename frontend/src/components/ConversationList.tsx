import { useState } from 'react';
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
                    onClick={() => onDelete(conversation.id)}
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
