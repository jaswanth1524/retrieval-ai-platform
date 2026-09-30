import { useEffect, useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';
import type { ConversationSummary } from '../hooks/useChat';
import { formatRelativeTimeAgo } from '../utils/relativeTime';
import './ConversationList.css';

// Below this many conversations a search box is clutter; above it, scrolling is slower.
const SEARCH_THRESHOLD = 5;

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
  const [query, setQuery] = useState('');
  const needle = query.trim().toLowerCase();
  // The active conversation always stays listed, so the list never hides where you are.
  const visible = needle
    ? conversations.filter(
        (conversation) =>
          conversation.id === activeId || conversation.title.toLowerCase().includes(needle),
      )
    : conversations;
  const listRef = useRef<HTMLUListElement>(null);
  // Row index to put focus back on once a confirmed delete has re-rendered the list;
  // the focused Delete button unmounts with its row, dropping focus to <body>.
  const refocusIndexRef = useRef<number | null>(null);

  useEffect(() => {
    const index = refocusIndexRef.current;
    if (index === null) return;
    refocusIndexRef.current = null;
    const rows = listRef.current?.querySelectorAll<HTMLElement>('.conversation-list__select');
    if (rows && rows.length > 0) rows[Math.min(index, rows.length - 1)].focus();
  }, [conversations]);

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
    if (event.key !== 'Escape') return;
    // Cancels the confirm only — not the narrow-screen drawer the list may sit in.
    event.stopPropagation();
    setConfirmingDeleteId(null);
  };

  return (
    <>
      {conversations.length > SEARCH_THRESHOLD && (
        <input
          type="search"
          className="conversation-list__search"
          placeholder="Search conversations…"
          aria-label="Search conversations"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          data-testid="conversation-search"
        />
      )}
      <ul ref={listRef} className="conversation-list" aria-label="Conversations">
        {visible.map((conversation, index) => (
          <li key={conversation.id} className="conversation-list__row-wrap" data-testid="conversation-item">
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
                      refocusIndexRef.current = index;
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
                    event.stopPropagation();
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
                  aria-current={conversation.id === activeId ? 'true' : undefined}
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
                    disabled={disabled}
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
          </li>
        ))}
      </ul>
    </>
  );
}

export default ConversationList;
