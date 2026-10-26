import { memo, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';
import { MAX_PINNED_CONVERSATIONS } from '../hooks/chatPersistence';
import type { ConversationMatch, ConversationSummary } from '../hooks/useChat';
import { formatRelativeTimeAgo } from '../utils/relativeTime';
import { ALL_WORKSPACES, type ListFilter, dateGroup, listedConversations } from '../utils/conversationList';
import ConfirmInline from './ConfirmInline';
import './ConversationList.css';
import { isImeComposing } from '../utils/keyboard';

// Below this many conversations a search box is clutter; above it, scrolling is slower.
const SEARCH_THRESHOLD = 5;

interface ConversationListProps {
  conversations: ConversationSummary[];
  activeId: string | null;
  /** `turnId`: a search matched inside this turn; the thread scrolls to it. */
  onSwitch: (id: string, turnId?: string) => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
  onPin?: (id: string, pinned: boolean) => void;
  /** Matches the query against question and answer text (titles are matched here). */
  onSearch?: (query: string) => Map<string, ConversationMatch>;
  disabled?: boolean;
  /** The conversation an answer is streaming into: it can't be deleted until it ends. */
  streamingId?: string | null;
  /** Told the ids in the order shown, whenever it changes (for ] and [). */
  onVisibleOrder?: (ids: string[]) => void;
}

function ConversationList({
  conversations,
  activeId,
  onSwitch,
  onRename,
  onDelete,
  onPin,
  onSearch,
  disabled,
  streamingId = null,
  onVisibleOrder,
}: ConversationListProps) {
  const [filter, setFilter] = useState<ListFilter>('all');
  const [workspaceFilter, setWorkspaceFilter] = useState(ALL_WORKSPACES);
  const workspaces = useMemo(
    () => [...new Set(conversations.map((conversation) => conversation.workspace ?? ''))].sort(),
    [conversations],
  );
  const anySaved = conversations.some((conversation) => (conversation.savedCount ?? 0) > 0);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState('');
  const [confirmingDeleteId, setConfirmingDeleteId] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  // Deferred: matching every answer's text runs on each keystroke otherwise.
  const needle = useDeferredValue(query.trim().toLowerCase());
  // Re-run when the list changes (a new answer may now match), not on every frame:
  // onSearch reads the latest history itself, and `conversations` says when it moved.
  const matches = useMemo(
    () => (needle && onSearch ? onSearch(needle) : new Map<string, ConversationMatch>()),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [needle, onSearch, conversations],
  );
  // Pinned first, each group most recently used first (the order they arrive in).
  const ordered = useMemo(
    () => listedConversations(conversations, filter, workspaceFilter),
    [conversations, filter, workspaceFilter],
  );
  // The active conversation always stays listed, so the list never hides where you are.
  const visible = needle
    ? ordered.filter(
        (conversation) =>
          conversation.id === activeId ||
          conversation.title.toLowerCase().includes(needle) ||
          matches.has(conversation.id),
      )
    : ordered;
  const visibleKey = visible.map((conversation) => conversation.id).join(',');
  useEffect(() => {
    onVisibleOrder?.(visibleKey ? visibleKey.split(',') : []);
  }, [visibleKey, onVisibleOrder]);
  // A filter that hides everything it could show resets itself.
  useEffect(() => {
    if (filter === 'saved' && !anySaved) setFilter('all');
  }, [filter, anySaved]);
  useEffect(() => {
    if (workspaceFilter !== ALL_WORKSPACES && !workspaces.includes(workspaceFilter)) {
      setWorkspaceFilter(ALL_WORKSPACES);
    }
  }, [workspaceFilter, workspaces]);
  // Date headings only for the plain list: a search or filter is its own grouping.
  const grouped = !needle && filter === 'all';
  const pinLimitReached =
    conversations.filter((conversation) => conversation.pinned).length >= MAX_PINNED_CONVERSATIONS;
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

  // Conversation whose row should take focus back once its rename field closes from the
  // keyboard: the field unmounts, which otherwise dropped focus to <body>. Not on blur,
  // which means focus already went somewhere the user chose.
  const refocusRowIdRef = useRef<string | null>(null);
  useEffect(() => {
    const id = refocusRowIdRef.current;
    if (editingId !== null || id === null) return;
    refocusRowIdRef.current = null;
    const row = [...(listRef.current?.querySelectorAll<HTMLElement>('[data-conversation-id]') ?? [])].find(
      (candidate) => candidate.dataset.conversationId === id,
    );
    row?.querySelector<HTMLElement>('.conversation-list__select')?.focus();
  }, [editingId]);

  const commitEditing = () => {
    if (editingId && draftTitle.trim()) onRename(editingId, draftTitle);
    setEditingId(null);
    setDraftTitle('');
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
      {(anySaved || workspaces.length > 1) && (
        <div className="conversation-list__filters" data-testid="conversation-filters">
          {anySaved && (
            <span role="group" aria-label="Show">
              <button
                type="button"
                className="conversation-list__filter"
                aria-pressed={filter === 'all'}
                onClick={() => setFilter('all')}
              >
                All
              </button>
              <button
                type="button"
                className="conversation-list__filter"
                aria-pressed={filter === 'saved'}
                onClick={() => setFilter('saved')}
                data-testid="conversation-filter-saved"
              >
                ★ Saved
              </button>
            </span>
          )}
          {workspaces.length > 1 && (
            <select
              className="conversation-list__workspace-filter"
              value={workspaceFilter}
              onChange={(event) => setWorkspaceFilter(event.target.value)}
              aria-label="Workspace"
              data-testid="conversation-filter-workspace"
            >
              <option value={ALL_WORKSPACES}>All workspaces</option>
              {workspaces.map((name) => (
                <option key={name || 'default'} value={name}>
                  {name || 'default'}
                </option>
              ))}
            </select>
          )}
        </div>
      )}
      <ul ref={listRef} className="conversation-list" aria-label="Conversations">
        {visible.map((conversation, index) => {
          const group = conversation.pinned ? 'Pinned' : dateGroup(conversation.updatedAt);
          const previous = visible[index - 1];
          const previousGroup = previous
            ? previous.pinned
              ? 'Pinned'
              : dateGroup(previous.updatedAt)
            : null;
          const heading = grouped && group !== previousGroup ? group : null;
          return (
          <li
            key={conversation.id}
            className={`conversation-list__row-wrap${
              conversation.pinned && !visible[index + 1]?.pinned && index < visible.length - 1
                ? ' conversation-list__row-wrap--last-pinned'
                : ''
            }`}
            data-testid="conversation-item"
            data-conversation-id={conversation.id}
            data-pinned={conversation.pinned ? 'true' : undefined}
          >
            {heading && (
              <span className="conversation-list__group" data-testid="conversation-group">
                {heading}
              </span>
            )}
            {confirmingDeleteId === conversation.id ? (
              <ConfirmInline
                className="conversation-list__confirm"
                prompt="Delete this conversation?"
                confirmLabel="Delete"
                onCancel={() => setConfirmingDeleteId(null)}
                onConfirm={() => {
                  refocusIndexRef.current = index;
                  onDelete(conversation.id);
                  setConfirmingDeleteId(null);
                }}
                confirmTestId="conversation-delete-confirm"
              />
            ) : editingId === conversation.id ? (
              <input
                className="conversation-list__edit"
                value={draftTitle}
                autoFocus
                onChange={(event) => setDraftTitle(event.target.value)}
                onBlur={commitEditing}
                onKeyDown={(event) => {
                  if (isImeComposing(event)) return;
                  if (event.key === 'Enter') {
                    refocusRowIdRef.current = conversation.id;
                    commitEditing();
                  }
                  if (event.key === 'Escape') {
                    event.stopPropagation();
                    refocusRowIdRef.current = conversation.id;
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
                  onClick={() => {
                    const match = matches.get(conversation.id);
                    if (match) onSwitch(conversation.id, match.turnId);
                    else if (filter === 'saved' && conversation.firstSavedTurnId) {
                      onSwitch(conversation.id, conversation.firstSavedTurnId);
                    } else onSwitch(conversation.id);
                  }}
                  onDoubleClick={() => startEditing(conversation)}
                  disabled={disabled}
                  title={conversation.title}
                  aria-current={conversation.id === activeId ? 'true' : undefined}
                  data-testid="conversation-select"
                >
                  <span className="conversation-list__title" dir="auto">
                    {conversation.pinned && (
                      <>
                        {/* aria-label on a plain span is ignored: screen readers said
                            "black star". The words are in hidden text instead. */}
                        <span className="conversation-list__pin-mark" aria-hidden="true">
                          ★{' '}
                        </span>
                        <span className="visually-hidden">Pinned: </span>
                      </>
                    )}
                    {conversation.title}
                  </span>
                  {matches.has(conversation.id) && (
                    <span className="conversation-list__snippet" data-testid="conversation-snippet">
                      {matches.get(conversation.id)?.snippet}
                    </span>
                  )}
                  <span className="conversation-list__meta mono">
                    {conversation.turnCount} {conversation.turnCount === 1 ? 'turn' : 'turns'} &middot;{' '}
                    {formatRelativeTimeAgo(conversation.updatedAt)}
                    {conversation.workspace && (
                      <span data-testid="conversation-workspace">
                        {' '}
                        &middot; {conversation.workspace}
                      </span>
                    )}
                    {(conversation.savedCount ?? 0) > 0 && (
                      <span data-testid="conversation-saved-count">
                        {' '}
                        &middot; ★ {conversation.savedCount}
                      </span>
                    )}
                  </span>
                </button>
                <span className="conversation-list__row-actions">
                  {onPin && (
                    <button
                      type="button"
                      className="conversation-list__row-action"
                      onClick={() => onPin(conversation.id, !conversation.pinned)}
                      disabled={!conversation.pinned && pinLimitReached}
                      title={
                        !conversation.pinned && pinLimitReached
                          ? `At most ${MAX_PINNED_CONVERSATIONS} pinned conversations`
                          : undefined
                      }
                      aria-label={`${conversation.pinned ? 'Unpin' : 'Pin'} ${conversation.title}`}
                      aria-pressed={conversation.pinned ? 'true' : 'false'}
                      data-testid="conversation-pin"
                    >
                      {conversation.pinned ? '★' : '☆'}
                    </button>
                  )}
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
                    disabled={disabled || conversation.id === streamingId}
                    title={
                      conversation.id === streamingId ? 'An answer is still being written here' : undefined
                    }
                    aria-label={`Delete ${conversation.title}`}
                  >
                    ✕
                  </button>
                </span>
              </div>
            )}
          </li>
          );
        })}
      </ul>
    </>
  );
}

// Memoized: App re-renders on every streamed token, and the list doesn't change then.
export default memo(ConversationList);
