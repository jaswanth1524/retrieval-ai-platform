import { memo, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';
import { MAX_PINNED_CONVERSATIONS } from '../hooks/chatPersistence';
import { useCompactActions } from '../hooks/useCompactActions';
import OverflowMenu from './OverflowMenu';
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
  /** Replace a conversation's labels (to filter the list by). */
  onSetLabels?: (id: string, labels: string[]) => void;
  /** Rename a label on every conversation (null: remove it everywhere). */
  onRenameLabel?: (from: string, to: string | null) => void;
  /** Download the conversations filed under a label. */
  onExportLabel?: (label: string) => void;
  /** Matches the query against question and answer text (titles are matched here). */
  onSearch?: (query: string) => Map<string, ConversationMatch>;
  disabled?: boolean;
  /** The conversation an answer is streaming into: it can't be deleted until it ends. */
  streamingId?: string | null;
  /** Told the ids in the order shown, whenever it changes (for ] and [). */
  onVisibleOrder?: (ids: string[]) => void;
  /** Delete several at once (Select): one Undo for them all. */
  onDeleteMany?: (ids: string[]) => void;
}

function ConversationList({
  conversations,
  activeId,
  onSwitch,
  onRename,
  onDelete,
  onPin,
  onSetLabels,
  onRenameLabel,
  onExportLabel,
  onSearch,
  disabled,
  streamingId = null,
  onVisibleOrder,
  onDeleteMany,
}: ConversationListProps) {
  const compactActions = useCompactActions();
  const [filter, setFilter] = useState<ListFilter>('all');
  const [workspaceFilter, setWorkspaceFilter] = useState(ALL_WORKSPACES);
  const workspaces = useMemo(
    () => [...new Set(conversations.map((conversation) => conversation.workspace ?? ''))].sort(),
    [conversations],
  );
  const anySaved = conversations.some((conversation) => (conversation.savedCount ?? 0) > 0);
  // Select: several conversations picked to delete, pin or label together.
  const [selecting, setSelecting] = useState(false);
  const [picked, setPicked] = useState<ReadonlySet<string>>(() => new Set());
  const [bulkConfirm, setBulkConfirm] = useState(false);
  const [bulkLabel, setBulkLabel] = useState<string | null>(null);
  // A bulk pin that pinned only some, said.
  const [pinNote, setPinNote] = useState<string | null>(null);
  // Down to one conversation the bar (and its Done) goes: the checkboxes stayed, with
  // nothing to turn them off.
  useEffect(() => {
    if (selecting && conversations.length <= 1) {
      setSelecting(false);
      setPicked(new Set());
      setBulkConfirm(false);
      setBulkLabel(null);
    }
  }, [selecting, conversations.length]);
  const endSelecting = () => {
    setSelecting(false);
    setPicked(new Set());
    setBulkConfirm(false);
    setBulkLabel(null);
  };
  const togglePicked = (id: string) =>
    setPicked((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState('');
  const [confirmingDeleteId, setConfirmingDeleteId] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [labelFilter, setLabelFilter] = useState<string | null>(null);
  const [labelingId, setLabelingId] = useState<string | null>(null);
  const [labelDraft, setLabelDraft] = useState('');
  // The filtered-by label being renamed (its new name), or removed (confirming).
  const [renamingTo, setRenamingTo] = useState<string | null>(null);
  const [removingLabel, setRemovingLabel] = useState(false);
  const allLabels = useMemo(() => {
    const byKey = new Map<string, string>();
    for (const conversation of conversations) {
      for (const label of conversation.labels ?? []) {
        if (!byKey.has(label.toLocaleLowerCase())) byKey.set(label.toLocaleLowerCase(), label);
      }
    }
    return [...byKey.values()].sort((a, b) => a.localeCompare(b));
  }, [conversations]);
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
    () => listedConversations(conversations, filter, workspaceFilter, labelFilter),
    [conversations, filter, workspaceFilter, labelFilter],
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
  // What a bulk action acts on: the picked rows still listed. A search or filter typed
  // after picking used to leave hidden rows picked, and Delete removed them unseen.
  const chosen = visible.filter((conversation) => picked.has(conversation.id)).map((c) => c.id);
  const deletable = chosen.filter((id) => id !== streamingId);
  useEffect(() => {
    onVisibleOrder?.(visibleKey ? visibleKey.split(',') : []);
  }, [visibleKey, onVisibleOrder]);
  // A filter that hides everything it could show resets itself.
  useEffect(() => {
    if (filter === 'saved' && !anySaved) setFilter('all');
  }, [filter, anySaved]);
  useEffect(() => {
    if (
      labelFilter !== null &&
      !allLabels.some((label) => label.toLocaleLowerCase() === labelFilter.toLocaleLowerCase())
    ) {
      setLabelFilter(null);
    }
  }, [labelFilter, allLabels]);
  useEffect(() => {
    if (workspaceFilter !== ALL_WORKSPACES && !workspaces.includes(workspaceFilter)) {
      setWorkspaceFilter(ALL_WORKSPACES);
    }
  }, [workspaceFilter, workspaces]);
  // Picked ones deleted meanwhile (another tab, an import) drop out of the selection.
  useEffect(() => {
    setPicked((prev) => {
      const live = new Set(conversations.map((conversation) => conversation.id));
      const kept = [...prev].filter((id) => live.has(id));
      return kept.length === prev.size ? prev : new Set(kept);
    });
  }, [conversations]);
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
    if (editingId !== null || labelingId !== null || id === null) return;
    refocusRowIdRef.current = null;
    const row = [...(listRef.current?.querySelectorAll<HTMLElement>('[data-conversation-id]') ?? [])].find(
      (candidate) => candidate.dataset.conversationId === id,
    );
    row?.querySelector<HTMLElement>('.conversation-list__select')?.focus();
  }, [editingId, labelingId]);

  const commitEditing = () => {
    if (editingId && draftTitle.trim()) onRename(editingId, draftTitle);
    setEditingId(null);
    setDraftTitle('');
  };

  return (
    <>
      {/* Shown while a search is typed even below the threshold: hidden, a filter left
          after a delete couldn't be seen or cleared. */}
      {(conversations.length > SEARCH_THRESHOLD || query !== '') && (
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
      {allLabels.length > 0 && (
        <div className="conversation-list__labels" role="group" aria-label="Labels" data-testid="conversation-label-filters">
          {allLabels.map((label) => (
            <button
              key={label}
              type="button"
              className="conversation-list__label"
              aria-pressed={labelFilter === label}
              onClick={() => setLabelFilter((current) => (current === label ? null : label))}
              data-testid="conversation-label-filter"
            >
              #{label}
            </button>
          ))}
          <datalist id="conversation-label-names">
            {allLabels.map((label) => (
              <option key={label} value={label} />
            ))}
          </datalist>
        </div>
      )}
      {labelFilter !== null && (onRenameLabel || onExportLabel) && (
        <div className="conversation-list__label-tools" data-testid="conversation-label-tools">
          {renamingTo !== null ? (
            <input
              className="conversation-list__edit"
              value={renamingTo}
              autoFocus
              maxLength={24}
              aria-label={`New name for #${labelFilter}`}
              onChange={(event) => setRenamingTo(event.target.value)}
              onKeyDown={(event) => {
                if (isImeComposing(event)) return;
                if (event.key === 'Enter') {
                  const name = renamingTo.trim();
                  if (name) {
                    onRenameLabel?.(labelFilter, name);
                    setLabelFilter(name);
                  }
                  setRenamingTo(null);
                }
                if (event.key === 'Escape') {
                  event.stopPropagation();
                  setRenamingTo(null);
                }
              }}
              onBlur={() => setRenamingTo(null)}
              data-testid="conversation-label-rename-input"
            />
          ) : removingLabel ? (
            <ConfirmInline
              prompt={`Remove #${labelFilter} from every conversation?`}
              confirmLabel="Remove"
              onCancel={() => setRemovingLabel(false)}
              onConfirm={() => {
                onRenameLabel?.(labelFilter, null);
                setRemovingLabel(false);
                setLabelFilter(null);
              }}
              confirmTestId="conversation-label-remove-confirm"
            />
          ) : (
            <>
              {onRenameLabel && (
                <button
                  type="button"
                  className="conversation-list__filter"
                  onClick={() => setRenamingTo(labelFilter)}
                  disabled={disabled}
                  data-testid="conversation-label-rename"
                >
                  Rename #{labelFilter}
                </button>
              )}
              {onRenameLabel && (
                <button
                  type="button"
                  className="conversation-list__filter"
                  onClick={() => setRemovingLabel(true)}
                  disabled={disabled}
                  data-testid="conversation-label-remove"
                >
                  Remove
                </button>
              )}
              {onExportLabel && (
                <button
                  type="button"
                  className="conversation-list__filter"
                  onClick={() => onExportLabel(labelFilter)}
                  data-testid="conversation-label-export"
                >
                  Export these
                </button>
              )}
            </>
          )}
        </div>
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
      {(onDeleteMany || onPin || onSetLabels) && conversations.length > 1 && (
        <div className="conversation-list__bulk" role="group" aria-label="Select conversations" data-testid="conversation-bulk">
          {!selecting ? (
            <button
              type="button"
              className="conversation-list__label"
              onClick={() => {
                setPinNote(null);
                setSelecting(true);
              }}
              disabled={disabled}
              data-testid="conversation-select-mode"
            >
              Select
            </button>
          ) : bulkConfirm ? (
            <ConfirmInline
              className="conversation-list__confirm"
              prompt={`Delete ${deletable.length} conversation${deletable.length === 1 ? '' : 's'}?${
                deletable.length < chosen.length ? ' (the one being answered stays)' : ''
              }`}
              confirmLabel="Delete"
              onCancel={() => setBulkConfirm(false)}
              onConfirm={() => {
                onDeleteMany?.(deletable);
                endSelecting();
              }}
              confirmTestId="conversation-bulk-delete-confirm"
            />
          ) : bulkLabel !== null ? (
            <input
              className="conversation-list__edit"
              value={bulkLabel}
              autoFocus
              placeholder="label to add"
              list="conversation-label-names"
              onChange={(event) => setBulkLabel(event.target.value)}
              onKeyDown={(event) => {
                if (isImeComposing(event)) return;
                if (event.key === 'Enter' && bulkLabel.trim()) {
                  for (const id of chosen) {
                    const current = conversations.find((c) => c.id === id)?.labels ?? [];
                    onSetLabels?.(id, [...current, bulkLabel.trim()]);
                  }
                  endSelecting();
                }
                if (event.key === 'Escape') {
                  event.stopPropagation();
                  setBulkLabel(null);
                }
              }}
              aria-label="Label to add to the selected conversations"
              data-testid="conversation-bulk-label-input"
            />
          ) : (
            <>
              <span className="conversation-list__bulk-count" aria-live="polite">
                {chosen.length} selected
              </span>
              <button
                type="button"
                className="conversation-list__label"
                onClick={() => setPicked(new Set(visible.map((c) => c.id)))}
                data-testid="conversation-select-all"
              >
                All
              </button>
              {onDeleteMany && (
                <button
                  type="button"
                  className="conversation-list__label"
                  disabled={deletable.length === 0}
                  onClick={() => setBulkConfirm(true)}
                  data-testid="conversation-bulk-delete"
                >
                  Delete
                </button>
              )}
              {onPin && (
                <button
                  type="button"
                  className="conversation-list__label"
                  disabled={chosen.length === 0}
                  onClick={() => {
                    const unpinned = chosen.filter(
                      (id) => !conversations.find((c) => c.id === id)?.pinned,
                    );
                    if (unpinned.length === 0) {
                      for (const id of chosen) onPin(id, false);
                    } else {
                      // As many as there is room for: each pin used to check the same
                      // stale count, and 15 picked went past the cap of 10.
                      const room = Math.max(
                        MAX_PINNED_CONVERSATIONS - conversations.filter((c) => c.pinned).length,
                        0,
                      );
                      for (const id of unpinned.slice(0, Math.max(room, 1))) onPin(id, true);
                      if (room > 0 && unpinned.length > room) {
                        setPinNote(
                          `Pinned ${room} of ${unpinned.length}: at most ${MAX_PINNED_CONVERSATIONS} can be pinned.`,
                        );
                      }
                    }
                    endSelecting();
                  }}
                  data-testid="conversation-bulk-pin"
                >
                  Pin / unpin
                </button>
              )}
              {onSetLabels && (
                <button
                  type="button"
                  className="conversation-list__label"
                  disabled={chosen.length === 0}
                  onClick={() => setBulkLabel('')}
                  data-testid="conversation-bulk-label"
                >
                  Label
                </button>
              )}
              <button
                type="button"
                className="conversation-list__label"
                onClick={endSelecting}
                data-testid="conversation-select-done"
              >
                Done
              </button>
            </>
          )}
        </div>
      )}
      {pinNote && (
        <p className="conversation-list__bulk-count" role="status" data-testid="conversation-pin-note">
          {pinNote}
        </p>
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
            {selecting && (
              <input
                type="checkbox"
                className="conversation-list__pick"
                checked={picked.has(conversation.id)}
                onChange={() => togglePicked(conversation.id)}
                aria-label={`Select ${conversation.title}`}
                data-testid="conversation-pick"
              />
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
            ) : labelingId === conversation.id ? (
              <input
                className="conversation-list__edit"
                value={labelDraft}
                autoFocus
                placeholder="labels, comma-separated"
                list="conversation-label-names"
                onChange={(event) => setLabelDraft(event.target.value)}
                onBlur={() => {
                  onSetLabels?.(conversation.id, labelDraft.split(','));
                  setLabelingId(null);
                }}
                onKeyDown={(event) => {
                  if (isImeComposing(event)) return;
                  if (event.key === 'Enter') {
                    refocusRowIdRef.current = conversation.id;
                    event.currentTarget.blur();
                  }
                  if (event.key === 'Escape') {
                    event.stopPropagation();
                    refocusRowIdRef.current = conversation.id;
                    setLabelingId(null);
                  }
                }}
                aria-label={`Labels of ${conversation.title}, comma-separated`}
                data-testid="conversation-labels-input"
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
                    {conversation.labels?.length ? (
                      <span data-testid="conversation-labels">
                        {' '}
                        &middot; {conversation.labels.map((label) => `#${label}`).join(' ')}
                      </span>
                    ) : null}
                    {(conversation.savedCount ?? 0) > 0 && (
                      <span data-testid="conversation-saved-count">
                        {' '}
                        &middot; ★ {conversation.savedCount}
                      </span>
                    )}
                  </span>
                </button>
                <span className="conversation-list__row-actions">
                  {compactActions ? (
                    <OverflowMenu
                      tone="rail"
                      label={`Actions for ${conversation.title}`}
                      testId="conversation-menu"
                      items={[
                        ...(onPin
                          ? [
                              {
                                label: conversation.pinned ? 'Unpin' : 'Pin',
                                onSelect: () => onPin(conversation.id, !conversation.pinned),
                                disabled: !conversation.pinned && pinLimitReached,
                                title:
                                  !conversation.pinned && pinLimitReached
                                    ? `At most ${MAX_PINNED_CONVERSATIONS} pinned conversations`
                                    : undefined,
                                testId: 'conversation-menu-pin',
                              },
                            ]
                          : []),
                        {
                          label: 'Rename',
                          onSelect: () => startEditing(conversation),
                          disabled,
                          testId: 'conversation-menu-rename',
                        },
                        ...(onSetLabels
                          ? [
                              {
                                label: 'Labels…',
                                onSelect: () => {
                                  setLabelingId(conversation.id);
                                  setLabelDraft((conversation.labels ?? []).join(', '));
                                },
                                testId: 'conversation-menu-labels',
                              },
                            ]
                          : []),
                        {
                          label: 'Delete',
                          onSelect: () => setConfirmingDeleteId(conversation.id),
                          disabled: disabled || conversation.id === streamingId,
                          title:
                            conversation.id === streamingId ? 'An answer is still being written here' : undefined,
                          testId: 'conversation-menu-delete',
                        },
                      ]}
                    />
                  ) : (
                    <>
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
                      {onSetLabels && (
                        <button
                          type="button"
                          className="conversation-list__row-action"
                          onClick={() => {
                            setLabelingId(conversation.id);
                            setLabelDraft((conversation.labels ?? []).join(', '));
                          }}
                          aria-label={`Label ${conversation.title}`}
                          title="Labels, to filter the list by"
                          data-testid="conversation-label"
                        >
                          #
                        </button>
                      )}
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
                    </>
                  )}
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
