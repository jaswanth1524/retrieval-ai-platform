import { memo, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';
import { api, apiErrorMessage } from '../api/client';
import {
  checkable,
  judgeSaved,
  SAVED_CHECK_LABELS,
  savedCheckRequest,
  type SavedCheck,
} from '../utils/savedCheck';
import type { SavedAnswer } from '../hooks/useChat';
import { SOURCE_CHANGE_HINTS, type SourceChange } from '../utils/citationFreshness';
import { formatRelativeTime } from '../utils/relativeTime';
import { normalized } from '../utils/questions';
import { sliceWhole } from '../utils/text';
import './UnansweredPanel.css';

interface SavedPanelProps {
  answers: SavedAnswer[];
  /** Open the conversation at that answer. */
  onOpen: (conversationId: string, turnId: string) => void;
  onUnsave: (conversationId: string, turnId: string) => void;
  /** Keep (or, with "", remove) the reader's note on a saved answer. */
  onSetNote?: (conversationId: string, turnId: string, note: string) => void;
  /** Ask the answer's question again in place: the old answer stays as a version. */
  onAskAgain?: (item: SavedAnswer) => void;
  /** Whether a cited document was deleted or uploaded again after an answer. */
  sourceChangeOf?: (filename: string, answeredAtMs: number) => SourceChange | null;
  /** The workspace whose documents ``sourceChangeOf`` knows: another's answers are
   *  never checked against them (each of their sources would read "deleted"). */
  workspace?: string;
}

/** The answer's sources that changed since it was saved, said in one line. */
function changedSources(
  item: SavedAnswer,
  sourceChangeOf: SavedPanelProps['sourceChangeOf'],
  workspace: string,
): string | null {
  if (!sourceChangeOf || item.workspace !== workspace) return null;
  const deleted: string[] = [];
  const changed: string[] = [];
  for (const filename of new Set(item.turn.sources.map((source) => source.filename))) {
    const change = sourceChangeOf(filename, item.turn.timestamp);
    if (change === 'deleted') deleted.push(filename);
    else if (change === 'changed') changed.push(filename);
  }
  const parts = [
    ...(changed.length ? [`changed since: ${changed.join(', ')}`] : []),
    ...(deleted.length ? [`deleted: ${deleted.join(', ')}`] : []),
  ];
  return parts.length ? parts.join(' · ') : null;
}

// How much of an answer a row shows.
const EXCERPT_CHARS = 180;

function excerpt(content: string): string {
  const flat = content.replace(/\s+/g, ' ').trim();
  return flat.length > EXCERPT_CHARS ? `${sliceWhole(flat, EXCERPT_CHARS).trimEnd()}…` : flat;
}

// The last check's verdicts, by workspace (this page load only).
let keptChecks: { workspace: string; checks: Record<string, SavedCheck> } = { workspace: '', checks: {} };

/** Every saved answer (★ on an answer), across the conversations: found again here
 *  rather than by remembering which conversation it was in. */
function SavedPanel({
  answers,
  onOpen,
  onUnsave,
  onSetNote,
  onAskAgain,
  sourceChangeOf,
  workspace = '',
}: SavedPanelProps) {
  const [filter, setFilter] = useState('');
  // The note being written: which answer, and its text so far.
  const [editing, setEditing] = useState<{ key: string; text: string } | null>(null);
  // "Check against the documents": each checkable answer's question searched again.
  const [checks, setChecksState] = useState<Record<string, SavedCheck>>(() =>
    keptChecks.workspace === workspace ? keptChecks.checks : {},
  );
  // Kept while the panel is closed (it unmounts): a check of many answers isn't
  // thrown away by a look at another panel.
  const setChecks = (update: (prev: Record<string, SavedCheck>) => Record<string, SavedCheck>) =>
    setChecksState((prev) => {
      const next = update(prev);
      keptChecks = { workspace, checks: next };
      return next;
    });
  const [checking, setChecking] = useState<{ done: number; total: number } | null>(null);
  const [checkError, setCheckError] = useState<string | null>(null);
  const checkRef = useRef<AbortController | null>(null);
  useEffect(() => () => checkRef.current?.abort(), []);
  const runCheck = async () => {
    checkRef.current?.abort();
    const controller = new AbortController();
    checkRef.current = controller;
    const items = answers.filter((item) => checkable(item, workspace));
    setChecks(() => ({}));
    setCheckError(null);
    setChecking({ done: 0, total: items.length });
    // One at a time: each search takes one of the server's question slots.
    for (const [index, item] of items.entries()) {
      try {
        // In the workspace the check began in, whatever is opened meanwhile.
        const response = await api.searchPassages(
          savedCheckRequest(item),
          controller.signal,
          workspace,
        );
        if (controller.signal.aborted) return;
        const verdict = judgeSaved(item, response);
        setChecks((prev) => ({ ...prev, [`${item.conversationId}:${item.turn.id}`]: verdict }));
      } catch (err) {
        if (controller.signal.aborted) return;
        setCheckError(apiErrorMessage(err, 'The check stopped.'));
        break;
      }
      setChecking({ done: index + 1, total: items.length });
    }
    setChecking(null);
  };
  // After Unsave the row goes, and focus with it: it moves to the next row's Open (the
  // one before when it was the last), else to the panel's message or filter.
  const listRef = useRef<HTMLDivElement | null>(null);
  const focusAfterUnsave = useRef<string | null | undefined>(undefined);
  const unsave = (item: SavedAnswer, rows: readonly SavedAnswer[]) => {
    const index = rows.indexOf(item);
    const neighbour = rows[index + 1] ?? rows[index - 1];
    focusAfterUnsave.current = neighbour ? `${neighbour.conversationId}:${neighbour.turn.id}` : null;
    onUnsave(item.conversationId, item.turn.id);
  };
  useEffect(() => {
    const key = focusAfterUnsave.current;
    if (key === undefined) return;
    focusAfterUnsave.current = undefined;
    const root = listRef.current;
    if (!root) return;
    const target =
      (key !== null &&
        [...root.querySelectorAll<HTMLElement>('[data-saved-key]')]
          .find((row) => row.dataset.savedKey === key)
          ?.querySelector<HTMLElement>('[data-testid="saved-open"]')) ||
      root.querySelector<HTMLElement>('[data-testid="saved-filter"], [data-testid="saved-empty"]');
    target?.focus();
  }, [answers]);
  const checkableCount = answers.filter((item) => checkable(item, workspace)).length;
  const needle = normalized(useDeferredValue(filter));
  const shown = useMemo(
    () =>
      needle
        ? answers.filter((item) =>
            normalized(`${item.question ?? ''} ${item.turn.content} ${item.conversationTitle}`).includes(
              needle,
            ),
          )
        : answers,
    [answers, needle],
  );

  if (answers.length === 0) {
    return (
      <div ref={listRef}>
        <p className="unanswered-panel__message" tabIndex={-1} data-testid="saved-empty">
          Nothing saved yet. Save an answer with ☆ under it to find it here.
        </p>
      </div>
    );
  }
  return (
    <div ref={listRef}>
      <div className="unanswered-panel__controls">
        <input
          type="search"
          className="unanswered-panel__filter"
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
          placeholder="Filter…"
          aria-label="Filter saved answers"
          data-testid="saved-filter"
        />
        {checkableCount > 0 && (
          <button
            type="button"
            onClick={() => void runCheck()}
            disabled={checking !== null}
            title="Search each saved question again (no answer is written) and flag the ones whose passages no longer rank"
            data-testid="saved-check"
          >
            {checking ? `Checking ${checking.done}/${checking.total}…` : 'Check against the documents'}
          </button>
        )}
      </div>
      {checkError && (
        <p className="unanswered-panel__message" role="alert">
          {checkError}
        </p>
      )}
      {shown.length === 0 ? (
        <p className="unanswered-panel__message">No saved answer matches.</p>
      ) : (
        <ul className="unanswered-panel__list">
          {shown.map((item) => (
            <li
              key={`${item.conversationId}:${item.turn.id}`}
              className="unanswered-panel__row"
              data-saved-key={`${item.conversationId}:${item.turn.id}`}
              data-testid="saved-row"
            >
              <span className="unanswered-panel__question" dir="auto">
                {item.question ?? 'Answer'}
              </span>
              <span className="unanswered-panel__meta" dir="auto">
                {excerpt(item.turn.content)}
              </span>
              {(() => {
                const note = changedSources(item, sourceChangeOf, workspace);
                return note ? (
                  <span
                    className="unanswered-panel__meta saved-panel__changed"
                    title={SOURCE_CHANGE_HINTS.changed}
                    data-testid="saved-changed"
                  >
                    Sources {note}
                  </span>
                ) : null;
              })()}
              {checks[`${item.conversationId}:${item.turn.id}`] && (
                <span
                  className={`unanswered-panel__meta${
                    checks[`${item.conversationId}:${item.turn.id}`] === 'found' ? '' : ' saved-panel__changed'
                  }`}
                  data-testid="saved-check-result"
                >
                  {SAVED_CHECK_LABELS[checks[`${item.conversationId}:${item.turn.id}`]]}
                </span>
              )}
              {editing?.key === `${item.conversationId}:${item.turn.id}` ? (
                <form
                  className="saved-panel__note-form"
                  onSubmit={(event) => {
                    event.preventDefault();
                    onSetNote?.(item.conversationId, item.turn.id, editing.text);
                    setEditing(null);
                  }}
                >
                  <textarea
                    value={editing.text}
                    onChange={(event) => setEditing({ ...editing, text: event.target.value })}
                    maxLength={2000}
                    rows={3}
                    aria-label="Your note on this answer"
                    autoFocus
                    onKeyDown={(event) => {
                      if (event.key === 'Escape') setEditing(null);
                    }}
                    data-testid="saved-note-input"
                  />
                  <span className="unanswered-panel__actions">
                    <button type="submit" data-testid="saved-note-save">
                      Save note
                    </button>
                    <button type="button" onClick={() => setEditing(null)}>
                      Cancel
                    </button>
                  </span>
                </form>
              ) : (
                item.turn.note && (
                  <span className="unanswered-panel__meta saved-panel__note" dir="auto" data-testid="saved-note">
                    Note: {item.turn.note}
                  </span>
                )
              )}
              <span className="unanswered-panel__meta mono">
                <span>{item.conversationTitle}</span>
                <span className="unanswered-panel__spacer" />
                <span>{formatRelativeTime(item.turn.timestamp)}</span>
              </span>
              <span className="unanswered-panel__actions">
                <button
                  type="button"
                  onClick={() => onOpen(item.conversationId, item.turn.id)}
                  aria-label={`Open the saved answer to “${item.question ?? item.conversationTitle}”`}
                  data-testid="saved-open"
                >
                  Open
                </button>
                {/* In the workspace in use only: it is asked against these documents. */}
                {onAskAgain && item.question && !item.task && item.workspace === workspace && (
                  <button
                    type="button"
                    onClick={() => onAskAgain(item)}
                    title="Ask it again now; the current answer is kept as an earlier version to compare"
                    data-testid="saved-ask-again"
                  >
                    Ask again
                  </button>
                )}
                {onSetNote && (
                  <button
                    type="button"
                    onClick={() =>
                      setEditing({
                        key: `${item.conversationId}:${item.turn.id}`,
                        text: item.turn.note ?? '',
                      })
                    }
                    aria-label={`${item.turn.note ? 'Edit the note on' : 'Add a note to'} the answer to “${item.question ?? item.conversationTitle}”`}
                    data-testid="saved-note-edit"
                  >
                    {item.turn.note ? 'Edit note' : 'Note'}
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => unsave(item, shown)}
                  aria-label={`Unsave the answer to “${item.question ?? item.conversationTitle}”`}
                  data-testid="saved-unsave"
                >
                  Unsave
                </button>
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// Memoized: App re-renders on every streamed frame; the list keeps its identity then.
export default memo(SavedPanel);
