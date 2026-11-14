import { memo, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';
import { api, apiErrorMessage } from '../api/client';
import {
  checkable,
  judgeSaved,
  SAVED_CHECK_LABELS,
  SAVED_CHECK_TOP_K,
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

/** Every saved answer (★ on an answer), across the conversations: found again here
 *  rather than by remembering which conversation it was in. */
function SavedPanel({ answers, onOpen, onUnsave, sourceChangeOf, workspace = '' }: SavedPanelProps) {
  const [filter, setFilter] = useState('');
  // "Check against the documents": each checkable answer's question searched again.
  const [checks, setChecks] = useState<Record<string, SavedCheck>>({});
  const [checking, setChecking] = useState<{ done: number; total: number } | null>(null);
  const [checkError, setCheckError] = useState<string | null>(null);
  const checkRef = useRef<AbortController | null>(null);
  useEffect(() => () => checkRef.current?.abort(), []);
  const runCheck = async () => {
    checkRef.current?.abort();
    const controller = new AbortController();
    checkRef.current = controller;
    const items = answers.filter((item) => checkable(item, workspace));
    setChecks({});
    setCheckError(null);
    setChecking({ done: 0, total: items.length });
    // One at a time: each search takes one of the server's question slots.
    for (const [index, item] of items.entries()) {
      try {
        const response = await api.searchPassages(
          { query: item.question ?? '', rerank_top_k: SAVED_CHECK_TOP_K },
          controller.signal,
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
      <p className="unanswered-panel__message">
        Nothing saved yet. Save an answer with ☆ under it to find it here.
      </p>
    );
  }
  return (
    <>
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
            <li key={`${item.conversationId}:${item.turn.id}`} className="unanswered-panel__row" data-testid="saved-row">
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
                <button
                  type="button"
                  onClick={() => onUnsave(item.conversationId, item.turn.id)}
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
    </>
  );
}

// Memoized: App re-renders on every streamed frame; the list keeps its identity then.
export default memo(SavedPanel);
