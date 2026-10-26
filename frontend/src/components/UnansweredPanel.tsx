import { memo, useDeferredValue, useMemo, useState } from 'react';
import type { QueryLogEntryResponse } from '../api/types';
import { formatRelativeTime } from '../utils/relativeTime';
import { UNCOVERED, groupQuestions, normalized } from '../utils/questions';
import './UnansweredPanel.css';

interface UnansweredPanelProps {
  /** Every logged question and summary, newest first (GET /query-log). */
  questions: QueryLogEntryResponse[] | null;
  error: string | null;
  /** Put the question in the composer, to ask it again (after adding a document). */
  onAsk: (question: string) => void;
  /** Look for its passages in the Search panel. */
  onSearch: (question: string) => void;
  /** Open its trace, while the server still has it. */
  onSelectTrace: (traceId: string) => void;
}

type View = 'unanswered' | 'all' | 'repeated';

/** What was asked (GET /query-log): the questions the documents didn't answer — what
 *  the corpus is missing — every question, or the most asked ones, each asked again,
 *  searched or traced from here. Needs QUERY_LOG_ENABLED and the full key. */
function UnansweredPanel({ questions, error, onAsk, onSearch, onSelectTrace }: UnansweredPanelProps) {
  const [view, setView] = useState<View>('unanswered');
  const [filter, setFilter] = useState('');
  const needle = normalized(useDeferredValue(filter));
  const grouped = useMemo(() => groupQuestions(questions ?? []), [questions]);
  const shown = useMemo(() => {
    let rows = grouped;
    if (view === 'unanswered') rows = rows.filter((row) => row.unanswered > 0);
    if (view === 'repeated') {
      rows = rows.filter((row) => row.count > 1).sort((a, b) => b.count - a.count);
    }
    return needle ? rows.filter((row) => normalized(row.question).includes(needle)) : rows;
  }, [grouped, view, needle]);

  if (error) {
    return (
      <p className="unanswered-panel__message" role="alert">
        {error}
      </p>
    );
  }
  if (questions === null) return <p className="unanswered-panel__message">Loading…</p>;

  const total = questions.length;
  const missed = questions.filter((entry) => entry.status === UNCOVERED).length;
  const answeredShare = total > 0 ? Math.round(((total - missed) / total) * 100) : 0;

  return (
    <>
      <p className="unanswered-panel__summary mono" data-testid="questions-summary">
        {total === 0
          ? 'Nothing asked yet.'
          : `${total} asked · ${answeredShare}% answered from the documents · ${missed} not`}
      </p>
      <div className="unanswered-panel__controls">
        <span role="group" aria-label="Show">
          {(
            [
              ['unanswered', 'Unanswered'],
              ['all', 'All'],
              ['repeated', 'Most asked'],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              className="unanswered-panel__view"
              aria-pressed={view === value}
              onClick={() => setView(value)}
              data-testid={`questions-view-${value}`}
            >
              {label}
            </button>
          ))}
        </span>
        <input
          type="search"
          className="unanswered-panel__filter"
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
          placeholder="Filter…"
          aria-label="Filter questions"
          data-testid="questions-filter"
        />
      </div>
      {shown.length === 0 ? (
        <p className="unanswered-panel__message">
          {view === 'unanswered' && !needle
            ? 'Nothing yet: every logged question was answered from the documents.'
            : view === 'repeated' && !needle
              ? 'No question was asked more than once.'
              : 'No question matches.'}
        </p>
      ) : (
        <ul className="unanswered-panel__list">
          {shown.map((row) => (
            <li
              key={normalized(row.question)}
              className="unanswered-panel__row"
              data-testid="unanswered-row"
            >
              <span className="unanswered-panel__question" dir="auto">
                {row.question}
              </span>
              <span className="unanswered-panel__meta mono">
                {row.count > 1 && <span data-testid="questions-count">×{row.count}</span>}
                {row.unanswered > 0 && view !== 'unanswered' && (
                  <span className="unanswered-panel__missed">
                    {row.unanswered === row.count ? 'not answered' : `${row.unanswered} not answered`}
                  </span>
                )}
                {row.latest.top_score !== null && (
                  <span>best match {row.latest.top_score.toFixed(2)}</span>
                )}
                {row.latest.mode === 'summary' && <span>summary</span>}
                <span className="unanswered-panel__spacer" />
                <span>{formatRelativeTime(row.latest.created_at * 1000)}</span>
              </span>
              <span className="unanswered-panel__actions">
                <button
                  type="button"
                  onClick={() => onAsk(row.question)}
                  aria-label={`Ask “${row.question}” again`}
                  data-testid="unanswered-ask"
                >
                  Ask
                </button>
                <button
                  type="button"
                  onClick={() => onSearch(row.question)}
                  aria-label={`Search the passages for “${row.question}”`}
                >
                  Search
                </button>
                {row.latest.trace_id && (
                  <button
                    type="button"
                    onClick={() => onSelectTrace(row.latest.trace_id!)}
                    aria-label={`Open the trace of “${row.question}”`}
                  >
                    Trace
                  </button>
                )}
              </span>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

export default memo(UnansweredPanel);
