import { memo } from 'react';
import type { QueryLogEntryResponse } from '../api/types';
import { formatRelativeTime } from '../utils/relativeTime';
import './UnansweredPanel.css';

interface UnansweredPanelProps {
  questions: QueryLogEntryResponse[] | null;
  error: string | null;
  /** Put the question in the composer, to ask it again (after adding a document). */
  onAsk: (question: string) => void;
  /** Look for its passages in the Search panel. */
  onSearch: (question: string) => void;
  /** Open its trace, while the server still has it. */
  onSelectTrace: (traceId: string) => void;
}

/** Questions the documents didn't answer (GET /query-log?uncovered=true): what the
 *  corpus is missing, newest first. Needs QUERY_LOG_ENABLED and the full key. */
function UnansweredPanel({ questions, error, onAsk, onSearch, onSelectTrace }: UnansweredPanelProps) {
  if (error) {
    return (
      <p className="unanswered-panel__message" role="alert">
        {error}
      </p>
    );
  }
  if (questions === null) return <p className="unanswered-panel__message">Loading…</p>;
  if (questions.length === 0) {
    return (
      <p className="unanswered-panel__message">
        Nothing yet: every logged question was answered from the documents.
      </p>
    );
  }
  return (
    <>
      <p className="unanswered-panel__summary mono">
        {questions.length} question{questions.length === 1 ? '' : 's'} the documents didn’t answer
      </p>
      <ul className="unanswered-panel__list">
        {questions.map((item, index) => (
          <li
            key={`${item.created_at}-${index}`}
            className="unanswered-panel__row"
            data-testid="unanswered-row"
          >
            <span className="unanswered-panel__question" dir="auto">
              {item.question}
            </span>
            <span className="unanswered-panel__meta mono">
              {item.top_score !== null && <span>best match {item.top_score.toFixed(2)}</span>}
              {item.mode === 'summary' && <span>summary</span>}
              <span className="unanswered-panel__spacer" />
              <span>{formatRelativeTime(item.created_at * 1000)}</span>
            </span>
            <span className="unanswered-panel__actions">
              <button type="button" onClick={() => onAsk(item.question)}>
                Ask
              </button>
              <button type="button" onClick={() => onSearch(item.question)}>
                Search
              </button>
              {item.trace_id && (
                <button type="button" onClick={() => onSelectTrace(item.trace_id!)}>
                  Trace
                </button>
              )}
            </span>
          </li>
        ))}
      </ul>
    </>
  );
}

export default memo(UnansweredPanel);
