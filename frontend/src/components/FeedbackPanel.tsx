import { memo, useState } from 'react';
import type { FeedbackItemResponse, FeedbackRating } from '../api/types';
import { formatRelativeTime } from '../utils/relativeTime';
import './FeedbackPanel.css';

interface FeedbackPanelProps {
  feedback: FeedbackItemResponse[] | null;
  error: string | null;
  /** Open the trace of a rated answer, when it is still on the server. */
  onSelectTrace: (traceId: string) => void;
}

type Filter = 'all' | FeedbackRating;

// The emoji is decoration: read aloud it made "thumbs down Down".
const FILTERS: { key: Filter; label: string; glyph?: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'down', label: 'Down', glyph: '👎' },
  { key: 'up', label: 'Up', glyph: '👍' },
];

/** Recorded answer ratings (GET /feedback), newest first — thumbs-down rows are the
 *  questions the corpus or the prompt is failing. */
function FeedbackPanel({ feedback, error, onSelectTrace }: FeedbackPanelProps) {
  const [filter, setFilter] = useState<Filter>('all');
  if (error) {
    return (
      <p className="feedback-panel__message" role="alert">
        {error}
      </p>
    );
  }
  if (feedback === null) return <p className="feedback-panel__message">Loading…</p>;

  const rows = filter === 'all' ? feedback : feedback.filter((item) => item.rating === filter);
  const downCount = feedback.filter((item) => item.rating === 'down').length;
  return (
    <>
      <div className="feedback-panel__filters" role="group" aria-label="Show ratings">
        {FILTERS.map((option) => (
          <button
            key={option.key}
            type="button"
            className={`feedback-panel__filter${filter === option.key ? ' feedback-panel__filter--active' : ''}`}
            aria-pressed={filter === option.key}
            onClick={() => setFilter(option.key)}
          >
            {option.glyph && <span aria-hidden="true">{option.glyph} </span>}
            {option.label}
          </button>
        ))}
      </div>
      <p className="feedback-panel__summary mono">
        {feedback.length} rating{feedback.length === 1 ? '' : 's'} · {downCount} down
      </p>
      {rows.length === 0 && <p className="feedback-panel__message">No ratings here yet.</p>}
      {rows.map((item) => {
        const body = (
          <>
            <span className="feedback-panel__question">
              {/* role="img": an aria-label on a bare span is not announced. */}
              <span role="img" aria-label={item.rating === 'up' ? 'Rated up' : 'Rated down'}>
                {item.rating === 'up' ? '👍' : '👎'}
              </span>{' '}
              {item.question}
            </span>
            {item.answer_excerpt && (
              <span className="feedback-panel__excerpt">{item.answer_excerpt}</span>
            )}
            <span className="feedback-panel__meta mono">
              <span>{item.cited_filenames.join(', ') || 'no sources'}</span>
              <span className="feedback-panel__spacer" />
              <span>{formatRelativeTime(item.created_at * 1000)}</span>
            </span>
          </>
        );
        return item.trace_id ? (
          <button
            key={item.id}
            type="button"
            className="feedback-panel__row"
            onClick={() => onSelectTrace(item.trace_id!)}
            title="Open this answer's trace"
            data-testid="feedback-panel-row"
          >
            {body}
          </button>
        ) : (
          <div key={item.id} className="feedback-panel__row" data-testid="feedback-panel-row">
            {body}
          </div>
        );
      })}
    </>
  );
}

// Memoized: App re-renders on every streamed frame, and none of this depends on it.
export default memo(FeedbackPanel);
