import { memo } from 'react';
import type { SavedAnswer } from '../hooks/useChat';
import type { SourceChange } from '../utils/citationFreshness';
import { formatRelativeTime } from '../utils/relativeTime';
import { reviewReasons } from '../utils/review';
import './UnansweredPanel.css';

interface ReviewPanelProps {
  /** This workspace's answers, newest first (useChat.answersIn). */
  answers: SavedAnswer[];
  onOpen: (conversationId: string, turnId: string) => void;
  sourceChangeOf?: (filename: string, answeredAtMs: number) => SourceChange | null;
}

// At most this many rows: the oldest problems matter least.
const MAX_ROWS = 200;

/** Answers worth a second look, across this workspace's conversations: not found, weak
 *  matches, uncited or auto-cited, cut off, or resting on a document that changed or was
 *  deleted since. */
function ReviewPanel({ answers, onOpen, sourceChangeOf }: ReviewPanelProps) {
  const rows = answers
    .map((item) => ({ item, reasons: reviewReasons(item.turn, sourceChangeOf) }))
    .filter((row) => row.reasons.length > 0)
    .slice(0, MAX_ROWS);
  if (rows.length === 0) {
    return (
      <p className="unanswered-panel__message" data-testid="review-empty">
        Nothing to review: every answer here was found, cited, and rests on documents
        as they are now.
      </p>
    );
  }
  return (
    <ul className="unanswered-panel__list" data-testid="review-list">
      {rows.map(({ item, reasons }) => (
        <li
          key={`${item.conversationId}:${item.turn.id}`}
          className="unanswered-panel__row"
          data-testid="review-row"
        >
          <span className="unanswered-panel__question" dir="auto">
            {item.question ?? 'Answer'}
          </span>
          <span className="unanswered-panel__meta" data-testid="review-reasons">
            {reasons.join(' · ')}
          </span>
          <span className="unanswered-panel__meta mono">
            <span>{item.conversationTitle}</span>
            <span className="unanswered-panel__spacer" />
            <span>{formatRelativeTime(item.turn.timestamp)}</span>
          </span>
          <span className="unanswered-panel__actions">
            <button
              type="button"
              onClick={() => onOpen(item.conversationId, item.turn.id)}
              aria-label={`Open the answer to “${item.question ?? item.conversationTitle}”`}
              data-testid="review-open"
            >
              Open
            </button>
          </span>
        </li>
      ))}
    </ul>
  );
}

export default memo(ReviewPanel);
