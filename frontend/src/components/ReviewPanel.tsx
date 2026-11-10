import { memo, useState } from 'react';
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
  /** Mark an answer looked at (hidden from the list unless reviewed ones are shown). */
  onMarkReviewed?: (conversationId: string, turnId: string, reviewed: boolean) => void;
}

// At most this many rows: the oldest problems matter least.
const MAX_ROWS = 200;

/** Answers worth a second look, across this workspace's conversations: not found, weak
 *  matches, uncited or auto-cited, cut off, or resting on a document that changed or was
 *  deleted since. */
function ReviewPanel({ answers, onOpen, sourceChangeOf, onMarkReviewed }: ReviewPanelProps) {
  const [showReviewed, setShowReviewed] = useState(false);
  const flagged = answers
    .map((item) => ({ item, reasons: reviewReasons(item.turn, sourceChangeOf) }))
    .filter((row) => row.reasons.length > 0);
  const reviewedCount = flagged.filter((row) => row.item.turn.reviewedAt).length;
  const rows = flagged
    .filter((row) => showReviewed || !row.item.turn.reviewedAt)
    .slice(0, MAX_ROWS);
  const toggle =
    reviewedCount > 0 ? (
      <label className="unanswered-panel__toggle">
        <input
          type="checkbox"
          checked={showReviewed}
          onChange={(event) => setShowReviewed(event.target.checked)}
          data-testid="review-show-reviewed"
        />{' '}
        Show the {reviewedCount} reviewed
      </label>
    ) : null;
  if (rows.length === 0) {
    return (
      <>
        <p className="unanswered-panel__message" data-testid="review-empty">
          {flagged.length > 0
            ? 'Every answer worth a second look here has been reviewed.'
            : 'Nothing to review: every answer here was found, cited, and rests on documents as they are now.'}
        </p>
        {toggle}
      </>
    );
  }
  return (
    <>
    {toggle}
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
            {onMarkReviewed && (
              <button
                type="button"
                aria-pressed={Boolean(item.turn.reviewedAt)}
                onClick={() =>
                  onMarkReviewed(item.conversationId, item.turn.id, !item.turn.reviewedAt)
                }
                data-testid="review-mark"
              >
                {item.turn.reviewedAt ? 'Reviewed ✓' : 'Mark reviewed'}
              </button>
            )}
          </span>
        </li>
      ))}
    </ul>
    </>
  );
}

export default memo(ReviewPanel);
