import { usePreviewTrigger } from '../hooks/usePreviewTrigger';
import { pageLabel } from '../utils/pages';
import './CitationCard.css';

interface CitationCardProps {
  sourceNumber: number;
  filename: string;
  page: number;
  pageEnd?: number | null;
  section: string;
  onOpen?: () => void;
  onHoverStart?: () => void;
  onHoverEnd?: () => void;
  /** Rate this source as not relevant to the answer (feedback on); once per source. */
  onNotRelevant?: () => void;
  markedNotRelevant?: boolean;
  /** The cross-encoder's score for the passage (0-1); shown when known. */
  score?: number | null;
  /** Ask the question again with this passage left out. */
  onAskWithout?: () => void;
  /** Asking can't start now (an answer is streaming). */
  askDisabled?: boolean;
}

function CitationCard({
  sourceNumber,
  filename,
  page,
  pageEnd,
  section,
  onOpen,
  onHoverStart,
  onHoverEnd,
  onNotRelevant,
  markedNotRelevant = false,
  score,
  onAskWithout,
  askDisabled = false,
}: CitationCardProps) {
  const trigger = usePreviewTrigger({ onOpen, onHoverStart, onHoverEnd });
  const card = (
    <button
      type="button"
      className="citation-card"
      {...trigger}
      data-testid="citation-card"
      aria-label={`Open ${filename} at source ${sourceNumber}`}
    >
      <span className="citation-card__index mono">{sourceNumber}</span>
      <span className="citation-card__file">{filename}</span>
      <span className="citation-card__loc mono">
        {pageLabel(page, pageEnd)} &middot; {section}
        {typeof score === 'number' && (
          <span
            className="citation-card__score"
            title="How closely the passage matched the question (cross-encoder, 0 to 1)"
            data-testid="citation-score"
          >
            {' '}
            &middot; {score.toFixed(2)}
          </span>
        )}
      </span>
    </button>
  );
  if (!onNotRelevant && !markedNotRelevant && !onAskWithout) return card;
  return (
    <span className="citation-card-wrap">
      {card}
      {onAskWithout && (
        <button
          type="button"
          className="citation-card__irrelevant"
          aria-disabled={askDisabled || undefined}
          onClick={() => !askDisabled && onAskWithout()}
          aria-label={`Ask again without source ${sourceNumber}`}
          title="Ask again without this passage"
          data-testid="citation-ask-without"
        >
          ↻ without
        </button>
      )}
      {(onNotRelevant || markedNotRelevant) && (
      <button
        type="button"
        className="citation-card__irrelevant"
        onClick={onNotRelevant}
        disabled={markedNotRelevant || !onNotRelevant}
        aria-pressed={markedNotRelevant}
        aria-label={
          markedNotRelevant
            ? `Source ${sourceNumber} marked not relevant`
            : `Mark source ${sourceNumber} not relevant`
        }
        title={markedNotRelevant ? 'Marked not relevant' : 'Not relevant to this answer'}
        data-testid="citation-not-relevant"
      >
        {markedNotRelevant ? '✓ not relevant' : '✕'}
      </button>
      )}
    </span>
  );
}

export default CitationCard;
