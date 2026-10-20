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
      </span>
    </button>
  );
  if (!onNotRelevant && !markedNotRelevant) return card;
  return (
    <span className="citation-card-wrap">
      {card}
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
    </span>
  );
}

export default CitationCard;
