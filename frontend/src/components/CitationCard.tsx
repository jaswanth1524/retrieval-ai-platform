import './CitationCard.css';

interface CitationCardProps {
  sourceNumber: number;
  filename: string;
  page: number;
  section: string;
  onOpen?: () => void;
  onHoverStart?: () => void;
  onHoverEnd?: () => void;
}

function CitationCard({
  sourceNumber,
  filename,
  page,
  section,
  onOpen,
  onHoverStart,
  onHoverEnd,
}: CitationCardProps) {
  return (
    <button
      type="button"
      className="citation-card"
      onClick={onOpen}
      onMouseEnter={onHoverStart}
      onMouseLeave={onHoverEnd}
      onFocus={onHoverStart}
      onBlur={onHoverEnd}
      data-testid="citation-card"
      aria-label={`Open ${filename} at source ${sourceNumber}`}
    >
      <span className="citation-card__index mono">{sourceNumber}</span>
      <span className="citation-card__file">{filename}</span>
      <span className="citation-card__loc mono">
        p.{page} &middot; {section}
      </span>
    </button>
  );
}

export default CitationCard;
