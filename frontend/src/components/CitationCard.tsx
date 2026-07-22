import './CitationCard.css';

interface CitationCardProps {
  sourceNumber: number;
  filename: string;
  page: number;
  section: string;
  chunkId: string;
  text: string;
  // When provided, the card header becomes a button that opens the source document
  // viewer scrolled to this chunk. Omitted = static display (pre-existing behavior).
  onOpen?: () => void;
}

function CitationCard({
  sourceNumber,
  filename,
  page,
  section,
  chunkId,
  text,
  onOpen,
}: CitationCardProps) {
  const headerContent = (
    <>
      <span className="citation-card__index mono">[{sourceNumber}]</span>
      <span className="citation-card__file">{filename}</span>
      <span className="citation-card__meta mono">
        p.{page} &middot; {section} &middot; {chunkId}
      </span>
    </>
  );

  return (
    <div className="citation-card" data-testid="citation-card">
      {onOpen ? (
        <button
          type="button"
          className="citation-card__header citation-card__header--button"
          onClick={onOpen}
          data-testid="citation-card-open"
          aria-label={`Open ${filename} at source ${sourceNumber}`}
        >
          {headerContent}
        </button>
      ) : (
        <div className="citation-card__header">{headerContent}</div>
      )}
      <blockquote className="citation-card__excerpt">{text}</blockquote>
    </div>
  );
}

export default CitationCard;
