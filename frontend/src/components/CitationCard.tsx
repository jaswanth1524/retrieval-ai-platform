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
}: CitationCardProps) {
  const trigger = usePreviewTrigger({ onOpen, onHoverStart, onHoverEnd });
  return (
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
}

export default CitationCard;
