import { useRef } from 'react';
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
  // A tap fires mouseenter too, and with no mouseleave to follow, the preview stuck
  // open over the thread on touch screens.
  const touched = useRef(false);
  // Opening the viewer moves focus away; closing it hands focus back to this card,
  // which re-opened the preview the user had just moved past.
  const skipNextFocus = useRef(false);
  return (
    <button
      type="button"
      className="citation-card"
      onClick={() => {
        onHoverEnd?.();
        skipNextFocus.current = true;
        onOpen?.();
      }}
      onTouchStart={() => {
        touched.current = true;
      }}
      onMouseEnter={() => {
        if (!touched.current) onHoverStart?.();
      }}
      onMouseLeave={() => {
        touched.current = false;
        onHoverEnd?.();
      }}
      onFocus={() => {
        if (skipNextFocus.current) {
          skipNextFocus.current = false;
          return;
        }
        if (!touched.current) onHoverStart?.();
      }}
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
