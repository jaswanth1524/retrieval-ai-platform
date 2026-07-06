import './CitationCard.css';

interface CitationCardProps {
  sourceNumber: number;
  filename: string;
  page: number;
  section: string;
  chunkId: string;
  text: string;
}

function CitationCard({ sourceNumber, filename, page, section, chunkId, text }: CitationCardProps) {
  return (
    <div className="citation-card" data-testid="citation-card">
      <div className="citation-card__header">
        <span className="citation-card__index mono">[{sourceNumber}]</span>
        <span className="citation-card__file">{filename}</span>
        <span className="citation-card__meta mono">
          p.{page} &middot; {section} &middot; {chunkId}
        </span>
      </div>
      <blockquote className="citation-card__excerpt">{text}</blockquote>
    </div>
  );
}

export default CitationCard;
