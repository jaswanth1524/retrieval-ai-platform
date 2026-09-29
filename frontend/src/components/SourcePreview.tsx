import type { CitationResponse } from '../api/types';
import './SourcePreview.css';

interface SourcePreviewProps {
  citation: CitationResponse;
}

function SourcePreview({ citation }: SourcePreviewProps) {
  return (
    <div className="source-preview" data-testid="source-preview">
      <div className="source-preview__card">
        <div className="source-preview__header">
          <span className="source-preview__index">{citation.source_number}</span>
          <span className="source-preview__file">{citation.filename}</span>
          <span className="source-preview__loc">
            p.{citation.page} &middot; {citation.section}
          </span>
        </div>
        <p className="source-preview__quote">
          {/* Older turns are saved without the passage text (see useChat); the citation
              still opens the document at this passage. */}
          {citation.text || 'Open the source to read this passage.'}
        </p>
      </div>
    </div>
  );
}

export default SourcePreview;
