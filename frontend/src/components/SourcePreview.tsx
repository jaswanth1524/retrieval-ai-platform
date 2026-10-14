import type { CitationResponse } from '../api/types';
import { SOURCE_PREVIEW_ID } from '../hooks/usePreviewTrigger';
import { pageLabel } from '../utils/pages';
import './SourcePreview.css';

interface SourcePreviewProps {
  citation: CitationResponse;
}

function SourcePreview({ citation }: SourcePreviewProps) {
  return (
    <div className="source-preview" id={SOURCE_PREVIEW_ID} role="tooltip" data-testid="source-preview">
      <div className="source-preview__card">
        <div className="source-preview__header">
          <span className="source-preview__index">{citation.source_number}</span>
          <span className="source-preview__file">{citation.filename}</span>
          <span className="source-preview__loc">
            {pageLabel(citation.page, citation.page_end)} &middot; {citation.section}
          </span>
        </div>
        <p className="source-preview__quote">
          {/* The passage the answer rests on when the server found one, else the
              chunk's opening. Older turns are saved without either (see useChat); the
              citation still opens the document at this passage. */}
          {citation.spans?.length
            ? citation.spans.map((span) => span.quote).join(' … ')
            : citation.text || 'Open the source to read this passage.'}
        </p>
      </div>
    </div>
  );
}

export default SourcePreview;
