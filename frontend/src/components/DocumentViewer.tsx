import { useEffect, useRef, useState } from 'react';
import { ApiClientError, api } from '../api/client';
import type { DocumentChunkResponse } from '../api/types';
import './DocumentViewer.css';

interface DocumentViewerProps {
  filename: string;
  chunkId: string | null;
  onClose: () => void;
}

function DocumentViewer({ filename, chunkId, onClose }: DocumentViewerProps) {
  const [chunks, setChunks] = useState<DocumentChunkResponse[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const targetRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    setChunks(null);
    setError(null);
    api
      .getDocumentContent(filename, controller.signal)
      .then((content) => setChunks(content.chunks))
      .catch((err) => {
        if (controller.signal.aborted) return;
        setError(err instanceof ApiClientError ? err.message : 'Could not load document.');
      });
    return () => controller.abort();
  }, [filename]);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  useEffect(() => {
    if (chunks && targetRef.current) {
      targetRef.current.scrollIntoView({ block: 'center' });
    }
  }, [chunks]);

  return (
    <div className="document-viewer" role="dialog" aria-modal="true" aria-label={`Source: ${filename}`}>
      <div className="document-viewer__backdrop" data-testid="viewer-backdrop" onClick={onClose} />
      <div className="document-viewer__panel">
        <div className="document-viewer__header">
          <span className="document-viewer__filename">{filename}</span>
          <button
            type="button"
            className="document-viewer__close"
            onClick={onClose}
            aria-label="Close document viewer"
            data-testid="viewer-close"
          >
            &times;
          </button>
        </div>
        <div className="document-viewer__body">
          {error && <p className="document-viewer__error" role="alert">{error}</p>}
          {!error && chunks === null && <p className="document-viewer__loading">Loading…</p>}
          {chunks?.map((chunk) => {
            const isTarget = chunkId !== null && chunk.chunk_id === chunkId;
            return (
              <div
                key={chunk.chunk_id}
                ref={isTarget ? targetRef : undefined}
                className={`document-viewer__chunk${
                  isTarget ? ' document-viewer__chunk--target' : ''
                }`}
                data-testid="viewer-chunk"
              >
                <div className="document-viewer__chunk-meta mono">
                  p.{chunk.page} &middot; {chunk.section}
                </div>
                <p className="document-viewer__chunk-text">{chunk.text}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default DocumentViewer;
