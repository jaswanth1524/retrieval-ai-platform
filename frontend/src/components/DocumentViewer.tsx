import { useEffect, useMemo, useRef, useState } from 'react';
import { ApiClientError, api } from '../api/client';
import type { DocumentChunkResponse } from '../api/types';
import { useDialog } from '../hooks/useDialog';
import './DocumentViewer.css';

interface DocumentViewerProps {
  filename: string;
  chunkId: string | null;
  onClose: () => void;
}

// Chunks either side of the cited target rendered on first paint. A naive "render the
// first N" would put the target chunk out of reach for a citation deep in a large
// document (e.g. chunk 400 of 500) — centering on it keeps click-through correct
// regardless of document size.
const WINDOW_RADIUS = 30;

function DocumentViewer({ filename, chunkId, onClose }: DocumentViewerProps) {
  const [chunks, setChunks] = useState<DocumentChunkResponse[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  // null = windowed to the default range below; set once the reader expands a bound
  // or asks to see everything.
  const [windowStart, setWindowStart] = useState(0);
  const [windowEnd, setWindowEnd] = useState<number | null>(null);
  const [showAll, setShowAll] = useState(false);
  const targetRef = useRef<HTMLDivElement>(null);
  // Which chunkId the viewer last auto-scrolled to. Compared against the current
  // chunkId (not a boolean) so scrolling fires exactly once per distinct target: not
  // on every window expansion (which would yank the reader back to the target after
  // they've scrolled elsewhere), but still again when App.tsx points the same open
  // viewer at a different citation in the same document (no remount — see below).
  const lastScrolledChunkIdRef = useRef<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setChunks(null);
    setError(null);
    setWindowStart(0);
    setWindowEnd(null);
    setShowAll(false);
    lastScrolledChunkIdRef.current = null;
    api
      .getDocumentContent(filename, controller.signal)
      .then((content) => setChunks(content.chunks))
      .catch((err) => {
        if (controller.signal.aborted) return;
        setError(err instanceof ApiClientError ? err.message : 'Could not load document.');
      });
    return () => controller.abort();
  }, [filename]);

  // Same hook the command palette and settings modal use. This component declared
  // role="dialog" aria-modal="true" while only handling Escape — no focus moved in on
  // open, no Tab trap, no focus restored to the citation that opened it. The hook
  // re-queries focusables on every Tab, which matters here because the chunk list (and
  // its Load-earlier/later buttons) only exists after the fetch resolves.
  const dialogRef = useDialog(true, onClose);

  const targetIndex = useMemo(
    () => (chunks && chunkId !== null ? chunks.findIndex((c) => c.chunk_id === chunkId) : -1),
    [chunks, chunkId],
  );

  // Center the window on the target once chunks arrive, and re-center if the reader
  // clicks a different citation into an already-open viewer for the same document
  // (App.tsx doesn't remount DocumentViewer for that — only `chunkId` changes). Reads
  // windowStart/windowEnd from the latest render rather than listing them as deps, so
  // a manual Load-earlier/later expansion that already covers the new target is left
  // alone instead of being reset every time the effect happens to re-run.
  useEffect(() => {
    if (!chunks) return;
    const center = targetIndex >= 0 ? targetIndex : 0;
    const alreadyWindowed = windowEnd !== null;
    const inWindow = alreadyWindowed && center >= windowStart && center < (windowEnd as number);
    if (inWindow) return;
    setWindowStart(Math.max(0, center - WINDOW_RADIUS));
    setWindowEnd(Math.min(chunks.length, center + WINDOW_RADIUS + 1));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chunks, targetIndex]);

  const visibleChunks = useMemo(() => {
    if (!chunks) return [];
    if (showAll) return chunks;
    return chunks.slice(windowStart, windowEnd ?? chunks.length);
  }, [chunks, showAll, windowStart, windowEnd]);

  useEffect(() => {
    if (!targetRef.current || lastScrolledChunkIdRef.current === chunkId) return;
    targetRef.current.scrollIntoView({ block: 'center' });
    lastScrolledChunkIdRef.current = chunkId;
  }, [visibleChunks, chunkId]);

  return (
    <div
      className="document-viewer"
      role="dialog"
      aria-modal="true"
      aria-label={`Source: ${filename}`}
      ref={dialogRef}
    >
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
          {chunks && !showAll && windowStart > 0 && (
            <button
              type="button"
              className="document-viewer__load-more"
              onClick={() => setWindowStart(Math.max(0, windowStart - WINDOW_RADIUS))}
              data-testid="viewer-load-earlier"
            >
              Load earlier ({windowStart} hidden)
            </button>
          )}
          {visibleChunks.map((chunk) => {
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
          {chunks && !showAll && windowEnd !== null && windowEnd < chunks.length && (
            <button
              type="button"
              className="document-viewer__load-more"
              onClick={() => setWindowEnd(Math.min(chunks.length, (windowEnd ?? 0) + WINDOW_RADIUS))}
              data-testid="viewer-load-later"
            >
              Load later ({chunks.length - windowEnd} hidden)
            </button>
          )}
          {chunks && !showAll && chunks.length > visibleChunks.length && (
            <button
              type="button"
              className="document-viewer__show-all"
              onClick={() => setShowAll(true)}
              data-testid="viewer-show-all"
            >
              Show all {chunks.length} chunks
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default DocumentViewer;
