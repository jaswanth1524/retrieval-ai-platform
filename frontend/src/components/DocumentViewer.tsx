import { useEffect, useRef, useState } from 'react';
import { ApiClientError, api } from '../api/client';
import type { ContentPage } from '../api/client';
import type { DocumentChunkResponse, DocumentContentResponse } from '../api/types';
import { useDialog } from '../hooks/useDialog';
import './DocumentViewer.css';

interface DocumentViewerProps {
  filename: string;
  chunkId: string | null;
  onClose: () => void;
  /** Download the uploaded file itself; absent when the server keeps no originals. */
  onDownloadOriginal?: () => void;
}

// Chunks either side of the cited target fetched on open, and fetched per Load
// earlier/later. The server pages by ordinal (GET /documents/{f}/content?around=), so a
// citation deep in a large document costs one small page, not the whole document.
// Mirrors CONTENT_UNPAGED_MAX_CHUNKS (api/main.py): the most one unpaged request returns.
const UNPAGED_MAX_CHUNKS = 2000;
const WINDOW_RADIUS = 30;

interface Loaded {
  chunks: DocumentChunkResponse[];
  total: number;
  targetFound: boolean | null;
}

function ordinalOf(chunk: DocumentChunkResponse | undefined): number | null {
  return chunk?.chunk_ordinal ?? null;
}

function DocumentViewer({ filename, chunkId, onClose, onDownloadOriginal }: DocumentViewerProps) {
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const targetRef = useRef<HTMLDivElement>(null);
  // Which chunkId the viewer last auto-scrolled to. Compared against the current
  // chunkId (not a boolean) so scrolling fires exactly once per distinct target: not
  // on every page loaded (which would yank the reader back to the target after
  // they've scrolled elsewhere), but still again when App.tsx points the same open
  // viewer at a different citation in the same document (no remount — see below).
  const lastScrolledChunkIdRef = useRef<string | null>(null);
  const loadedRef = useRef<Loaded | null>(null);
  loadedRef.current = loaded;
  const controllerRef = useRef<AbortController | null>(null);

  const fetchPage = (options: ContentPage, apply: (content: DocumentContentResponse) => void) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    return api
      .getDocumentContent(filename, controller.signal, options)
      .then((content) => {
        if (!controller.signal.aborted) apply(content);
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setError(err instanceof ApiClientError ? err.message : 'Could not load document.');
      });
  };

  // (Re)load around the target when the document or the cited chunk changes — unless
  // that chunk is already on screen (a second citation into the same open document).
  useEffect(() => {
    const current = loadedRef.current;
    if (current && current.chunks.some((chunk) => chunk.chunk_id === chunkId)) return;
    setLoaded(null);
    setError(null);
    lastScrolledChunkIdRef.current = null;
    const page: ContentPage = chunkId
      ? { around: chunkId, radius: WINDOW_RADIUS }
      : { start: 1, end: 2 * WINDOW_RADIUS + 1 };
    void fetchPage(page, (content) =>
      setLoaded({
        chunks: content.chunks,
        total: content.total_chunks ?? content.chunks.length,
        // An older server ignores the paging and sends everything: look for it here.
        targetFound:
          content.target_found ??
          (chunkId ? content.chunks.some((chunk) => chunk.chunk_id === chunkId) : null),
      }),
    );
    return () => controllerRef.current?.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filename, chunkId]);

  // Same hook the command palette and settings modal use. This component declared
  // role="dialog" aria-modal="true" while only handling Escape — no focus moved in on
  // open, no Tab trap, no focus restored to the citation that opened it. The hook
  // re-queries focusables on every Tab, which matters here because the chunk list (and
  // its Load-earlier/later buttons) only exists after the fetch resolves.
  const dialogRef = useDialog(true, onClose);

  const chunks = loaded?.chunks ?? null;
  const first = ordinalOf(chunks?.[0]);
  const last = ordinalOf(chunks?.[chunks.length - 1]);
  // Legacy points without ordinals can't be paged: the server sent them all.
  const hiddenBefore = chunks && first !== null ? first - 1 : 0;
  const hiddenAfter = chunks && loaded && last !== null ? Math.max(0, loaded.total - last) : 0;

  const loadMore = (direction: 'earlier' | 'later' | 'all') => {
    if (!loaded || first === null || last === null) return;
    setLoadingMore(true);
    const options: ContentPage =
      direction === 'earlier'
        ? { start: Math.max(1, first - WINDOW_RADIUS), end: first - 1 }
        : direction === 'later'
          ? { start: last + 1, end: last + WINDOW_RADIUS }
          : {};
    void fetchPage(options, (content) => {
      setLoaded((prev) => {
        if (!prev) return prev;
        const merged =
          direction === 'all'
            ? content.chunks
            : direction === 'earlier'
              ? [...content.chunks, ...prev.chunks]
              : [...prev.chunks, ...content.chunks];
        return { ...prev, chunks: merged, total: content.total_chunks ?? prev.total };
      });
    }).finally(() => setLoadingMore(false));
  };

  useEffect(() => {
    if (!targetRef.current || lastScrolledChunkIdRef.current === chunkId) return;
    targetRef.current.scrollIntoView({ block: 'center' });
    lastScrolledChunkIdRef.current = chunkId;
  }, [chunks, chunkId]);

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
          {onDownloadOriginal && (
            <button
              type="button"
              className="document-viewer__original"
              onClick={onDownloadOriginal}
              data-testid="viewer-download-original"
            >
              Download original
            </button>
          )}
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
          {/* Chunk ids hash the chunk text, so a re-index or re-upload since the answer
              leaves the citation pointing at a passage that no longer exists. Opening at
              the top with nothing highlighted read as a broken link. */}
          {chunks && chunkId !== null && loaded?.targetFound === false && (
            <p className="document-viewer__notice" role="status" data-testid="viewer-passage-missing">
              This passage changed since the answer was written — the document has been
              re-indexed. Showing the document from the start.
            </p>
          )}
          {chunks && hiddenBefore > 0 && (
            <button
              type="button"
              className="document-viewer__load-more"
              onClick={() => loadMore('earlier')}
              disabled={loadingMore}
              data-testid="viewer-load-earlier"
            >
              Load earlier ({hiddenBefore} hidden)
            </button>
          )}
          {(chunks ?? []).map((chunk) => {
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
          {chunks && hiddenAfter > 0 && (
            <button
              type="button"
              className="document-viewer__load-more"
              onClick={() => loadMore('later')}
              disabled={loadingMore}
              data-testid="viewer-load-later"
            >
              Load later ({hiddenAfter} hidden)
            </button>
          )}
          {chunks && loaded && hiddenBefore + hiddenAfter > 0 && (
            <button
              type="button"
              className="document-viewer__show-all"
              onClick={() => loadMore('all')}
              disabled={loadingMore}
              data-testid="viewer-show-all"
            >
              {loaded.total > UNPAGED_MAX_CHUNKS
                ? `Show the first ${UNPAGED_MAX_CHUNKS} chunks`
                : `Show all ${loaded.total} chunks`}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default DocumentViewer;
