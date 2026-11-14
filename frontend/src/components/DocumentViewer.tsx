import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { matchRanges } from '../utils/find';
import { isShortcutModifier } from '../utils/platform';
import { ApiClientError, api, apiErrorMessage } from '../api/client';
import type { ContentPage } from '../api/client';
import type { DocumentChunkResponse, DocumentContentResponse } from '../api/types';
import { useDialog } from '../hooks/useDialog';
import { CITATION_WINDOW_RADIUS, takePrefetchedContent } from '../utils/contentPrefetch';
import './DocumentViewer.css';
import { pageLabel } from '../utils/pages';
import { copyText } from '../utils/clipboard';
import ViewerOutline from './ViewerOutline';
import MarkedText from './MarkedText';
import { MAX_QUESTION_CHARS } from '../utils/limits';

interface DocumentViewerProps {
  filename: string;
  chunkId: string | null;
  onClose: () => void;
  /** Download the uploaded file itself; absent when the server keeps no originals. */
  onDownloadOriginal?: () => void;
  /** The stored original is a PDF small enough to show here (see App). */
  canPreviewOriginal?: boolean;
  /** Passages of the cited chunk the answer rests on, marked within it. */
  highlights?: string[];
  /** Ask a question about one passage only (absent while asking can't start). */
  onAskAboutPassage?: (question: string, chunkId: string) => void;
  /** Search for the passages most like one ("more like this"). */
  onFindSimilar?: (chunkId: string, label: string) => void;
  /** Copy a link that opens this document (at its cited passage) on this server. */
  onCopyLink?: () => void;
  /** Open this document at another passage (its contents, "go to page"). */
  onGoTo?: (chunkId: string) => void;
}

type ViewerMode = 'passages' | 'original';

interface OriginalPreview {
  filename: string;
  url: string | null;
  error: string | null;
}

// The original's download and blob URL, owned per document (see the effects below).
interface OriginalFetch {
  filename: string;
  controller: AbortController;
  url: string | null;
}

// Chunks either side of the cited target fetched on open, and fetched per Load
// earlier/later. The server pages by ordinal (GET /documents/{f}/content?around=), so a
// citation deep in a large document costs one small page, not the whole document.
// Mirrors CONTENT_UNPAGED_MAX_CHUNKS (api/main.py): the most one unpaged request returns.
const UNPAGED_MAX_CHUNKS = 2000;
const WINDOW_RADIUS = CITATION_WINDOW_RADIUS;

interface Loaded {
  chunks: DocumentChunkResponse[];
  total: number;
  targetFound: boolean | null;
}

/** ``text`` with each of ``quotes`` marked. Found by search rather than by the span's
 *  offsets, which count code points where JS strings count UTF-16 units; a quote that
 *  isn't in the text (the document changed since) is simply not marked. */
function ordinalOf(chunk: DocumentChunkResponse | undefined): number | null {
  return chunk?.chunk_ordinal ?? null;
}

function DocumentViewer({
  filename,
  chunkId,
  onClose,
  onDownloadOriginal,
  canPreviewOriginal = false,
  highlights,
  onAskAboutPassage,
  onFindSimilar,
  onCopyLink,
  onGoTo,
}: DocumentViewerProps) {
  // The passage whose question box is open, and what is typed in it.
  const [asking, setAsking] = useState<{ chunkId: string; question: string } | null>(null);
  // Find in the loaded passages (⌘F / Ctrl+F while the viewer is open).
  const [findOpen, setFindOpen] = useState(false);
  const [outlineOpen, setOutlineOpen] = useState(false);
  // Which passage "Copy with citation" just copied (its button says so).
  const [copiedChunk, setCopiedChunk] = useState<string | null>(null);
  const [findQuery, setFindQuery] = useState('');
  const [findIndex, setFindIndex] = useState(0);
  const findInputRef = useRef<HTMLInputElement>(null);
  // The passage whose ask box Escape closed: focus goes back to its "Ask about this"
  // button, rather than to the page body with the box gone.
  const askClosedRef = useRef<string | null>(null);
  useEffect(() => {
    const chunkId = askClosedRef.current;
    if (asking !== null || chunkId === null) return;
    askClosedRef.current = null;
    document
      .querySelector<HTMLElement>(`[data-chunk-id="${CSS.escape(chunkId)}"] [data-testid="viewer-ask"]`)
      ?.focus();
  }, [asking]);
  // Both keyed by filename, so pointing the open viewer at another document shows its
  // passages straight away, in the same render, with nothing of the previous original.
  const [modeFor, setModeFor] = useState<{ filename: string; mode: ViewerMode }>({
    filename,
    mode: 'passages',
  });
  const mode: ViewerMode = modeFor.filename === filename ? modeFor.mode : 'passages';
  const setMode = (next: ViewerMode) => setModeFor({ filename, mode: next });
  const [originalState, setOriginal] = useState<OriginalPreview | null>(null);
  const original = originalState?.filename === filename ? originalState : null;
  const originalFetchRef = useRef<OriginalFetch | null>(null);
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

  const fetchPage = (
    options: ContentPage,
    apply: (content: DocumentContentResponse) => void,
    prefetched?: Promise<DocumentContentResponse>,
  ) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    return (prefetched ?? api.getDocumentContent(filename, controller.signal, options))
      .then((content) => {
        if (controller.signal.aborted) return false;
        // A load that worked clears an earlier load's error (a failed "Load later"
        // left its alert up after the retry succeeded).
        setError(null);
        apply(content);
        return true;
      })
      .catch((err) => {
        if (controller.signal.aborted) return false;
        setError(apiErrorMessage(err, 'Could not load document.'));
        return true;
      });
  };

  // (Re)load around the target when the document or the cited chunk changes — unless
  // that chunk is already on screen (a second citation into the same open document).
  useEffect(() => {
    const current = loadedRef.current;
    if (current && current.chunks.some((chunk) => chunk.chunk_id === chunkId)) {
      // Found on screen, so it exists: a "passage changed" note left by an earlier
      // citation whose chunk was missing no longer applies to this one.
      if (current.targetFound === false) setLoaded({ ...current, targetFound: true });
      return;
    }
    setLoaded(null);
    setError(null);
    lastScrolledChunkIdRef.current = null;
    const page: ContentPage = chunkId
      ? { around: chunkId, radius: WINDOW_RADIUS }
      : { start: 1, end: 2 * WINDOW_RADIUS + 1 };
    const prefetched = chunkId ? takePrefetchedContent(filename, chunkId) : undefined;
    void fetchPage(
      page,
      (content) =>
        setLoaded({
          chunks: content.chunks,
          total: content.total_chunks ?? content.chunks.length,
          // An older server ignores the paging and sends everything: look for it here.
          targetFound:
            content.target_found ??
            (chunkId ? content.chunks.some((chunk) => chunk.chunk_id === chunkId) : null),
        }),
      prefetched,
    );
    // Only the request this run started: a second citation into the open document
    // (the early return above) used to abort a "Load more" in flight on its way out.
    const started = controllerRef.current;
    return () => started?.abort();
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
  // Past the unpaged cap "show all" fetches the first UNPAGED_MAX_CHUNKS; once those are
  // on screen it would fetch them again and change nothing (Load later still pages on).
  const firstCapLoaded = first === 1 && (chunks?.length ?? 0) >= UNPAGED_MAX_CHUNKS;

  // Where the reader was before chunks were added above them, restored after: "Show
  // all" (and Load earlier) put up to 2000 chunks above the passage being read, and the
  // browser's own scroll anchoring didn't hold it, so the page jumped far away. (Not
  // compatible with content-visibility on the chunks: estimated heights above the
  // anchor resolved after this ran and moved it again.)
  const bodyRef = useRef<HTMLDivElement>(null);
  const anchorRef = useRef<{ chunkId: string; top: number } | null>(null);
  const captureAnchor = () => {
    const body = bodyRef.current;
    if (!body) return;
    const frameTop = body.getBoundingClientRect().top;
    const visible = [...body.querySelectorAll<HTMLElement>('[data-chunk-id]')].find(
      (element) => element.getBoundingClientRect().bottom > frameTop,
    );
    anchorRef.current = visible
      ? { chunkId: visible.dataset.chunkId ?? '', top: visible.getBoundingClientRect().top - frameTop }
      : null;
  };
  // The chunk to focus once a load lands, when the button that asked for it is gone
  // (the last page loaded): focus fell to <body> inside a modal dialog otherwise.
  const focusChunkRef = useRef<string | null>(null);
  useLayoutEffect(() => {
    const wanted = focusChunkRef.current;
    focusChunkRef.current = null;
    const body = bodyRef.current;
    if (!wanted || !body) return;
    const active = document.activeElement;
    if (active && active !== document.body && body.closest('[role="dialog"]')?.contains(active)) {
      return; // the button is still there, keeping focus
    }
    [...body.querySelectorAll<HTMLElement>('[data-chunk-id]')]
      .find((element) => element.dataset.chunkId === wanted)
      ?.focus({ preventScroll: true });
  }, [loaded]);
  useLayoutEffect(() => {
    const anchor = anchorRef.current;
    const body = bodyRef.current;
    anchorRef.current = null;
    if (!anchor || !body) return;
    const element = [...body.querySelectorAll<HTMLElement>('[data-chunk-id]')].find(
      (candidate) => candidate.dataset.chunkId === anchor.chunkId,
    );
    if (element) {
      body.scrollTop +=
        element.getBoundingClientRect().top - body.getBoundingClientRect().top - anchor.top;
    }
  }, [loaded]);

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
      if (direction !== 'later') captureAnchor();
      focusChunkRef.current =
        direction === 'all'
          ? (chunkId ?? content.chunks[0]?.chunk_id ?? null)
          : (content.chunks[0]?.chunk_id ?? null);
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
    }).then((settled) => {
      // An aborted load (a newer one took its place) leaves the newer one's state be.
      if (settled) setLoadingMore(false);
    });
  };

  // The original is fetched the first time it is asked for, once per document: the
  // browser's PDF viewer shows it from a blob URL, since an <iframe src> to the API
  // couldn't send the X-API-Key header. The download and the URL belong to the document,
  // not the mode: switching to Passages and back shows the same copy (revoking it there
  // left the frame pointing at a dead URL), and only another document or closing the
  // viewer cancels the download and frees the URL.
  useEffect(
    () => () => {
      const fetched = originalFetchRef.current;
      originalFetchRef.current = null;
      fetched?.controller.abort();
      if (fetched?.url) URL.revokeObjectURL(fetched.url);
    },
    [filename],
  );
  useEffect(() => {
    if (mode !== 'original' || originalFetchRef.current?.filename === filename) return;
    const fetched: OriginalFetch = { filename, controller: new AbortController(), url: null };
    originalFetchRef.current = fetched;
    const { signal } = fetched.controller;
    setOriginal({ filename, url: null, error: null });
    api
      .getDocumentOriginal(filename, signal)
      .then((blob) => {
        if (signal.aborted) return;
        // Typed explicitly: the browser shows a PDF only for application/pdf.
        fetched.url = URL.createObjectURL(new Blob([blob], { type: 'application/pdf' }));
        setOriginal({ filename, url: fetched.url, error: null });
      })
      .catch((err) => {
        if (signal.aborted) return;
        setOriginal({
          filename,
          url: null,
          error:
            err instanceof ApiClientError && err.statusCode === 404
              ? 'The original file is not stored for this document.'
              : err instanceof ApiClientError
                ? err.message
                : 'Could not load the original.',
        });
      });
  }, [mode, filename]);
  // Opens at the cited passage's page.
  const targetPage =
    loaded?.chunks.find((chunk) => chunkId !== null && chunk.chunk_id === chunkId)?.page ?? 1;

  // Every occurrence in the loaded passages, in reading order: next/previous step
  // through these, not through passages (a passage with three mentions is three stops).
  const findMatches = useMemo(
    () =>
      findOpen && findQuery.trim()
        ? (chunks ?? []).flatMap((chunk) =>
            matchRanges(chunk.text, findQuery).map(([start]) => ({ chunk_id: chunk.chunk_id, start })),
          )
        : [],
    [chunks, findOpen, findQuery],
  );
  const matchedPassages = useMemo(
    () => new Set(findMatches.map((match) => match.chunk_id)).size,
    [findMatches],
  );
  const currentMatch = findMatches.length > 0 ? findMatches[findIndex % findMatches.length] : null;
  const scrollToChunk = (id: string) =>
    bodyRef.current
      ?.querySelector<HTMLElement>(`[data-chunk-id="${CSS.escape(id)}"]`)
      ?.scrollIntoView({ block: 'center' });
  const goToMatch = (step: number) => {
    if (findMatches.length === 0) return;
    const next = (((findIndex + step) % findMatches.length) + findMatches.length) % findMatches.length;
    setFindIndex(next);
    scrollToChunk(findMatches[next].chunk_id);
  };
  // Then to the occurrence itself, once it is marked (a long passage holds several).
  useEffect(() => {
    if (!findOpen) return;
    bodyRef.current
      ?.querySelector<HTMLElement>('mark[data-current="true"]')
      ?.scrollIntoView?.({ block: 'nearest' });
  }, [findIndex, findQuery, findOpen]);
  // As in a browser's find: typing shows the first match ("1 of N" is where the view
  // is), and Enter then moves to the next.
  const firstMatchId = findMatches[0]?.chunk_id;
  useEffect(() => {
    if (firstMatchId) scrollToChunk(firstMatchId);
    // Only when the query (and so the first match) changes, not on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [findQuery, firstMatchId]);
  // Closing find puts focus back on the button that opened it (it unmounted with
  // the bar open; focus fell to <body>).
  const findOpenRef = useRef<HTMLButtonElement | null>(null);
  const closeFind = () => {
    setFindOpen(false);
    requestAnimationFrame(() => findOpenRef.current?.focus());
  };
  const openFind = () => {
    setFindOpen(true);
    // Focused when it shows; what was searched before is selected, to type over (only
    // that: typed before the frame came, a selection would eat the first letter).
    const previous = findQuery !== '';
    requestAnimationFrame(() => {
      const input = findInputRef.current;
      if (!input) return;
      input.focus();
      if (previous) input.select();
    });
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
      onKeyDown={(event) => {
        // The browser's own find can't see passages not loaded, and missed the dialog's
        // scroll box: this one searches what is loaded and offers the rest.
        if (event.key.toLowerCase() === 'f' && isShortcutModifier(event.nativeEvent) && mode === 'passages') {
          event.preventDefault();
          openFind();
        }
      }}
    >
      <div className="document-viewer__backdrop" data-testid="viewer-backdrop" onClick={onClose} />
      <div className="document-viewer__panel">
        <div className="document-viewer__header">
          <span className="document-viewer__filename" title={filename}>
            {filename}
          </span>
          {canPreviewOriginal && (
            <span className="document-viewer__modes" role="group" aria-label="View">
              <button
                type="button"
                className="document-viewer__mode"
                aria-pressed={mode === 'passages'}
                onClick={() => setMode('passages')}
                data-testid="viewer-mode-passages"
              >
                Passages
              </button>
              <button
                type="button"
                className="document-viewer__mode"
                aria-pressed={mode === 'original'}
                onClick={() => setMode('original')}
                data-testid="viewer-mode-original"
              >
                Original PDF
              </button>
            </span>
          )}
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
          {onCopyLink && (
            <button
              type="button"
              className="document-viewer__original"
              onClick={onCopyLink}
              title={chunkId ? 'A link to this passage, for anyone with access to this server' : 'A link to this document'}
              data-testid="viewer-copy-link"
            >
              Copy link
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
          {mode === 'passages' && onGoTo && (
            <button
              type="button"
              className="document-viewer__original"
              aria-expanded={outlineOpen}
              onClick={() => setOutlineOpen((open) => !open)}
              data-testid="viewer-outline-open"
            >
              Contents
            </button>
          )}
          {mode === 'passages' && !findOpen && (
            <button
              ref={findOpenRef}
              type="button"
              className="document-viewer__original"
              onClick={openFind}
              aria-label="Find in this document"
              data-testid="viewer-find-open"
            >
              Find
            </button>
          )}
        </div>
        {mode === 'passages' && outlineOpen && onGoTo && (
          <ViewerOutline
            filename={filename}
            onGoTo={(target) => {
              setOutlineOpen(false);
              onGoTo(target);
            }}
          />
        )}
        {mode === 'original' && (
          <div className="document-viewer__pdf" data-testid="viewer-original">
            {original?.error && (
              <p className="document-viewer__error" role="alert">
                {original.error}
              </p>
            )}
            {!original?.error && !original?.url && (
              <p className="document-viewer__loading">Loading the original…</p>
            )}
            {original?.url && (
              <iframe
                className="document-viewer__pdf-frame"
                title={`${filename}, page ${targetPage}`}
                src={`${original.url}#page=${targetPage}`}
                data-testid="viewer-original-frame"
              />
            )}
          </div>
        )}
        {findOpen && mode === 'passages' && (
          <div className="document-viewer__find" role="search" data-testid="viewer-find">
            <input
              ref={findInputRef}
              type="search"
              value={findQuery}
              onChange={(event) => {
                setFindQuery(event.target.value);
                setFindIndex(0);
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault();
                  goToMatch(event.shiftKey ? -1 : 1);
                } else if (event.key === 'Escape') {
                  // Closes the find bar, not the viewer.
                  event.stopPropagation();
                  closeFind();
                }
              }}
              placeholder="Find in the passages…"
              aria-label="Find in this document"
              data-testid="viewer-find-input"
            />
            <span className="mono" role="status" data-testid="viewer-find-count">
              {findQuery.trim()
                ? findMatches.length > 0
                  ? `${(findIndex % findMatches.length) + 1} of ${findMatches.length}${
                      matchedPassages > 1 ? ` in ${matchedPassages} passages` : ''
                    }`
                  : 'no passage'
                : ''}
            </span>
            <button type="button" onClick={() => goToMatch(-1)} aria-label="Previous match" disabled={findMatches.length === 0}>
              ‹
            </button>
            <button type="button" onClick={() => goToMatch(1)} aria-label="Next match" disabled={findMatches.length === 0} data-testid="viewer-find-next">
              ›
            </button>
            {hiddenBefore + hiddenAfter > 0 && !firstCapLoaded && (
              <button
                type="button"
                onClick={() => loadMore('all')}
                disabled={loadingMore}
                title="Only the loaded passages are searched"
                data-testid="viewer-find-load-all"
              >
                Search the whole document
              </button>
            )}
            <button type="button" onClick={closeFind} aria-label="Close find">
              ✕
            </button>
          </div>
        )}
        <div ref={bodyRef} className="document-viewer__body" hidden={mode === 'original'}>
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
                }${currentMatch?.chunk_id === chunk.chunk_id ? ' document-viewer__chunk--found' : ''}`}
                data-chunk-id={chunk.chunk_id}
                data-quote-source={`${filename}, ${pageLabel(chunk.page, chunk.page_end)}${
                  chunk.section ? `, ${chunk.section}` : ''
                }`}
                // Focusable from script only: where focus goes when a load button goes.
                tabIndex={-1}
                data-testid="viewer-chunk"
              >
                <div className="document-viewer__chunk-meta mono">
                  {pageLabel(chunk.page, chunk.page_end)} &middot; {chunk.section}
                  {onAskAboutPassage && asking?.chunkId !== chunk.chunk_id && (
                    <button
                      type="button"
                      className="document-viewer__ask"
                      onClick={() => setAsking({ chunkId: chunk.chunk_id, question: '' })}
                      aria-label={`Ask about this passage (${pageLabel(chunk.page, chunk.page_end)})`}
                      data-testid="viewer-ask"
                    >
                      Ask about this
                    </button>
                  )}
                  <button
                    type="button"
                    className="document-viewer__ask"
                    onClick={() =>
                      void copyText(
                        `${chunk.text.trim()}\n— ${filename}, ${pageLabel(chunk.page, chunk.page_end)}` +
                          (chunk.section ? `, ${chunk.section}` : ''),
                      ).then((copied) => setCopiedChunk(copied ? chunk.chunk_id : null))
                    }
                    aria-label={`Copy this passage with where it comes from (${pageLabel(chunk.page, chunk.page_end)})`}
                    data-testid="viewer-copy-quote"
                  >
                    {copiedChunk === chunk.chunk_id ? 'Copied' : 'Copy with citation'}
                  </button>
                  {onFindSimilar && (
                    <button
                      type="button"
                      className="document-viewer__ask"
                      onClick={() =>
                        onFindSimilar(
                          chunk.chunk_id,
                          `${filename} ${pageLabel(chunk.page, chunk.page_end)}`,
                        )
                      }
                      aria-label={`Passages like this one (${pageLabel(chunk.page, chunk.page_end)})`}
                      data-testid="viewer-similar"
                    >
                      Similar
                    </button>
                  )}
                </div>
                {onAskAboutPassage && asking?.chunkId === chunk.chunk_id && (
                  <form
                    className="document-viewer__ask-form"
                    onSubmit={(event) => {
                      event.preventDefault();
                      const question = asking.question.trim();
                      if (!question) return;
                      onAskAboutPassage(question, chunk.chunk_id);
                      setAsking(null);
                    }}
                  >
                    <input
                      autoFocus
                      value={asking.question}
                      maxLength={MAX_QUESTION_CHARS}
                      onChange={(event) =>
                        setAsking({ chunkId: chunk.chunk_id, question: event.target.value })
                      }
                      onKeyDown={(event) => {
                        if (event.key === 'Escape') {
                          event.stopPropagation();
                          askClosedRef.current = chunk.chunk_id;
                          setAsking(null);
                        }
                      }}
                      placeholder="Your question about this passage…"
                      aria-label="Question about this passage"
                      data-testid="viewer-ask-input"
                    />
                    <button type="submit" disabled={!asking.question.trim()} data-testid="viewer-ask-submit">
                      Ask
                    </button>
                  </form>
                )}
                <p className="document-viewer__chunk-text" dir="auto">
                  {findOpen && findQuery.trim() && matchRanges(chunk.text, findQuery).length > 0 ? (
                    <FoundText
                      text={chunk.text}
                      query={findQuery}
                      current={currentMatch?.chunk_id === chunk.chunk_id ? currentMatch.start : null}
                    />
                  ) : isTarget && highlights?.length ? (
                    <MarkedText
                      text={chunk.text}
                      quotes={highlights}
                      className="document-viewer__quote"
                      testId="viewer-quote"
                    />
                  ) : (
                    chunk.text
                  )}
                </p>
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
          {chunks && loaded && hiddenBefore + hiddenAfter > 0 && !firstCapLoaded && (
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

/** A passage with every match of the find query marked. */
function FoundText({ text, query, current = null }: { text: string; query: string; current?: number | null }) {
  const parts: ReactNode[] = [];
  let position = 0;
  for (const [start, end] of matchRanges(text, query)) {
    parts.push(text.slice(position, start));
    const here = start === current;
    parts.push(
      <mark
        key={start}
        className={`document-viewer__found${here ? ' document-viewer__found--current' : ''}`}
        data-testid="viewer-found"
        {...(here ? { 'aria-current': true as const, 'data-current': 'true' } : {})}
      >
        {text.slice(start, end)}
      </mark>,
    );
    position = end;
  }
  parts.push(text.slice(position));
  return <>{parts}</>;
}

// Memoized: App re-renders on every frame of a streaming answer, and an open viewer
// holds up to every passage of a document.
export default memo(DocumentViewer);
