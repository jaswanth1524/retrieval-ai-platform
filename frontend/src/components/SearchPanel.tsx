import { memo, useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { ApiClientError, api, apiErrorMessage } from '../api/client';
import type { SearchResultResponse } from '../api/types';
import './SearchPanel.css';
import { MAX_QUESTION_CHARS } from '../utils/limits';
import { pageLabel } from '../utils/pages';
import { useStableHandler } from '../hooks/useStableHandler';

interface SearchPanelProps {
  /** The active conversation's scope: the same documents a question would search. */
  filenames: string[];
  tags: string[];
  /** Open the passage in the document viewer. */
  onOpenSource: (filename: string, chunkId: string) => void;
  /** Bumped by the panel's own action button to clear the search. */
  resetKey: number;
  onAuthFailure: (err: unknown) => boolean;
  /** A search asked for from elsewhere (a "not found" answer): shown and run at once.
   *  A new object (``id``) each time, so the same query can be asked for twice. */
  request?: { query: string; id: number } | null;
  /** Ask the search's question over only the passages ticked here. */
  onAskAboutPassages?: (question: string, chunkIds: string[]) => void;
  /** Asking can't start now (booting, an answer streaming): the bar's button waits. */
  askDisabled?: boolean;
}

const SNIPPET_CHARS = 220;
// The server takes at most this many chunk ids per question (REQUEST_CHUNK_IDS_MAX).
const MAX_PICKED_PASSAGES = 100;

/** Passage search (POST /search): the ranked chunks a question would be answered from,
 *  with their scores, and no LLM involved — so it works when none is reachable and shows
 *  at a glance whether a document answers a question at all. */
function SearchPanel({
  filenames,
  tags,
  onOpenSource,
  resetKey,
  onAuthFailure,
  request = null,
  onAskAboutPassages,
  askDisabled = false,
}: SearchPanelProps) {
  const [query, setQuery] = useState('');
  // Ticked passages, by chunk id, and the query that found them (asked about them).
  const [picked, setPicked] = useState<Set<string>>(() => new Set());
  const [searchedQuery, setSearchedQuery] = useState('');
  const [results, setResults] = useState<SearchResultResponse[] | null>(null);
  const [weak, setWeak] = useState(false);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => () => controllerRef.current?.abort(), []);
  useEffect(() => {
    if (resetKey === 0) return;
    controllerRef.current?.abort();
    setQuery('');
    setResults(null);
    setPicked(new Set());
    setWeak(false);
    setError(null);
    setSearching(false);
  }, [resetKey]);

  const run = useStableHandler(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    // A newer search replaces one still running.
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setSearching(true);
    setError(null);
    try {
      const response = await api.searchPassages(
        {
          query: trimmed,
          ...(filenames.length > 0 ? { filenames } : {}),
          ...(tags.length > 0 ? { tags } : {}),
        },
        controller.signal,
      );
      if (controller.signal.aborted) return;
      setResults(response.results);
      setPicked(new Set());
      setSearchedQuery(trimmed);
      setWeak(response.weak_matches === true && response.results.length > 0);
    } catch (err) {
      if (controller.signal.aborted) return;
      setResults(null);
      if (onAuthFailure(err)) return;
      setError(
        // A 429 is the question slots being full, unless it is RATE_LIMIT_PER_MINUTE,
        // whose message ("Too many requests; try again in N s.") says it better.
        err instanceof ApiClientError &&
          err.statusCode === 429 &&
          !err.message.startsWith('Too many requests')
          ? 'The server is busy answering questions. Try again in a few seconds.'
          : apiErrorMessage(err, 'Search failed.'),
      );
    } finally {
      if (controllerRef.current === controller) setSearching(false);
    }
  });

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    await run(query);
  };

  useEffect(() => {
    if (!request) return;
    setQuery(request.query.slice(0, MAX_QUESTION_CHARS));
    void run(request.query.slice(0, MAX_QUESTION_CHARS));
  }, [request, run]);

  const scope =
    filenames.length > 0 || tags.length > 0
      ? `Searching ${filenames.length > 0 ? `${filenames.length} selected document${filenames.length === 1 ? '' : 's'}` : 'all documents'}${tags.length > 0 ? ` · #${tags.join(' #')}` : ''}`
      : 'Searching all documents';

  return (
    <>
      <form className="search-panel__form" onSubmit={(event) => void submit(event)} role="search">
        <input
          type="search"
          className="search-panel__input"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Find passages…"
          aria-label="Search passages"
          maxLength={MAX_QUESTION_CHARS}
          data-testid="search-input"
        />
        <button
          type="submit"
          className="search-panel__submit"
          disabled={searching || !query.trim()}
          data-testid="search-submit"
        >
          {searching ? '…' : 'Search'}
        </button>
      </form>
      <p className="search-panel__scope mono">{scope}</p>
      {/* Mounted before any search, so the first result count is read out too. */}
      <span className="visually-hidden" role="status" data-testid="search-status">
        {searching
          ? 'Searching…'
          : results === null
            ? ''
            : results.length === 0
              ? 'No passages found.'
              : weak
                ? `No strong matches; ${results.length} closest passage${results.length === 1 ? '' : 's'} shown.`
                : `${results.length} passage${results.length === 1 ? '' : 's'} found.`}
      </span>
      {error && (
        <p className="search-panel__message" role="alert" data-testid="search-error">
          {error}
        </p>
      )}
      {results !== null && results.length === 0 && (
        <p className="search-panel__message">No passages matched.</p>
      )}
      {weak && results !== null && results.length > 0 && (
        <p className="search-panel__message" data-testid="search-weak">
          No strong matches. These are the closest passages — a fuller question usually finds
          better ones.
        </p>
      )}
      {onAskAboutPassages && picked.size > 0 && (
        <div className="search-panel__ask-bar" data-testid="search-ask-bar">
          <span className="mono">
            {picked.size} passage{picked.size === 1 ? '' : 's'} picked
          </span>
          <button
            type="button"
            onClick={() => setPicked(new Set())}
            className="search-panel__ask-clear"
          >
            Clear
          </button>
          <button
            type="button"
            className="search-panel__ask"
            aria-disabled={askDisabled}
            onClick={() => {
              if (askDisabled) return;
              onAskAboutPassages(searchedQuery, [...picked]);
            }}
            data-testid="search-ask-picked"
          >
            Ask about {picked.size === 1 ? 'it' : 'these'}
          </button>
        </div>
      )}
      {results?.map((result) => (
        <div key={result.chunk_id} className="search-panel__item">
          {onAskAboutPassages && (
            <input
              type="checkbox"
              className="search-panel__pick"
              checked={picked.has(result.chunk_id)}
              onChange={(event) => {
                const checked = event.target.checked;
                setPicked((prev) => {
                  const next = new Set(prev);
                  if (checked && next.size < MAX_PICKED_PASSAGES) next.add(result.chunk_id);
                  else next.delete(result.chunk_id);
                  return next;
                });
              }}
              aria-label={`Pick passage ${result.rank} of ${result.filename}`}
              data-testid="search-pick"
            />
          )}
        <button
          type="button"
          className="search-panel__result"
          onClick={() => onOpenSource(result.filename, result.chunk_id)}
          title="Open this passage in the document"
          data-testid="search-result"
        >
          <span className="search-panel__source">
            <span className="search-panel__rank mono">{result.rank}</span>
            <span className="search-panel__file">{result.filename}</span>
            <span className="search-panel__score mono" title="Cross-encoder relevance, 0 to 1">
              {result.rerank_score.toFixed(2)}
            </span>
          </span>
          <span className="search-panel__where mono">
            {pageLabel(result.page, result.page_end)}
            {result.section ? ` · ${result.section}` : ''}
          </span>
          <span className="search-panel__snippet" dir="auto">
            {result.text.length > SNIPPET_CHARS
              ? `${result.text.slice(0, SNIPPET_CHARS).trimEnd()}…`
              : result.text}
          </span>
        </button>
        </div>
      ))}
    </>
  );
}

// Memoized: App re-renders on every streamed frame, and none of this depends on it.
export default memo(SearchPanel);
