import { memo, useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { ApiClientError, api, apiErrorMessage } from '../api/client';
import type { PassageNarrowing, SearchResultResponse } from '../api/types';
import MarkedText from './MarkedText';
import { snippetAround } from '../utils/snippet';
import { toDateInput } from '../utils/dates';
import {
  NO_FILTERS,
  type SearchFilters,
  positive,
  searchNarrowing,
  searchTags,
} from '../utils/searchFilters';
import './SearchPanel.css';
import { MAX_QUESTION_CHARS } from '../utils/limits';
import { pageLabel } from '../utils/pages';
import { useStableHandler } from '../hooks/useStableHandler';

interface SearchPanelProps {
  /** The active conversation's scope: the same documents a question would search. */
  filenames: string[];
  tags: string[];
  /** The conversation's page range (with exactly one document scoped) and upload date. */
  narrowing?: PassageNarrowing;
  /** Every tag in use, for the tag filter. */
  availableTags?: string[];
  /** Open the passage in the document viewer, its matching sentences marked. */
  onOpenSource: (filename: string, chunkId: string, highlights?: string[]) => void;
  /** Bumped by the panel's own action button to clear the search. */
  resetKey: number;
  onAuthFailure: (err: unknown) => boolean;
  /** A search asked for from elsewhere (a "not found" answer, or passages like one the
   *  viewer shows): shown and run at once. A new object (``id``) each time, so the
   *  same search can be asked for twice. */
  request?: SearchRequestFromElsewhere | null;
  /** Ask the search's question over only the passages ticked here. */
  onAskAboutPassages?: (question: string, chunkIds: string[]) => void;
  /** Asking can't start now (booting, an answer streaming): the bar's button waits. */
  askDisabled?: boolean;
}

/** A query to search for, or a passage to find passages like. */
export interface SearchRequestFromElsewhere {
  query?: string;
  like?: SimilarTo;
  id: number;
}

/** The passage a "more like this" search starts from. */
export interface SimilarTo {
  chunkId: string;
  /** How it is named while its results show ("handbook.pdf p.3"). */
  label: string;
}

const SNIPPET_CHARS = 220;
// POST /search's rerank_top_k bound (REQUEST_RERANK_TOP_K_MAX).
const MAX_RESULTS = 50;
// The server takes at most this many chunk ids per question (REQUEST_CHUNK_IDS_MAX).
const MAX_PICKED_PASSAGES = 100;

/** Passage search (POST /search): the ranked chunks a question would be answered from,
 *  with their scores, and no LLM involved — so it works when none is reachable and shows
 *  at a glance whether a document answers a question at all. */
function SearchPanel({
  filenames,
  tags,
  narrowing,
  availableTags = [],
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
  const [filters, setFilters] = useState<SearchFilters>(NO_FILTERS);
  // The conversation scope the shown results were searched in: another conversation (or
  // a scope change) leaves them describing documents that are no longer the scope.
  const [searchedScope, setSearchedScope] = useState('');
  // The passage the shown results are like (a "more like this" search), if they are.
  const [like, setLike] = useState<SimilarTo | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const scopeKey = JSON.stringify([filenames, tags, narrowing ?? null]);
  const setFilter = (name: keyof SearchFilters) => (value: string) =>
    setFilters((prev) => ({ ...prev, [name]: value }));

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
    setFilters(NO_FILTERS);
    setLike(null);
  }, [resetKey]);

  const run = useStableHandler(async (text: string, similarTo?: SimilarTo) => {
    const trimmed = similarTo ? '' : text.trim();
    if (!trimmed && !similarTo) return;
    // A newer search replaces one still running.
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setSearching(true);
    setError(null);
    try {
      const count = positive(filters.count, MAX_RESULTS);
      const response = await api.searchPassages(
        {
          query: trimmed,
          ...(similarTo ? { like_chunk_id: similarTo.chunkId } : {}),
          ...(filenames.length > 0 ? { filenames } : {}),
          ...(searchTags(filters, tags).length > 0 ? { tags: searchTags(filters, tags) } : {}),
          ...searchNarrowing(filters, narrowing),
          ...(count !== undefined ? { rerank_top_k: count } : {}),
        },
        controller.signal,
      );
      if (controller.signal.aborted) return;
      setResults(response.results);
      setPicked(new Set());
      setSearchedQuery(trimmed);
      setLike(similarTo ?? null);
      setSearchedScope(scopeKey);
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
    if (request.like) {
      setQuery('');
      void run('', request.like);
      return;
    }
    const text = (request.query ?? '').slice(0, MAX_QUESTION_CHARS);
    setQuery(text);
    void run(text);
  }, [request, run]);

  const looking = searchNarrowing(filters, narrowing);
  const filterCount = Object.values(filters).filter((value) => value.trim() !== '').length;
  const scope =
    (filenames.length > 0 || tags.length > 0
      ? `Searching ${filenames.length > 0 ? `${filenames.length} selected document${filenames.length === 1 ? '' : 's'}` : 'all documents'}${tags.length > 0 ? ` · #${tags.join(' #')}` : ''}`
      : 'Searching all documents') +
    (looking.page_min !== undefined || looking.page_max !== undefined
      ? ` · p.${looking.page_min ?? 1}${looking.page_max !== undefined ? `–${looking.page_max}` : '+'}`
      : '') +
    (looking.uploaded_after !== undefined ? ` · uploaded since ${toDateInput(looking.uploaded_after)}` : '') +
    (looking.uploaded_before !== undefined ? ` · up to ${toDateInput(looking.uploaded_before)}` : '') +
    (filters.tag ? ` · #${filters.tag} only` : '');

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
      <p className="search-panel__scope mono" data-testid="search-scope">{scope}</p>
      <details className="search-panel__filters" data-testid="search-filters">
        <summary>Filters{filterCount > 0 ? ` (${filterCount})` : ''}</summary>
        <div className="search-panel__filter-grid">
          <label>
            <span>Section</span>
            <input
              value={filters.section}
              onChange={(event) => setFilter('section')(event.target.value)}
              placeholder="exact label"
              data-testid="search-filter-section"
            />
          </label>
          <label>
            <span>Leave out</span>
            <input
              value={filters.exclude}
              onChange={(event) => setFilter('exclude')(event.target.value)}
              placeholder="a.pdf, b.docx"
              data-testid="search-filter-exclude"
            />
          </label>
          <label>
            <span>Pages</span>
            <span className="search-panel__filter-pair">
              <input
                type="number"
                min={1}
                inputMode="numeric"
                value={filters.pageMin}
                onChange={(event) => setFilter('pageMin')(event.target.value)}
                aria-label="First page"
                placeholder="from"
                data-testid="search-filter-page-min"
              />
              <input
                type="number"
                min={1}
                inputMode="numeric"
                value={filters.pageMax}
                onChange={(event) => setFilter('pageMax')(event.target.value)}
                aria-label="Last page"
                placeholder="to"
                data-testid="search-filter-page-max"
              />
            </span>
          </label>
          <label>
            <span>Uploaded since</span>
            <input
              type="date"
              value={filters.uploadedAfter}
              onChange={(event) => setFilter('uploadedAfter')(event.target.value)}
              data-testid="search-filter-uploaded"
            />
          </label>
          <label>
            <span>Uploaded up to</span>
            <input
              type="date"
              value={filters.uploadedBefore}
              onChange={(event) => setFilter('uploadedBefore')(event.target.value)}
              data-testid="search-filter-uploaded-before"
            />
          </label>
          {availableTags.length > 0 && (
            <label>
              <span>Tag</span>
              <select
                value={filters.tag}
                onChange={(event) => setFilter('tag')(event.target.value)}
                data-testid="search-filter-tag"
              >
                <option value="">{tags.length > 0 ? `the conversation's (#${tags.join(' #')})` : 'any'}</option>
                {availableTags.map((tag) => (
                  <option key={tag} value={tag}>
                    #{tag}
                  </option>
                ))}
              </select>
            </label>
          )}
          <label>
            <span>Results</span>
            <input
              type="number"
              min={1}
              max={MAX_RESULTS}
              inputMode="numeric"
              value={filters.count}
              onChange={(event) => setFilter('count')(event.target.value)}
              placeholder="8"
              data-testid="search-filter-count"
            />
          </label>
          {filterCount > 0 && (
            <button type="button" className="search-panel__ask-clear" onClick={() => setFilters(NO_FILTERS)}>
              Clear filters
            </button>
          )}
        </div>
      </details>
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
      {like && results !== null && (
        <p className="search-panel__message" data-testid="search-like">
          Passages like {like.label}
        </p>
      )}
      {results !== null && searchedScope !== scopeKey && (
        <p className="search-panel__message" data-testid="search-stale">
          These results are from another scope. Search again to look in this one.
        </p>
      )}
      {weak && results !== null && results.length > 0 && (
        <p className="search-panel__message" data-testid="search-weak">
          No strong matches. These are the closest passages — a fuller question usually finds
          better ones.
        </p>
      )}
      {onAskAboutPassages && picked.size > 0 && searchedQuery && (
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
          onClick={() =>
            onOpenSource(
              result.filename,
              result.chunk_id,
              result.spans?.length ? result.spans.map((span) => span.quote) : undefined,
            )
          }
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
            <MarkedText
              {...snippetAround(result.text, quotesOf(result), SNIPPET_CHARS)}
              className="search-panel__match"
              testId="search-match"
            />
          </span>
        </button>
          <button
            type="button"
            className="search-panel__similar mono"
            onClick={() =>
              void run('', {
                chunkId: result.chunk_id,
                label: `${result.filename} ${pageLabel(result.page, result.page_end)}`,
              })
            }
            aria-label={`Passages like passage ${result.rank} of ${result.filename}`}
            title="Find passages like this one"
            data-testid="search-similar"
          >
            similar
          </button>
        </div>
      ))}
    </>
  );
}

function quotesOf(result: SearchResultResponse): string[] {
  return (result.spans ?? []).map((span) => span.quote);
}

// Memoized: App re-renders on every streamed frame, and none of this depends on it.
export default memo(SearchPanel);
