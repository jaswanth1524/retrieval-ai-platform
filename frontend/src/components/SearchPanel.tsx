import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { ApiClientError, api } from '../api/client';
import type { SearchResultResponse } from '../api/types';
import './SearchPanel.css';

interface SearchPanelProps {
  /** The active conversation's scope: the same documents a question would search. */
  filenames: string[];
  tags: string[];
  /** Open the passage in the document viewer. */
  onOpenSource: (filename: string, chunkId: string) => void;
  /** Bumped by the panel's own action button to clear the search. */
  resetKey: number;
  onAuthFailure: (err: unknown) => boolean;
}

const SNIPPET_CHARS = 220;

/** Passage search (POST /search): the ranked chunks a question would be answered from,
 *  with their scores, and no LLM involved — so it works when none is reachable and shows
 *  at a glance whether a document answers a question at all. */
function SearchPanel({ filenames, tags, onOpenSource, resetKey, onAuthFailure }: SearchPanelProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResultResponse[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => () => controllerRef.current?.abort(), []);
  useEffect(() => {
    if (resetKey === 0) return;
    controllerRef.current?.abort();
    setQuery('');
    setResults(null);
    setError(null);
    setSearching(false);
  }, [resetKey]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const trimmed = query.trim();
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
    } catch (err) {
      if (controller.signal.aborted) return;
      setResults(null);
      if (onAuthFailure(err)) return;
      setError(
        err instanceof ApiClientError && err.statusCode === 429
          ? 'The server is busy answering questions. Try again in a few seconds.'
          : err instanceof ApiClientError
            ? err.message
            : 'Search failed.',
      );
    } finally {
      if (controllerRef.current === controller) setSearching(false);
    }
  };

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
          maxLength={4000}
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
      {error && (
        <p className="search-panel__message" role="alert" data-testid="search-error">
          {error}
        </p>
      )}
      {results !== null && results.length === 0 && (
        <p className="search-panel__message">No passages matched.</p>
      )}
      {results?.map((result) => (
        <button
          key={result.chunk_id}
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
            p. {result.page}
            {result.section ? ` · ${result.section}` : ''}
          </span>
          <span className="search-panel__snippet">
            {result.text.length > SNIPPET_CHARS
              ? `${result.text.slice(0, SNIPPET_CHARS).trimEnd()}…`
              : result.text}
          </span>
        </button>
      ))}
    </>
  );
}

export default SearchPanel;
