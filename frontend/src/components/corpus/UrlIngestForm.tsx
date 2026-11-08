import { useState } from 'react';

interface UrlIngestFormProps {
  disabled?: boolean;
  /** Starts indexing the URL's document (a card follows it like an upload). */
  onSubmit: (url: string) => void;
  /** The URL is a sitemap or feed: its pages are added in the background. */
  onSubmitSitemap?: (url: string) => void;
}

/** "From URL": the server fetches the document (URL_INGEST_ENABLED). Only http(s)
 *  is accepted here; the server applies the real checks (public addresses, size). */
export default function UrlIngestForm({ disabled, onSubmit, onSubmitSitemap }: UrlIngestFormProps) {
  const [url, setUrl] = useState('');
  const [sitemap, setSitemap] = useState(false);
  const [error, setError] = useState<string | null>(null);

  return (
    <form
      className="corpus-panel__url"
      onSubmit={(event) => {
        event.preventDefault();
        const trimmed = url.trim();
        if (!/^https?:\/\/\S+$/i.test(trimmed)) {
          setError('Enter an http:// or https:// address.');
          return;
        }
        setError(null);
        setUrl('');
        if (sitemap && onSubmitSitemap) onSubmitSitemap(trimmed);
        else onSubmit(trimmed);
      }}
      data-testid="url-ingest-form"
    >
      <input
        type="url"
        className="corpus-panel__url-input"
        value={url}
        onChange={(event) => setUrl(event.target.value)}
        placeholder="https://…"
        title="A page or file for the server to fetch and index"
        aria-label="Document URL to index"
        aria-invalid={error !== null}
        aria-describedby={error ? 'url-ingest-error' : undefined}
        disabled={disabled}
        data-testid="url-ingest-input"
      />
      <button
        type="submit"
        className="corpus-panel__bulk-button"
        disabled={disabled || !url.trim()}
        data-testid="url-ingest-submit"
      >
        {sitemap ? 'Add its pages' : 'Add URL'}
      </button>
      {onSubmitSitemap && (
        <label className="corpus-panel__url-sitemap" title="A sitemap, sitemap index, RSS or Atom feed: every page it lists on that site">
          <input
            type="checkbox"
            checked={sitemap}
            onChange={(event) => setSitemap(event.target.checked)}
            disabled={disabled}
            data-testid="url-ingest-sitemap"
          />
          sitemap or feed
        </label>
      )}
      {error && (
        <p id="url-ingest-error" className="corpus-panel__reject" role="alert">
          {error}
        </p>
      )}
    </form>
  );
}
