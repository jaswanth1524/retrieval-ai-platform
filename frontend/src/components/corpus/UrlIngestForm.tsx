import { useState } from 'react';

interface UrlIngestFormProps {
  disabled?: boolean;
  /** Starts indexing the URL's document (a card follows it like an upload). */
  onSubmit: (url: string) => void;
}

/** "From URL": the server fetches the document (URL_INGEST_ENABLED). Only http(s)
 *  is accepted here; the server applies the real checks (public addresses, size). */
export default function UrlIngestForm({ disabled, onSubmit }: UrlIngestFormProps) {
  const [url, setUrl] = useState('');
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
        onSubmit(trimmed);
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
        Add URL
      </button>
      {error && (
        <p id="url-ingest-error" className="corpus-panel__reject" role="alert">
          {error}
        </p>
      )}
    </form>
  );
}
