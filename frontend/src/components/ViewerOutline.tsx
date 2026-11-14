import { useEffect, useState } from 'react';
import { api, apiErrorMessage } from '../api/client';
import type { DocumentOutlineResponse } from '../api/types';

/** A document's contents (its sections, in order) and "go to page", from the server's
 *  outline: the viewer holds 30 passages at a time, so it can't list them itself. */
export default function ViewerOutline({
  filename,
  onGoTo,
}: {
  filename: string;
  onGoTo: (chunkId: string) => void;
}) {
  const [outline, setOutline] = useState<DocumentOutlineResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    api
      .documentOutline(filename, controller.signal)
      .then(setOutline)
      .catch((err) => {
        if (!controller.signal.aborted) setError(apiErrorMessage(err, 'Could not read the contents.'));
      });
    return () => controller.abort();
  }, [filename]);

  if (error) {
    return (
      <p className="document-viewer__notice" role="alert">
        {error}
      </p>
    );
  }
  if (!outline) return <p className="document-viewer__notice">Loading the contents…</p>;
  const lastPage = outline.pages.at(-1)?.page ?? 1;
  const goToPage = () => {
    const wanted = Number(page);
    if (!Number.isInteger(wanted)) return;
    // The last page starting at or before the one asked for (a page with no passage
    // of its own is shown with the passage that runs onto it).
    const target = [...outline.pages].reverse().find((entry) => entry.page <= wanted) ?? outline.pages[0];
    if (target) onGoTo(target.chunk_id);
  };
  return (
    <nav className="document-viewer__outline" aria-label="Contents" data-testid="viewer-outline">
      {outline.pages.length > 1 && (
        <form
          className="document-viewer__goto"
          onSubmit={(event) => {
            event.preventDefault();
            goToPage();
          }}
        >
          <label>
            Go to page{' '}
            <input
              type="number"
              min={1}
              max={lastPage}
              inputMode="numeric"
              value={page}
              onChange={(event) => setPage(event.target.value)}
              data-testid="viewer-goto-page"
            />
          </label>{' '}
          <button type="submit" disabled={!page} data-testid="viewer-goto-submit">
            Go
          </button>{' '}
          <span className="mono">of {lastPage}</span>
        </form>
      )}
      {outline.sections.length > 1 && (
        <ol className="document-viewer__outline-list">
          {outline.sections.map((entry) => (
            <li key={entry.chunk_id}>
              <button
                type="button"
                onClick={() => onGoTo(entry.chunk_id)}
                data-testid="viewer-outline-entry"
              >
                <span dir="auto">{entry.section || '(untitled)'}</span>{' '}
                <span className="mono">p. {entry.page}</span>
              </button>
            </li>
          ))}
        </ol>
      )}
      {outline.truncated && <p className="document-viewer__notice">The first 1000 sections.</p>}
    </nav>
  );
}
