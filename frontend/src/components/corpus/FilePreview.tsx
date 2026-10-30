import { useRef, useState } from 'react';
import { api, apiErrorMessage } from '../../api/client';
import type { DocumentPreviewResponse } from '../../api/types';
import { pageLabel } from '../../utils/pages';

interface FilePreviewProps {
  /** Extensions the server reads (for the file picker). */
  accept: readonly string[];
  workspace: string;
  disabled?: boolean;
}

// Passages shown from the preview (the server sends up to 20).
const SHOWN_PASSAGES = 5;

/** "Preview a file…": how the server would read a file — its sections and first
 *  passages (POST /documents/preview) — without indexing it. Useful before uploading
 *  a scanned PDF or a spreadsheet, to see what the parser makes of it. */
export default function FilePreview({ accept, workspace, disabled = false }: FilePreviewProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<DocumentPreviewResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const read = (file: File) => {
    setLoading(true);
    setError(null);
    setPreview(null);
    api.previewDocument(file, undefined, workspace).then(
      (result) => {
        setPreview(result);
        setLoading(false);
      },
      (err) => {
        setError(apiErrorMessage(err, `Could not preview ${file.name}.`));
        setLoading(false);
      },
    );
  };

  return (
    <>
      <button
        type="button"
        className="corpus-panel__folder-button"
        onClick={() => inputRef.current?.click()}
        disabled={disabled || loading}
        data-testid="preview-file"
      >
        {loading ? 'Reading…' : 'Preview a file…'}
      </button>
      <input
        ref={inputRef}
        type="file"
        className="corpus-panel__folder-input"
        accept={accept.join(',')}
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = '';
          if (file) read(file);
        }}
        aria-hidden="true"
        tabIndex={-1}
        data-testid="preview-file-input"
      />
      {error && (
        <p className="corpus-panel__reject" role="alert" data-testid="preview-error">
          {error}
        </p>
      )}
      {preview && (
        <section
          className="corpus-panel__preview"
          aria-label={`Preview of ${preview.filename}`}
          data-testid="preview-result"
        >
          <p className="corpus-panel__preview-head mono">
            <span>
              {preview.filename}: {preview.chunk_count} passage{preview.chunk_count === 1 ? '' : 's'} in{' '}
              {preview.section_count} section{preview.section_count === 1 ? '' : 's'}
              {preview.page_count > 1 ? `, ${preview.page_count} pages` : ''} — not indexed
            </span>
            <button
              type="button"
              className="corpus-panel__job-cancel"
              onClick={() => setPreview(null)}
              aria-label="Close the preview"
            >
              close
            </button>
          </p>
          <ol className="corpus-panel__preview-list">
            {preview.first_chunks.slice(0, SHOWN_PASSAGES).map((chunk) => (
              <li key={chunk.ordinal} dir="auto">
                <span className="mono">
                  {pageLabel(chunk.page, chunk.page_end)} · {chunk.section}
                </span>
                <span className="corpus-panel__preview-text">{chunk.text}</span>
              </li>
            ))}
          </ol>
        </section>
      )}
    </>
  );
}
