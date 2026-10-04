import { memo, useMemo, useRef, useState } from 'react';
import type { DragEvent, RefObject } from 'react';
import type { DocumentIngestResponse, IngestJobState } from '../api/types';
import { formatRelativeTime } from '../utils/relativeTime';
import {
  ACCEPTED_EXTENSIONS,
  DEFAULT_MAX_UPLOAD_BYTES,
  formatBytes,
  validateUploads,
} from '../utils/uploadValidation';
import type { RejectedFile } from '../utils/uploadValidation';
import ConfirmInline from './ConfirmInline';
import './CorpusPanel.css';

// Both are structural details of UploadItem below and have no consumers outside this
// file — kept unexported so the module's public surface is just UploadItem + default.
type UploadItemStatus = 'uploading' | 'success' | 'error';

interface UploadProgress {
  // 'waiting': the server's indexing queue was full (503); retrying after its Retry-After.
  state: IngestJobState | 'waiting';
  chunksDone: number;
  chunksTotal: number;
}

export interface UploadItem {
  id: string;
  filename: string;
  status: UploadItemStatus;
  progress?: UploadProgress;
  result?: DocumentIngestResponse;
  error?: string;
  /** Kept until the upload succeeds or is dismissed, so a failed one can be retried. */
  file?: File;
  /** A re-index of the stored original rather than an upload — retried the same way. */
  reindex?: boolean;
  /** Cancel was pressed and hasn't settled yet. */
  cancelling?: boolean;
}

/** One indexed document's metadata, from GET /documents. */
export interface DocumentMeta {
  chunks: number;
  // Absent for a document indexed before this metadata was stamped at ingest time —
  // the detail line simply omits it.
  pages?: number;
  bytes?: number;
  /** Seconds since the epoch. */
  uploadedAt?: number;
  /** Indexed by an older chunker. */
  stale: boolean;
  /** Has a stored original the server can re-index from (and serve back). */
  reindexable: boolean;
  tags: string[];
}

interface CorpusPanelProps {
  /** Indexed documents, in display order. */
  filenames: string[];
  /** Their metadata. A document indexed moments ago may have none yet (it arrives with
   *  the next refresh); its card shows what it can. */
  documents: Record<string, DocumentMeta>;
  uploads: UploadItem[];
  onUpload: (files: File[]) => Promise<void>;
  onDelete: (filename: string) => Promise<void>;
  maxUploadBytes?: number;
  disabled?: boolean;
  // Lets the context panel's header "Upload" action open this same file dialog.
  browseInputRef?: RefObject<HTMLInputElement | null>;
  onRetryUpload?: (uploadId: string) => void;
  onDismissUpload?: (uploadId: string) => void;
  /** Stop an upload or re-index that is waiting or hasn't started writing yet. */
  onCancelUpload?: (uploadId: string) => void;
  onReindex?: (filename: string) => void;
  /** Re-index every stale document that has a stored original, in one request. */
  onReindexAllStale?: () => void;
  /** Replace a document's tags; rejects with a message to show on failure. */
  onSetTags?: (filename: string, tags: string[]) => Promise<void>;
  /** The read-only API key: no upload or delete (the server would answer 403). */
  readOnly?: boolean;
  /** Download a document's stored original (RAW_DOCUMENT_DIR); absent when none are kept. */
  onOpenOriginal?: (filename: string) => void;
}

interface TagEditorProps {
  filename: string;
  tags: string[];
  disabled?: boolean;
  onSave: (filename: string, tags: string[]) => Promise<void>;
}

/** A document's tags as chips, edited inline as a comma-separated list. */
function TagEditor({ filename, tags, disabled, onSave }: TagEditorProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const save = async () => {
    const next = draft
      .split(',')
      .map((tag) => tag.trim())
      .filter(Boolean);
    setSaving(true);
    setError(null);
    try {
      await onSave(filename, next);
      setEditing(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not save tags.');
    } finally {
      setSaving(false);
    }
  };

  if (editing) {
    return (
      <div className="corpus-panel__tags-edit">
        <input
          type="text"
          className="corpus-panel__tags-input"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault();
              void save();
            } else if (event.key === 'Escape') {
              // Cancels the edit only — not the drawer this panel may sit in.
              event.stopPropagation();
              setEditing(false);
            }
          }}
          placeholder="legal, 2026"
          aria-label={`Tags for ${filename}, comma separated`}
          disabled={saving}
          autoFocus
          data-testid="corpus-tags-input"
        />
        {error && (
          <span className="corpus-panel__tags-error" role="alert">
            {error}
          </span>
        )}
      </div>
    );
  }
  return (
    <div className="corpus-panel__tags" data-testid="corpus-tags">
      {tags.map((tag) => (
        <span key={tag} className="corpus-panel__tag">
          #{tag}
        </span>
      ))}
      <button
        type="button"
        className="corpus-panel__tags-button"
        onClick={() => {
          setDraft(tags.join(', '));
          setError(null);
          setEditing(true);
        }}
        disabled={disabled}
        aria-label={`Edit tags for ${filename}`}
        data-testid="corpus-tags-edit"
      >
        {tags.length > 0 ? '✎' : '+ tag'}
      </button>
    </div>
  );
}

function extensionOf(filename: string): string {
  const dot = filename.lastIndexOf('.');
  return dot === -1 ? '' : filename.slice(dot + 1).toUpperCase();
}

function progressPercent(progress?: UploadProgress): number {
  if (!progress) return 0;
  if (progress.chunksTotal > 0) return Math.round((progress.chunksDone / progress.chunksTotal) * 100);
  return progress.state === 'embedding' ? 50 : 0;
}

interface DocCard {
  key: string;
  filename: string;
  /** Set for cards backed by an upload (in flight or failed) rather than the index. */
  uploadId?: string;
  canRetry?: boolean;
  cancelling?: boolean;
  status: 'indexed' | 'indexing' | 'failed';
  stale?: boolean;
  canReindex?: boolean;
  hasOriginal?: boolean;
  detail: string;
  pct: number;
  error?: string;
}

function CorpusPanel({
  filenames,
  documents,
  uploads,
  onUpload,
  onDelete,
  maxUploadBytes = DEFAULT_MAX_UPLOAD_BYTES,
  disabled,
  browseInputRef,
  onRetryUpload,
  onDismissUpload,
  onCancelUpload,
  onReindex,
  onReindexAllStale,
  onSetTags,
  readOnly = false,
  onOpenOriginal,
}: CorpusPanelProps) {
  const [stagedFiles, setStagedFiles] = useState<File[]>([]);
  const [rejectedFiles, setRejectedFiles] = useState<RejectedFile[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const internalInputRef = useRef<HTMLInputElement>(null);
  const inputRef = browseInputRef ?? internalInputRef;
  const submitting = uploads.some((item) => item.status === 'uploading');

  const handleFilesChange = (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    const { accepted, rejected } = validateUploads(Array.from(fileList), maxUploadBytes);
    setRejectedFiles(rejected);
    setStagedFiles((prev) => [...prev, ...accepted]);
  };

  const removeStagedFile = (index: number) => {
    setStagedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async () => {
    if (stagedFiles.length === 0 || submitting) return;
    const files = stagedFiles;
    setStagedFiles([]);
    setRejectedFiles([]);
    if (inputRef.current) inputRef.current.value = '';
    await onUpload(files);
  };

  const handleDragOver = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    if (!submitting) setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);

  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragOver(false);
    if (submitting) return;
    handleFilesChange(event.dataTransfer.files);
  };

  const handleDelete = async (filename: string) => {
    setDeleting(filename);
    setDeleteError(null);
    try {
      await onDelete(filename);
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : 'Failed to delete document.');
    } finally {
      setDeleting(null);
      setConfirming(null);
    }
  };

  // An upload in flight for an already-indexed name (a re-upload) takes that document's
  // card over, so its progress shows; the index keeps serving the previous version.
  const inFlightNames = new Set(
    uploads.filter((item) => item.status === 'uploading').map((item) => item.filename),
  );
  // A Set, not Array.includes per upload card: with a large corpus that was a quadratic
  // scan on every render.
  const indexed = useMemo(() => new Set(filenames), [filenames]);
  const indexedCards: DocCard[] = filenames.filter((filename) => !inFlightNames.has(filename)).map((filename) => {
    const meta = documents[filename];
    const count = meta?.chunks ?? 0;
    const parts = [`${count} ${count === 1 ? 'chunk' : 'chunks'}`];
    if (meta?.pages) parts.push(`${meta.pages} page${meta.pages === 1 ? '' : 's'}`);
    if (meta?.bytes !== undefined) parts.push(formatBytes(meta.bytes));
    if (meta?.uploadedAt !== undefined) parts.push(formatRelativeTime(meta.uploadedAt * 1000));
    const hasOriginal = meta?.reindexable ?? false;
    const canReindex = onReindex !== undefined && hasOriginal;
    // Only flagged when it can be acted on. Without a stored original there is nothing
    // to re-index from, and a document indexed before versions were recorded counts as
    // stale even if the current chunker produced it — a label the user can't clear.
    const isStale = canReindex && (meta?.stale ?? false);
    if (isStale) parts.push('older chunking');
    return {
      key: `doc:${filename}`,
      filename,
      stale: isStale,
      canReindex,
      hasOriginal,
      status: 'indexed' as const,
      detail: parts.join(' · '),
      pct: 100,
    };
  });

  // In-flight and failed uploads — including re-uploads of an indexed name, whose
  // failure leaves the previous version indexed (that card stays alongside). A
  // successful upload is shown by its indexed card, so it's left out here.
  const uploadCards: DocCard[] = uploads
    .filter((item) => item.status !== 'success')
    .map((item) => {
      const reindexing = indexed.has(item.filename);
      return item.status === 'error'
        ? {
            key: `upload:${item.id}`,
            uploadId: item.id,
            canRetry: item.file !== undefined || item.reindex === true,
            filename: item.filename,
            status: 'failed',
            detail: item.error ?? 'Ingestion failed.',
            pct: 0,
            error: item.error,
          }
        : {
            key: `upload:${item.id}`,
            uploadId: item.id,
            filename: item.filename,
            status: 'indexing',
            cancelling: item.cancelling,
            detail: item.cancelling
              ? 'cancelling…'
              : item.progress?.state === 'waiting'
                ? 'waiting for room in the indexing queue…'
                : `${reindexing ? 're-indexing' : 'indexing'} ${progressPercent(item.progress)}%`,
            pct: progressPercent(item.progress),
          };
    });

  const cards = [...indexedCards, ...uploadCards];
  const trimmedQuery = searchQuery.trim().toLowerCase();
  const filteredCards =
    trimmedQuery === ''
      ? cards
      : cards.filter((card) => card.filename.toLowerCase().includes(trimmedQuery));

  return (
    <>
      {readOnly && (
        <p className="corpus-panel__readonly" data-testid="corpus-readonly">
          Read-only key: you can search and ask, but not change documents.
        </p>
      )}
      {!readOnly && (
      <label
        className={`corpus-panel__dropzone${dragOver ? ' corpus-panel__dropzone--active' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        data-testid="upload-dropzone"
      >
        Drop files &middot; pdf docx md txt html csv
        <input
          ref={inputRef}
          type="file"
          className="corpus-panel__input"
          accept={ACCEPTED_EXTENSIONS.join(',')}
          multiple
          onChange={(event) => handleFilesChange(event.target.files)}
          disabled={submitting}
          data-testid="upload-input"
        />
      </label>
      )}

      {stagedFiles.length > 0 && (
        <div className="corpus-panel__staged" data-testid="upload-staged-list">
          <ul className="corpus-panel__staged-list">
            {stagedFiles.map((file, index) => (
              <li key={`${file.name}-${index}`} className="corpus-panel__staged-item" data-testid="upload-staged-item">
                <span className="corpus-panel__staged-name mono">{file.name}</span>
                <button
                  type="button"
                  className="corpus-panel__staged-remove"
                  onClick={() => removeStagedFile(index)}
                  aria-label={`Remove ${file.name}`}
                  data-testid="upload-staged-remove"
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
          <button
            type="button"
            className="corpus-panel__submit"
            onClick={() => void handleSubmit()}
            disabled={submitting}
            data-testid="upload-button"
          >
            {submitting ? 'Uploading…' : stagedFiles.length > 1 ? `Upload ${stagedFiles.length} files` : 'Upload'}
          </button>
        </div>
      )}

      {rejectedFiles.map((rejected) => (
        <p key={rejected.filename} className="corpus-panel__reject" role="alert" data-testid="upload-size-error">
          {rejected.message}
        </p>
      ))}

      {deleteError && (
        <div className="corpus-panel__error" role="alert">
          {deleteError}
        </div>
      )}

      {/* One card's ↻ is enough for one document; after a chunker upgrade every
          document is stale at once, and clicking each in turn doesn't scale. */}
      {onReindexAllStale && indexedCards.filter((card) => card.stale).length > 1 && (
        <button
          type="button"
          className="corpus-panel__reindex-all"
          onClick={onReindexAllStale}
          disabled={disabled}
          data-testid="reindex-all-stale"
        >
          ↻ Re-index {indexedCards.filter((card) => card.stale).length} documents with older chunking
        </button>
      )}

      {cards.length > 0 && (
        <input
          type="search"
          className="corpus-panel__search"
          placeholder="Filter documents…"
          aria-label="Filter documents"
          value={searchQuery}
          onChange={(event) => setSearchQuery(event.target.value)}
          data-testid="corpus-panel-search"
        />
      )}

      {trimmedQuery !== '' && filteredCards.length === 0 && (
        <p className="corpus-panel__empty-filter">No documents match "{searchQuery.trim()}".</p>
      )}

      {filteredCards.map((card) => (
        <article key={card.key} className="corpus-panel__card" data-testid="corpus-panel-item">
          <div className="corpus-panel__card-top">
            <span className="corpus-panel__ext">{extensionOf(card.filename)}</span>
            <span className="corpus-panel__name">{card.filename}</span>
            {confirming === card.key ? (
              <ConfirmInline
                className="corpus-panel__confirm"
                prompt="Delete?"
                confirmLabel="Confirm"
                onCancel={() => setConfirming(null)}
                onConfirm={() => void handleDelete(card.filename)}
                busy={deleting === card.filename}
              />
            ) : card.status === 'indexed' ? (
              <div className="corpus-panel__card-actions">
                {card.hasOriginal && onOpenOriginal && (
                  <button
                    type="button"
                    className="corpus-panel__open-original"
                    onClick={() => onOpenOriginal(card.filename)}
                    aria-label={`Download the original of ${card.filename}`}
                    title="Download the uploaded file"
                  >
                    ⇱
                  </button>
                )}
                {card.canReindex && (
                  <button
                    type="button"
                    className={`corpus-panel__reindex${card.stale ? ' corpus-panel__reindex--stale' : ''}`}
                    onClick={() => onReindex?.(card.filename)}
                    disabled={disabled}
                    aria-label={`Re-index ${card.filename}`}
                    title={
                      card.stale
                        ? 'Indexed by an older chunker — re-index from the stored original'
                        : 'Re-index from the stored original'
                    }
                  >
                    ↻
                  </button>
                )}
                {!readOnly && (
                  <button
                    type="button"
                    className="corpus-panel__remove"
                    onClick={() => setConfirming(card.key)}
                    disabled={disabled}
                    aria-label={`Delete ${card.filename}`}
                  >
                    ✕
                  </button>
                )}
              </div>
            ) : card.status === 'indexing' ? (
              !readOnly &&
              onCancelUpload &&
              card.uploadId !== undefined && (
                <button
                  type="button"
                  className="corpus-panel__cancel"
                  onClick={() => onCancelUpload(card.uploadId!)}
                  disabled={card.cancelling}
                  aria-label={`Cancel indexing ${card.filename}`}
                  title="Stop before anything is written to the index"
                  data-testid="corpus-cancel-upload"
                >
                  Cancel
                </button>
              )
            ) : (
              card.status === 'failed' &&
              card.uploadId !== undefined && (
                <div className="corpus-panel__failed-actions">
                  {card.canRetry && onRetryUpload && (
                    <button
                      type="button"
                      className="corpus-panel__retry"
                      onClick={() => onRetryUpload(card.uploadId!)}
                      disabled={disabled}
                      aria-label={`Retry ${card.filename}`}
                    >
                      Retry
                    </button>
                  )}
                  {onDismissUpload && (
                    <button
                      type="button"
                      className="corpus-panel__remove"
                      onClick={() => onDismissUpload(card.uploadId!)}
                      aria-label={`Dismiss ${card.filename}`}
                    >
                      ✕
                    </button>
                  )}
                </div>
              )
            )}
          </div>
          <div className="corpus-panel__status">
            <span className={`corpus-panel__dot corpus-panel__dot--${card.status}`} aria-hidden="true" />
            {card.status === 'failed' ? (
              <span role="status">{`failed — ${card.detail}`}</span>
            ) : (
              <span>{card.detail}</span>
            )}
          </div>
          {card.status === 'indexing' && (
            <div
              className="corpus-panel__bar"
              role="progressbar"
              aria-label={`Indexing ${card.filename}`}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={card.pct}
            >
              <div className="corpus-panel__bar-fill" style={{ width: `${card.pct}%` }} />
            </div>
          )}
          {card.status === 'indexed' && onSetTags && (
            <TagEditor
              filename={card.filename}
              tags={documents[card.filename]?.tags ?? []}
              disabled={disabled}
              onSave={onSetTags}
            />
          )}
        </article>
      ))}
    </>
  );
}

// Memoized: App re-renders on every streamed token, and nothing here changes then.
export default memo(CorpusPanel);
