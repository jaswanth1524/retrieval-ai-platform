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
}

interface CorpusPanelProps {
  filenames: string[];
  chunkCounts: Record<string, number>;
  // Absent (not present with a null value) for a filename ingested before this
  // metadata was stamped at ingest time — the detail line simply omits it.
  pageCounts?: Record<string, number>;
  byteSizes?: Record<string, number>;
  uploadedAts?: Record<string, number>;
  uploads: UploadItem[];
  onUpload: (files: File[]) => Promise<void>;
  onDelete: (filename: string) => Promise<void>;
  maxUploadBytes?: number;
  disabled?: boolean;
  // Lets the context panel's header "Upload" action open this same file dialog.
  browseInputRef?: RefObject<HTMLInputElement | null>;
  onRetryUpload?: (uploadId: string) => void;
  onDismissUpload?: (uploadId: string) => void;
  /** Indexed by an older chunker. */
  staleFilenames?: string[];
  /** Have a stored original the server can re-index from. */
  reindexableFilenames?: string[];
  onReindex?: (filename: string) => void;
  /** Re-index every stale document that has a stored original, in one request. */
  onReindexAllStale?: () => void;
  /** Each tagged document's tags. */
  tags?: Record<string, string[]>;
  /** Replace a document's tags; rejects with a message to show on failure. */
  onSetTags?: (filename: string, tags: string[]) => Promise<void>;
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
  status: 'indexed' | 'indexing' | 'failed';
  stale?: boolean;
  canReindex?: boolean;
  detail: string;
  pct: number;
  error?: string;
}

function CorpusPanel({
  filenames,
  chunkCounts,
  pageCounts = {},
  byteSizes = {},
  uploadedAts = {},
  uploads,
  onUpload,
  onDelete,
  maxUploadBytes = DEFAULT_MAX_UPLOAD_BYTES,
  disabled,
  browseInputRef,
  onRetryUpload,
  onDismissUpload,
  staleFilenames = [],
  reindexableFilenames = [],
  onReindex,
  onReindexAllStale,
  tags = {},
  onSetTags,
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
  // Sets, not Array.includes per card: with a large corpus that was a quadratic scan on
  // every render.
  const reindexable = useMemo(() => new Set(reindexableFilenames), [reindexableFilenames]);
  const stale = useMemo(() => new Set(staleFilenames), [staleFilenames]);
  const indexed = useMemo(() => new Set(filenames), [filenames]);
  const indexedCards: DocCard[] = filenames.filter((filename) => !inFlightNames.has(filename)).map((filename) => {
    const count = chunkCounts[filename] ?? 0;
    const parts = [`${count} ${count === 1 ? 'chunk' : 'chunks'}`];
    const pages = pageCounts[filename];
    if (pages) parts.push(`${pages} page${pages === 1 ? '' : 's'}`);
    const bytes = byteSizes[filename];
    if (bytes !== undefined) parts.push(formatBytes(bytes));
    const uploadedAt = uploadedAts[filename];
    if (uploadedAt !== undefined) parts.push(formatRelativeTime(uploadedAt * 1000));
    const canReindex = onReindex !== undefined && reindexable.has(filename);
    // Only flagged when it can be acted on. Without a stored original there is nothing
    // to re-index from, and a document indexed before versions were recorded counts as
    // stale even if the current chunker produced it — a label the user can't clear.
    const isStale = canReindex && stale.has(filename);
    if (isStale) parts.push('older chunking');
    return {
      key: `doc:${filename}`,
      filename,
      stale: isStale,
      canReindex,
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
            detail:
              item.progress?.state === 'waiting'
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
              <div
                className="corpus-panel__confirm"
                onKeyDown={(event) => {
                  if (event.key !== 'Escape') return;
                  // Cancels the confirm only — not the drawer this panel may sit in.
                  event.stopPropagation();
                  setConfirming(null);
                }}
              >
                <span>Delete?</span>
                <button
                  type="button"
                  className="corpus-panel__confirm-yes"
                  onClick={() => void handleDelete(card.filename)}
                  disabled={deleting === card.filename}
                >
                  {deleting === card.filename ? '…' : 'Confirm'}
                </button>
                {/* Focus the safe choice, as ConversationList does: the ✕ that opened
                    this confirm just unmounted, which dropped focus to <body>. */}
                <button
                  type="button"
                  className="corpus-panel__confirm-no"
                  autoFocus
                  onClick={() => setConfirming(null)}
                >
                  Cancel
                </button>
              </div>
            ) : card.status === 'indexed' ? (
              <div className="corpus-panel__card-actions">
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
                <button
                  type="button"
                  className="corpus-panel__remove"
                  onClick={() => setConfirming(card.key)}
                  disabled={disabled}
                  aria-label={`Delete ${card.filename}`}
                >
                  ✕
                </button>
              </div>
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
              tags={tags[card.filename] ?? []}
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
