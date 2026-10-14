import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { DragEvent, RefObject } from 'react';
import type { DocumentIngestResponse, IngestJobState } from '../api/types';
import { useStableHandler } from '../hooks/useStableHandler';
import { formatRelativeTime } from '../utils/relativeTime';
import { readStored, writeStored } from '../utils/safeStorage';
import {
  acceptedExtensions,
  DEFAULT_MAX_UPLOAD_BYTES,
  extensionsLabel,
  formatBytes,
  validateUploads,
} from '../utils/uploadValidation';
import type { RejectedFile } from '../utils/uploadValidation';
import ConfirmInline from './ConfirmInline';
import './CorpusPanel.css';
import { isImeComposing } from '../utils/keyboard';

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
  /** Upload suffixes the server accepts; the built-in list when absent. */
  supportedExtensions?: readonly string[];
  disabled?: boolean;
  // Lets the context panel's header "Upload" action open this same file dialog.
  browseInputRef?: RefObject<HTMLInputElement | null>;
  onRetryUpload?: (uploadId: string) => void;
  onDismissUpload?: (uploadId: string) => void;
  /** Stop an upload or re-index that is waiting or hasn't started writing yet. */
  onCancelUpload?: (uploadId: string) => void;
  onReindex?: (filename: string) => void;
  /** Re-index several documents at once (the bulk bar): one listing refresh at the end.
   *  Without it the bulk bar calls `onReindex` per document. */
  onReindexMany?: (filenames: string[]) => void;
  /** Re-index every stale document that has a stored original, in one request. */
  onReindexAllStale?: () => void;
  /** Replace a document's tags; rejects with a message to show on failure. */
  onSetTags?: (filename: string, tags: string[]) => Promise<void>;
  /** The read-only API key: no upload or delete (the server would answer 403). */
  readOnly?: boolean;
  /** Download a document's stored original (RAW_DOCUMENT_DIR); absent when none are kept. */
  onOpenOriginal?: (filename: string) => void;
  /** Open a document's indexed text in the viewer, from its first passage. */
  onRead?: (filename: string) => void;
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
  // Closing the editor unmounts the control that had focus (the field, Save, Cancel);
  // focus goes back to the button that opened it rather than to <body>.
  const editButtonRef = useRef<HTMLButtonElement>(null);
  const refocusRef = useRef(false);
  const close = () => {
    refocusRef.current = true;
    setEditing(false);
  };
  useEffect(() => {
    if (editing || !refocusRef.current) return;
    refocusRef.current = false;
    editButtonRef.current?.focus();
  }, [editing]);

  const save = async () => {
    const next = draft
      .split(',')
      .map((tag) => tag.trim())
      .filter(Boolean);
    setSaving(true);
    setError(null);
    try {
      await onSave(filename, next);
      close();
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
            if (isImeComposing(event)) return;
            if (event.key === 'Enter') {
              event.preventDefault();
              void save();
            } else if (event.key === 'Escape') {
              // Cancels the edit only — not the drawer this panel may sit in.
              event.stopPropagation();
              close();
            }
          }}
          placeholder="legal, 2026"
          aria-label={`Tags for ${filename}, comma separated`}
          disabled={saving}
          autoFocus
          data-testid="corpus-tags-input"
        />
        {/* Enter and Esc did this alone: a mouse or touch user had no way to save. */}
        <div className="corpus-panel__tags-actions">
          <button
            type="button"
            className="corpus-panel__tags-button"
            onClick={() => void save()}
            disabled={saving}
            data-testid="corpus-tags-save"
          >
            Save
          </button>
          <button
            type="button"
            className="corpus-panel__tags-button"
            onClick={close}
            disabled={saving}
            data-testid="corpus-tags-cancel"
          >
            Cancel
          </button>
        </div>
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
        ref={editButtonRef}
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
  tags: string[];
  bytes?: number;
  uploadedAt?: number;
}

// 'listed': the order the corpus arrives in (the server's listing, then this session's
// uploads), as the panel always showed it.
type SortKey = 'listed' | 'name' | 'newest' | 'largest';
const SORT_LABELS: Record<SortKey, string> = {
  listed: 'Default',
  name: 'Name',
  newest: 'Newest',
  largest: 'Largest',
};
const SORT_STORAGE_KEY = 'docrag-corpus-sort';
// api/documents.py's MAX_TAGS_PER_DOCUMENT: a bulk "add tag" past it is refused there.
const MAX_TAGS_PER_DOCUMENT = 20;
// Shared, so a card with no tags doesn't get a fresh [] (and a re-render) every time.
const NO_TAGS: string[] = [];

function loadSort(): SortKey {
  const stored = readStored(SORT_STORAGE_KEY);
  return stored === 'name' || stored === 'newest' || stored === 'largest' ? stored : 'listed';
}

function compareCards(sort: SortKey): ((a: DocCard, b: DocCard) => number) | null {
  if (sort === 'listed') return null;
  const byName = (a: DocCard, b: DocCard) =>
    a.filename.localeCompare(b.filename, undefined, { sensitivity: 'base' });
  // Documents indexed before sizes and times were recorded have neither: listed last.
  if (sort === 'newest') {
    return (a, b) => (b.uploadedAt ?? -Infinity) - (a.uploadedAt ?? -Infinity) || byName(a, b);
  }
  if (sort === 'largest') {
    return (a, b) => (b.bytes ?? -Infinity) - (a.bytes ?? -Infinity) || byName(a, b);
  }
  return byName;
}

interface DocumentCardProps {
  card: DocCard;
  disabled?: boolean;
  readOnly: boolean;
  confirming: boolean;
  deleting: boolean;
  selectable: boolean;
  selected: boolean;
  onToggleSelect: (filename: string) => void;
  onAskDelete: (key: string) => void;
  onCancelDelete: () => void;
  onConfirmDelete: (filename: string) => void;
  onRead?: (filename: string) => void;
  onOpenOriginal?: (filename: string) => void;
  onReindex?: (filename: string) => void;
  onCancelUpload?: (uploadId: string) => void;
  onRetryUpload?: (uploadId: string) => void;
  onDismissUpload?: (uploadId: string) => void;
  onSetTags?: (filename: string, tags: string[]) => Promise<void>;
}

/** One document (or upload) card. Memoized with stable card objects and callbacks, so
 *  an upload's progress poll re-renders that card, not every card in a large corpus. */
const DocumentCard = memo(function DocumentCard({
  card,
  disabled,
  readOnly,
  confirming,
  deleting,
  selectable,
  selected,
  onToggleSelect,
  onAskDelete,
  onCancelDelete,
  onConfirmDelete,
  onRead,
  onOpenOriginal,
  onReindex,
  onCancelUpload,
  onRetryUpload,
  onDismissUpload,
  onSetTags,
}: DocumentCardProps) {
  return (
    <article
      className={`corpus-panel__card${selected ? ' corpus-panel__card--selected' : ''}`}
      data-testid="corpus-panel-item"
      data-corpus-card={card.filename}
    >
      <div className="corpus-panel__card-top">
        {selectable && (
          <input
            type="checkbox"
            className="corpus-panel__select"
            checked={selected}
            onChange={() => onToggleSelect(card.filename)}
            disabled={disabled}
            aria-label={`Select ${card.filename}`}
            data-testid="corpus-select"
          />
        )}
        <span className="corpus-panel__ext">{extensionOf(card.filename)}</span>
        <span className="corpus-panel__name">{card.filename}</span>
        {confirming ? (
          <ConfirmInline
            className="corpus-panel__confirm"
            prompt="Delete?"
            confirmLabel="Confirm"
            onCancel={onCancelDelete}
            onConfirm={() => onConfirmDelete(card.filename)}
            busy={deleting}
          />
        ) : card.status === 'indexed' ? (
          <div className="corpus-panel__card-actions">
            {onRead && (
              <button
                type="button"
                className="corpus-panel__read"
                onClick={() => onRead(card.filename)}
                aria-label={`Read ${card.filename}`}
                title="Read the indexed text"
                data-testid="corpus-read"
              >
                ☰
              </button>
            )}
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
                onClick={() => onAskDelete(card.key)}
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
        <TagEditor filename={card.filename} tags={card.tags} disabled={disabled} onSave={onSetTags} />
      )}
    </article>
  );
});

interface BulkBarProps {
  selected: string[];
  disabled?: boolean;
  /** Names among the selection with a stored original (re-indexable). */
  reindexable: string[];
  onReindex?: (filenames: string[]) => void;
  onAddTag?: (tag: string) => Promise<void>;
  onDelete: () => Promise<void>;
  onClear: () => void;
}

/** What to do with every selected document at once. */
function BulkBar({ selected, disabled, reindexable, onReindex, onAddTag, onDelete, onClear }: BulkBarProps) {
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [busy, setBusy] = useState(false);
  const [tagDraft, setTagDraft] = useState('');
  const count = selected.length;
  const label = `${count} document${count === 1 ? '' : 's'}`;

  const run = async (work: () => Promise<void>) => {
    setBusy(true);
    try {
      await work();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="corpus-panel__bulk" role="group" aria-label="Selected documents" data-testid="corpus-bulk">
      <span className="corpus-panel__bulk-count mono">{count} selected</span>
      {confirmingDelete ? (
        <ConfirmInline
          className="corpus-panel__confirm"
          prompt={`Delete ${label}?`}
          confirmLabel="Delete"
          onCancel={() => setConfirmingDelete(false)}
          onConfirm={() =>
            void run(async () => {
              await onDelete();
              setConfirmingDelete(false);
            })
          }
          busy={busy}
          confirmTestId="corpus-bulk-delete-confirm"
        />
      ) : (
        <>
          {onAddTag && (
            <form
              className="corpus-panel__bulk-tag"
              onSubmit={(event) => {
                event.preventDefault();
                const tag = tagDraft.trim();
                if (!tag) return;
                void run(async () => {
                  await onAddTag(tag);
                  setTagDraft('');
                });
              }}
            >
              <input
                type="text"
                className="corpus-panel__bulk-input"
                value={tagDraft}
                onChange={(event) => setTagDraft(event.target.value)}
                placeholder="tag"
                aria-label={`Tag to add to ${label}`}
                disabled={disabled || busy}
                data-testid="corpus-bulk-tag-input"
              />
              <button
                type="submit"
                className="corpus-panel__bulk-button"
                disabled={disabled || busy || !tagDraft.trim()}
                data-testid="corpus-bulk-tag"
              >
                + tag
              </button>
            </form>
          )}
          {onReindex && reindexable.length > 0 && (
            <button
              type="button"
              className="corpus-panel__bulk-button"
              onClick={() => onReindex(reindexable)}
              disabled={disabled || busy}
              data-testid="corpus-bulk-reindex"
            >
              ↻ Re-index {reindexable.length}
            </button>
          )}
          <button
            type="button"
            className="corpus-panel__bulk-button corpus-panel__bulk-delete"
            onClick={() => setConfirmingDelete(true)}
            disabled={disabled || busy}
            data-testid="corpus-bulk-delete"
          >
            Delete {count}
          </button>
          <button
            type="button"
            className="corpus-panel__bulk-button"
            onClick={onClear}
            disabled={busy}
            data-testid="corpus-bulk-clear"
          >
            Clear
          </button>
        </>
      )}
    </div>
  );
}

function CorpusPanel({
  filenames,
  documents,
  uploads,
  onUpload,
  onDelete,
  maxUploadBytes = DEFAULT_MAX_UPLOAD_BYTES,
  supportedExtensions,
  disabled,
  browseInputRef,
  onRetryUpload,
  onDismissUpload,
  onCancelUpload,
  onReindex,
  onReindexMany,
  onReindexAllStale,
  onSetTags,
  readOnly = false,
  onOpenOriginal,
  onRead,
}: CorpusPanelProps) {
  const [stagedFiles, setStagedFiles] = useState<File[]>([]);
  const [rejectedFiles, setRejectedFiles] = useState<RejectedFile[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [tagFilter, setTagFilter] = useState<string | null>(null);
  const [sort, setSort] = useState<SortKey>(loadSort);
  const [selected, setSelected] = useState<ReadonlySet<string>>(() => new Set());
  const internalInputRef = useRef<HTMLInputElement>(null);
  const inputRef = browseInputRef ?? internalInputRef;
  const cardsRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  // Where focus goes after a confirmed delete: the card now in the deleted one's place.
  // Its Confirm button unmounts with it, which otherwise dropped focus to <body>.
  const refocusIndexRef = useRef<number | null>(null);
  // Files of this browser's still being sent or indexed — not re-indexes or a backup's
  // documents (no file of their own), which used to disable the upload input, so ⌘U and
  // the Upload buttons silently did nothing for as long as a re-index ran.
  const submitting = uploads.some((item) => item.status === 'uploading' && item.file !== undefined);

  const handleFilesChange = (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    const { accepted, rejected } = validateUploads(
      Array.from(fileList),
      maxUploadBytes,
      acceptedExtensions(supportedExtensions),
    );
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
    setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);

  // Staged even while an upload runs (Upload waits for it): the drop used to be
  // swallowed silently — marked handled, so the page-wide drop zone skipped it too.
  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragOver(false);
    handleFilesChange(event.dataTransfer.files);
  };

  const handleDelete = useStableHandler(async (filename: string) => {
    const cards = [...(cardsRef.current?.querySelectorAll<HTMLElement>('[data-corpus-card]') ?? [])];
    const index = cards.findIndex((card) => card.dataset.corpusCard === filename);
    setDeleting(filename);
    setDeleteError(null);
    // Set before the delete: the listing that drops the card can land while it runs.
    refocusIndexRef.current = index;
    try {
      await onDelete(filename);
    } catch (err) {
      refocusIndexRef.current = null;
      setDeleteError(err instanceof Error ? err.message : 'Failed to delete document.');
    } finally {
      setDeleting(null);
      setConfirming(null);
    }
  });
  const cancelDelete = useCallback(() => setConfirming(null), []);
  const toggleSelect = useCallback((filename: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (!next.delete(filename)) next.add(filename);
      return next;
    });
  }, []);

  // An upload in flight for an already-indexed name (a re-upload) takes that document's
  // card over, so its progress shows; the index keeps serving the previous version.
  // Joined into a string, so the memo below survives the polls that don't change it.
  const inFlightKey = uploads
    .filter((item) => item.status === 'uploading')
    .map((item) => item.filename)
    .sort()
    .join('\n');
  // A Set, not Array.includes per upload card: with a large corpus that was a quadratic
  // scan on every render.
  const indexed = useMemo(() => new Set(filenames), [filenames]);
  // Memoized, so an upload's progress poll keeps every indexed card's object (and so
  // its memoized DocumentCard) as it was.
  const indexedCards: DocCard[] = useMemo(() => {
    const inFlight = new Set(inFlightKey ? inFlightKey.split('\n') : []);
    return filenames
      .filter((filename) => !inFlight.has(filename))
      .map((filename) => {
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
          tags: meta?.tags.length ? meta.tags : NO_TAGS,
          bytes: meta?.bytes,
          uploadedAt: meta?.uploadedAt,
        };
      });
  }, [filenames, documents, inFlightKey, onReindex]);

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
            tags: NO_TAGS,
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
            tags: NO_TAGS,
          };
    });

  const availableTags = useMemo(
    () =>
      [...new Set(indexedCards.flatMap((card) => card.tags))].sort((a, b) =>
        a.localeCompare(b, undefined, { sensitivity: 'base' }),
      ),
    [indexedCards],
  );
  // A filter on a tag nobody carries any more would hide every card with no way back.
  const activeTag = tagFilter !== null && availableTags.includes(tagFilter) ? tagFilter : null;
  const trimmedQuery = searchQuery.trim().toLowerCase();
  const visibleIndexed = useMemo(() => {
    const matching = indexedCards.filter(
      (card) =>
        (trimmedQuery === '' || card.filename.toLowerCase().includes(trimmedQuery)) &&
        (activeTag === null || card.tags.includes(activeTag)),
    );
    const compare = compareCards(sort);
    return compare ? matching.sort(compare) : matching;
  }, [indexedCards, trimmedQuery, activeTag, sort]);
  const visibleUploads =
    trimmedQuery === ''
      ? uploadCards
      : uploadCards.filter((card) => card.filename.toLowerCase().includes(trimmedQuery));
  const filteredCards = [...visibleIndexed, ...visibleUploads];
  const cardCount = indexedCards.length + uploadCards.length;

  // Selection is of the indexed documents on screen: one deleted (here or in another
  // tab) leaves it, and so does one the search or tag filter hides — "Delete 3" must
  // never include a document the user can't see.
  const selectedNames = useMemo(
    () => visibleIndexed.filter((card) => selected.has(card.filename)).map((card) => card.filename),
    [visibleIndexed, selected],
  );
  useEffect(() => {
    setSelected((prev) => {
      if (prev.size === 0) return prev;
      const visible = new Set(visibleIndexed.map((card) => card.filename));
      const kept = [...prev].filter((filename) => visible.has(filename));
      return kept.length === prev.size ? prev : new Set(kept);
    });
  }, [visibleIndexed]);
  const canBulk = !readOnly && indexedCards.length > 1;
  const visibleNames = visibleIndexed.map((card) => card.filename);
  const allVisibleSelected =
    visibleNames.length > 0 && visibleNames.every((filename) => selected.has(filename));

  const changeSort = (next: SortKey) => {
    setSort(next);
    writeStored(SORT_STORAGE_KEY, next);
  };

  useEffect(() => {
    const index = refocusIndexRef.current;
    if (index === null) return;
    refocusIndexRef.current = null;
    const cards = cardsRef.current?.querySelectorAll<HTMLElement>('[data-corpus-card]');
    const next = cards && cards.length > 0 ? cards[Math.min(index, cards.length - 1)] : null;
    (next?.querySelector<HTMLElement>('button, input') ?? searchRef.current)?.focus();
  }, [filenames]);

  const bulkDelete = async () => {
    const failed: string[] = [];
    // One at a time: each delete takes that document's write lock on the server.
    for (const filename of selectedNames) {
      try {
        await onDelete(filename);
      } catch {
        failed.push(filename);
      }
    }
    setSelected(new Set(failed));
    setDeleteError(failed.length > 0 ? `Couldn't delete: ${failed.join(', ')}` : null);
    // The bulk bar (and its focused Confirm) unmounts when nothing is left selected;
    // without this, focus fell to <body>.
    if (failed.length === 0) searchRef.current?.focus();
  };

  const bulkAddTag = async (tag: string) => {
    if (!onSetTags) return;
    const failed: string[] = [];
    for (const card of indexedCards.filter((candidate) => selected.has(candidate.filename))) {
      if (card.tags.some((existing) => existing.toLowerCase() === tag.toLowerCase())) continue;
      if (card.tags.length >= MAX_TAGS_PER_DOCUMENT) {
        failed.push(card.filename);
        continue;
      }
      try {
        await onSetTags(card.filename, [...card.tags, tag]);
      } catch {
        failed.push(card.filename);
      }
    }
    setDeleteError(failed.length > 0 ? `Couldn't tag: ${failed.join(', ')}` : null);
  };

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
        Drop files &middot; {extensionsLabel(acceptedExtensions(supportedExtensions))}
        <input
          ref={inputRef}
          type="file"
          className="corpus-panel__input"
          accept={acceptedExtensions(supportedExtensions).join(',')}
          multiple
          onChange={(event) => {
            handleFilesChange(event.target.files);
            // Cleared, so picking the same file again (after removing it from the
            // staged list, or after a size rejection) fires a change at all.
            event.target.value = '';
          }}
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

      {cardCount > 0 && (
        <input
          ref={searchRef}
          type="search"
          className="corpus-panel__search"
          placeholder="Filter documents…"
          aria-label="Filter documents"
          value={searchQuery}
          onChange={(event) => setSearchQuery(event.target.value)}
          data-testid="corpus-panel-search"
        />
      )}

      {indexedCards.length > 1 && (
        <div className="corpus-panel__controls">
          {canBulk && (
            <label className="corpus-panel__control">
              <input
                type="checkbox"
                checked={allVisibleSelected}
                onChange={() =>
                  setSelected((prev) => {
                    const next = new Set(prev);
                    for (const filename of visibleNames) {
                      if (allVisibleSelected) next.delete(filename);
                      else next.add(filename);
                    }
                    return next;
                  })
                }
                disabled={disabled || visibleNames.length === 0}
                data-testid="corpus-select-all"
              />
              All
            </label>
          )}
          <label className="corpus-panel__control">
            Sort
            <select
              value={sort}
              onChange={(event) => changeSort(event.target.value as SortKey)}
              data-testid="corpus-sort"
            >
              {(Object.keys(SORT_LABELS) as SortKey[]).map((key) => (
                <option key={key} value={key}>
                  {SORT_LABELS[key]}
                </option>
              ))}
            </select>
          </label>
          {availableTags.length > 0 && (
            <label className="corpus-panel__control">
              Tag
              <select
                value={activeTag ?? ''}
                onChange={(event) => setTagFilter(event.target.value || null)}
                data-testid="corpus-tag-filter"
              >
                <option value="">All</option>
                {availableTags.map((tag) => (
                  <option key={tag} value={tag}>
                    {tag}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
      )}

      {canBulk && selectedNames.length > 0 && (
        <BulkBar
          selected={selectedNames}
          disabled={disabled}
          reindexable={visibleIndexed
            .filter((card) => card.canReindex && selected.has(card.filename))
            .map((card) => card.filename)}
          onReindex={
            onReindexMany ??
            (onReindex ? (names: string[]) => names.forEach((name) => onReindex(name)) : undefined)
          }
          onAddTag={onSetTags ? bulkAddTag : undefined}
          onDelete={bulkDelete}
          onClear={() => setSelected(new Set())}
        />
      )}

      {(trimmedQuery !== '' || activeTag !== null) && filteredCards.length === 0 && (
        <p className="corpus-panel__empty-filter">
          {trimmedQuery !== ''
            ? `No documents match "${searchQuery.trim()}".`
            : `No documents tagged #${activeTag}.`}
        </p>
      )}

      <div className="corpus-panel__cards" ref={cardsRef}>
        {filteredCards.map((card) => (
          <DocumentCard
            key={card.key}
            card={card}
            disabled={disabled}
            readOnly={readOnly}
            confirming={confirming === card.key}
            deleting={deleting === card.filename}
            selectable={canBulk && card.status === 'indexed'}
            selected={selected.has(card.filename)}
            onToggleSelect={toggleSelect}
            onAskDelete={setConfirming}
            onCancelDelete={cancelDelete}
            onConfirmDelete={handleDelete}
            onRead={onRead}
            onOpenOriginal={onOpenOriginal}
            onReindex={onReindex}
            onCancelUpload={onCancelUpload}
            onRetryUpload={onRetryUpload}
            onDismissUpload={onDismissUpload}
            onSetTags={onSetTags}
          />
        ))}
      </div>
    </>
  );
}

// Memoized: App re-renders on every streamed token, and nothing here changes then.
export default memo(CorpusPanel);
