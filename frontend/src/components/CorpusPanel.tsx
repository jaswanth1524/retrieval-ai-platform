import { useRef, useState } from 'react';
import type { DragEvent, RefObject } from 'react';
import type { DocumentIngestResponse, IngestJobState } from '../api/types';
import './CorpusPanel.css';

// Both are structural details of UploadItem below and have no consumers outside this
// file — kept unexported so the module's public surface is just UploadItem + default.
type UploadItemStatus = 'uploading' | 'success' | 'error';

interface UploadProgress {
  state: IngestJobState;
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
}

interface RejectedFile {
  filename: string;
  message: string;
}

// Matches the backend's AppSettings.max_upload_bytes default — used only until
// /config has loaded and supplies the server's actual configured limit.
const DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

interface CorpusPanelProps {
  filenames: string[];
  chunkCounts: Record<string, number>;
  uploads: UploadItem[];
  onUpload: (files: File[]) => Promise<void>;
  onDelete: (filename: string) => Promise<void>;
  maxUploadBytes?: number;
  disabled?: boolean;
  // Lets the context panel's header "Upload" action open this same file dialog.
  browseInputRef?: RefObject<HTMLInputElement | null>;
}

function formatBytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
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
  filename: string;
  status: 'indexed' | 'indexing' | 'failed';
  detail: string;
  pct: number;
  error?: string;
}

function CorpusPanel({
  filenames,
  chunkCounts,
  uploads,
  onUpload,
  onDelete,
  maxUploadBytes = DEFAULT_MAX_UPLOAD_BYTES,
  disabled,
  browseInputRef,
}: CorpusPanelProps) {
  const [stagedFiles, setStagedFiles] = useState<File[]>([]);
  const [rejectedFiles, setRejectedFiles] = useState<RejectedFile[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const internalInputRef = useRef<HTMLInputElement>(null);
  const inputRef = browseInputRef ?? internalInputRef;
  const submitting = uploads.some((item) => item.status === 'uploading');

  const handleFilesChange = (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    const accepted: File[] = [];
    const rejected: RejectedFile[] = [];
    Array.from(fileList).forEach((file) => {
      if (file.size > maxUploadBytes) {
        rejected.push({
          filename: file.name,
          message: `${file.name} is too large (${formatBytes(file.size)}). Maximum is ${formatBytes(maxUploadBytes)}.`,
        });
      } else {
        accepted.push(file);
      }
    });
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

  const indexedCards: DocCard[] = filenames.map((filename) => ({
    filename,
    status: 'indexed',
    detail: `${chunkCounts[filename] ?? 0} chunks`,
    pct: 100,
  }));

  // Only in-flight/failed uploads not yet reflected in `filenames` — a successful
  // upload's filename lands in `filenames` on the same render its status flips to
  // 'success', so without this filter it would render twice.
  const inFlightCards: DocCard[] = uploads
    .filter((item) => item.status !== 'success' && !filenames.includes(item.filename))
    .map((item) =>
      item.status === 'error'
        ? { filename: item.filename, status: 'failed', detail: item.error ?? 'Ingestion failed.', pct: 0, error: item.error }
        : {
            filename: item.filename,
            status: 'indexing',
            detail: `indexing ${progressPercent(item.progress)}%`,
            pct: progressPercent(item.progress),
          },
    );

  const cards = [...indexedCards, ...inFlightCards];

  return (
    <>
      <label
        className={`corpus-panel__dropzone${dragOver ? ' corpus-panel__dropzone--active' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        data-testid="upload-dropzone"
      >
        Drop files &middot; pdf docx md txt
        <input
          ref={inputRef}
          type="file"
          className="corpus-panel__input"
          accept=".pdf,.txt,.md,.markdown,.docx,.html,.htm,.csv"
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

      {cards.map((card) => (
        <article key={card.filename} className="corpus-panel__card" data-testid="corpus-panel-item">
          <div className="corpus-panel__card-top">
            <span className="corpus-panel__ext">{extensionOf(card.filename)}</span>
            <span className="corpus-panel__name">{card.filename}</span>
            {confirming === card.filename ? (
              <div className="corpus-panel__confirm">
                <span>Delete?</span>
                <button
                  type="button"
                  className="corpus-panel__confirm-yes"
                  onClick={() => void handleDelete(card.filename)}
                  disabled={deleting === card.filename}
                >
                  {deleting === card.filename ? '…' : 'Confirm'}
                </button>
                <button type="button" className="corpus-panel__confirm-no" onClick={() => setConfirming(null)}>
                  Cancel
                </button>
              </div>
            ) : (
              card.status === 'indexed' && (
                <button
                  type="button"
                  className="corpus-panel__remove"
                  onClick={() => setConfirming(card.filename)}
                  disabled={disabled}
                  aria-label={`Delete ${card.filename}`}
                >
                  ✕
                </button>
              )
            )}
          </div>
          <div className="corpus-panel__status">
            <span className={`corpus-panel__dot corpus-panel__dot--${card.status}`} aria-hidden="true" />
            <span>{card.status === 'failed' ? `failed — ${card.detail}` : card.detail}</span>
          </div>
          {card.status === 'indexing' && (
            <div className="corpus-panel__bar">
              <div className="corpus-panel__bar-fill" style={{ width: `${card.pct}%` }} />
            </div>
          )}
        </article>
      ))}
    </>
  );
}

export default CorpusPanel;
