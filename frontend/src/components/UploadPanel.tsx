import { useRef, useState } from 'react';
import type { DragEvent } from 'react';
import type { DocumentIngestResponse, IngestJobState } from '../api/types';
import './UploadPanel.css';

export type UploadItemStatus = 'uploading' | 'success' | 'error';

export interface UploadProgress {
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

const PROGRESS_LABELS: Record<IngestJobState, string> = {
  queued: 'Queued...',
  parsing: 'Parsing...',
  embedding: 'Embedding...',
  done: 'Done',
  failed: 'Failed',
};

function uploadingLabel(progress?: UploadProgress): string {
  if (!progress) return 'Uploading...';
  const label = PROGRESS_LABELS[progress.state];
  if (progress.chunksTotal > 0) {
    return `${label} (${progress.chunksDone}/${progress.chunksTotal})`;
  }
  return label;
}

// Matches the backend's AppSettings.max_upload_bytes default — used only until
// /config has loaded and supplies the server's actual configured limit.
const DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

interface UploadPanelProps {
  onUpload: (files: File[]) => Promise<void>;
  uploads: UploadItem[];
  maxUploadBytes?: number;
}

function formatBytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function UploadPanel({ onUpload, uploads, maxUploadBytes = DEFAULT_MAX_UPLOAD_BYTES }: UploadPanelProps) {
  const [stagedFiles, setStagedFiles] = useState<File[]>([]);
  const [rejectedFiles, setRejectedFiles] = useState<RejectedFile[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const uploading = uploads.some((item) => item.status === 'uploading');

  const handleFilesChange = (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;

    const accepted: File[] = [];
    const rejected: RejectedFile[] = [];
    Array.from(fileList).forEach((file) => {
      if (file.size > maxUploadBytes) {
        // Reject client-side before any network call — an oversized file would just
        // be rejected with a 413 after a wasted round-trip, or worse, no feedback at
        // all if the request hangs. Other valid files in the same batch still stage.
        rejected.push({
          filename: file.name,
          message: `File is too large (${formatBytes(file.size)}). Maximum allowed is ${formatBytes(maxUploadBytes)}.`,
        });
      } else {
        accepted.push(file);
      }
    });
    setRejectedFiles(rejected);
    // Append rather than replace — a browse followed by a drop accumulates into
    // one staged batch instead of the second selection clobbering the first.
    setStagedFiles((prev) => [...prev, ...accepted]);
  };

  const removeStagedFile = (index: number) => {
    setStagedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (stagedFiles.length === 0 || uploading) return;
    const files = stagedFiles;
    setStagedFiles([]);
    setRejectedFiles([]);
    if (inputRef.current) inputRef.current.value = '';
    await onUpload(files);
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (!uploading) setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragOver(false);
    if (uploading) return;
    handleFilesChange(event.dataTransfer.files);
  };

  return (
    <div className="upload-panel">
      <h2 className="upload-panel__title">Documents</h2>
      <div
        className={`upload-panel__dropzone${dragOver ? ' upload-panel__dropzone--active' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        data-testid="upload-dropzone"
      >
        <span className="upload-panel__dropzone-icon" aria-hidden="true" />
        <span className="upload-panel__dropzone-text">
          Drop files or <span className="upload-panel__dropzone-browse">browse</span>
        </span>
        <span className="upload-panel__dropzone-hint">PDF, DOCX, HTML, CSV, TXT, MD &middot; up to {formatBytes(maxUploadBytes)} each</span>
        <input
          ref={inputRef}
          type="file"
          className="upload-panel__input"
          accept=".pdf,.txt,.md,.markdown,.docx,.html,.htm,.csv"
          multiple
          onChange={(event) => handleFilesChange(event.target.files)}
          disabled={uploading}
          data-testid="upload-input"
        />
      </div>
      {stagedFiles.length > 0 && (
        <ul className="upload-panel__staged-list" data-testid="upload-staged-list">
          {stagedFiles.map((file, index) => (
            <li key={`${file.name}-${index}`} className="upload-panel__chip" data-testid="upload-staged-item">
              <span className="upload-panel__chip-name mono">{file.name}</span>
              <span className="upload-panel__chip-size">{formatBytes(file.size)}</span>
              <button
                type="button"
                className="upload-panel__chip-remove"
                onClick={() => removeStagedFile(index)}
                aria-label={`Remove ${file.name}`}
                data-testid="upload-staged-remove"
              >
                &times;
              </button>
            </li>
          ))}
        </ul>
      )}
      <button
        type="button"
        className="upload-panel__button"
        onClick={handleUpload}
        disabled={stagedFiles.length === 0 || uploading}
        data-testid="upload-button"
      >
        {uploading
          ? 'Uploading...'
          : stagedFiles.length > 1
            ? `Upload ${stagedFiles.length} files`
            : 'Upload'}
      </button>
      {rejectedFiles.map((rejected) => (
        <p key={rejected.filename} className="upload-panel__error" role="alert" data-testid="upload-size-error">
          {rejected.message}
        </p>
      ))}
      {uploads.length > 0 && (
        <ul className="upload-panel__job-list" data-testid="upload-job-list">
          {uploads.map((item) => (
            <li
              key={item.id}
              className={`upload-panel__job upload-panel__job--${item.status}`}
              data-testid="upload-job-item"
            >
              <span className="upload-panel__job-name mono">{item.filename}</span>
              {item.status === 'uploading' && <span>{uploadingLabel(item.progress)}</span>}
              {item.status === 'success' && item.result && (
                <span className="upload-panel__success">
                  {item.result.chunks_ingested} chunks &middot; <span className="mono">{item.result.collection_name}</span>
                </span>
              )}
              {item.status === 'error' && (
                <span className="upload-panel__error" role="alert">
                  {item.error}
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default UploadPanel;
