import { useRef, useState } from 'react';
import type { DragEvent } from 'react';
import type { DocumentIngestResponse } from '../api/types';
import './UploadPanel.css';

export type UploadStatus = 'idle' | 'uploading' | 'success' | 'error';

export interface UploadState {
  status: UploadStatus;
  result?: DocumentIngestResponse;
  error?: string;
}

// Matches the backend's AppSettings.max_upload_bytes default — used only until
// /config has loaded and supplies the server's actual configured limit.
const DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

interface UploadPanelProps {
  onUpload: (file: File) => Promise<void>;
  state: UploadState;
  maxUploadBytes?: number;
  // Lets the owner (App) clear a stale success/error result from a previous upload
  // the moment a new file is picked, instead of it lingering until the next upload
  // completes.
  onFileSelected?: () => void;
}

function formatBytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function UploadPanel({
  onUpload,
  state,
  maxUploadBytes = DEFAULT_MAX_UPLOAD_BYTES,
  onFileSelected,
}: UploadPanelProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [sizeError, setSizeError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const uploading = state.status === 'uploading';

  const handleFileChange = (file: File | null) => {
    if (file) onFileSelected?.();

    if (file && file.size > maxUploadBytes) {
      // Reject client-side before any network call — an oversized file would just
      // be rejected with a 413 after a wasted round-trip, or worse, no feedback at
      // all if the request hangs.
      setSizeError(
        `File is too large (${formatBytes(file.size)}). Maximum allowed is ${formatBytes(maxUploadBytes)}.`,
      );
      setSelectedFile(null);
      if (inputRef.current) inputRef.current.value = '';
      return;
    }
    setSizeError(null);
    setSelectedFile(file);
  };

  const handleUpload = async () => {
    if (!selectedFile || uploading) return;
    await onUpload(selectedFile);
    setSelectedFile(null);
    if (inputRef.current) inputRef.current.value = '';
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
    handleFileChange(event.dataTransfer.files?.[0] ?? null);
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
          Drop a file or <span className="upload-panel__dropzone-browse">browse</span>
        </span>
        <span className="upload-panel__dropzone-hint">PDF, TXT, MD &middot; up to {formatBytes(maxUploadBytes)}</span>
        <input
          ref={inputRef}
          type="file"
          className="upload-panel__input"
          accept=".pdf,.txt,.md,.markdown"
          onChange={(event) => handleFileChange(event.target.files?.[0] ?? null)}
          disabled={uploading}
          data-testid="upload-input"
        />
      </div>
      <button
        type="button"
        className="upload-panel__button"
        onClick={handleUpload}
        disabled={!selectedFile || uploading}
        data-testid="upload-button"
      >
        {uploading ? 'Uploading...' : 'Upload'}
      </button>
      {sizeError && (
        <p className="upload-panel__error" role="alert" data-testid="upload-size-error">
          {sizeError}
        </p>
      )}
      {state.status === 'success' && state.result && (
        <p className="upload-panel__success">
          {state.result.filename} &middot; {state.result.chunks_ingested} chunks &middot;{' '}
          <span className="mono">{state.result.collection_name}</span>
        </p>
      )}
      {state.status === 'error' && state.error && (
        <p className="upload-panel__error" role="alert">
          {state.error}
        </p>
      )}
    </div>
  );
}

export default UploadPanel;
