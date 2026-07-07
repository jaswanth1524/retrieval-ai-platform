import { useRef, useState } from 'react';
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
}

function formatBytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function UploadPanel({ onUpload, state, maxUploadBytes = DEFAULT_MAX_UPLOAD_BYTES }: UploadPanelProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [sizeError, setSizeError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const uploading = state.status === 'uploading';

  const handleFileChange = (file: File | null) => {
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

  return (
    <div className="upload-panel">
      <h2 className="upload-panel__title">Documents</h2>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.txt,.md,.markdown"
        onChange={(event) => handleFileChange(event.target.files?.[0] ?? null)}
        disabled={uploading}
        data-testid="upload-input"
      />
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
