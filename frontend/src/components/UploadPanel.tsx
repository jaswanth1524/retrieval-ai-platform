import { useRef, useState } from 'react';
import type { DocumentIngestResponse } from '../api/types';
import './UploadPanel.css';

export type UploadStatus = 'idle' | 'uploading' | 'success' | 'error';

export interface UploadState {
  status: UploadStatus;
  result?: DocumentIngestResponse;
  error?: string;
}

interface UploadPanelProps {
  onUpload: (file: File) => Promise<void>;
  state: UploadState;
}

function UploadPanel({ onUpload, state }: UploadPanelProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const uploading = state.status === 'uploading';

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
        onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
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
      {state.status === 'success' && state.result && (
        <p className="upload-panel__success">
          {state.result.filename} &middot; {state.result.chunks_ingested} chunks &middot;{' '}
          <span className="mono">{state.result.collection_name}</span>
        </p>
      )}
      {state.status === 'error' && state.error && (
        <p className="upload-panel__error">{state.error}</p>
      )}
    </div>
  );
}

export default UploadPanel;
