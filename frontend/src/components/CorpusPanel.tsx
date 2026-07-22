import { useState } from 'react';
import './CorpusPanel.css';

interface CorpusPanelProps {
  filenames: string[];
  onDelete: (filename: string) => Promise<void>;
  disabled?: boolean;
}

function CorpusPanel({ filenames, onDelete, disabled }: CorpusPanelProps) {
  const [confirming, setConfirming] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (filenames.length === 0) return null;

  const handleDelete = async (filename: string) => {
    setDeleting(filename);
    setError(null);
    try {
      await onDelete(filename);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete document.');
    } finally {
      setDeleting(null);
      setConfirming(null);
    }
  };

  return (
    <div className="corpus-panel">
      <div className="corpus-panel__label">Indexed documents</div>
      {error && (
        <div className="corpus-panel__error" role="alert">
          {error}
        </div>
      )}
      <ul className="corpus-panel__list">
        {filenames.map((filename) => (
          <li key={filename} className="corpus-panel__item" data-testid="corpus-panel-item">
            {confirming === filename ? (
              <div className="corpus-panel__confirm">
                <span className="corpus-panel__confirm-text">Delete?</span>
                <button
                  type="button"
                  className="corpus-panel__confirm-yes"
                  onClick={() => void handleDelete(filename)}
                  disabled={deleting === filename}
                >
                  {deleting === filename ? 'Deleting…' : 'Confirm'}
                </button>
                <button
                  type="button"
                  className="corpus-panel__confirm-no"
                  onClick={() => setConfirming(null)}
                  disabled={deleting === filename}
                >
                  Cancel
                </button>
              </div>
            ) : (
              <>
                <span className="corpus-panel__filename mono">{filename}</span>
                <button
                  type="button"
                  className="corpus-panel__delete"
                  onClick={() => setConfirming(filename)}
                  disabled={disabled}
                  aria-label={`Delete ${filename}`}
                >
                  ✕
                </button>
              </>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default CorpusPanel;
