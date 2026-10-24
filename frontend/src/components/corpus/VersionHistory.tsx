import { useEffect, useState } from 'react';
import { api, apiErrorMessage } from '../../api/client';
import type { DocumentVersionResponse } from '../../api/types';
import { formatBytes } from '../../utils/uploadValidation';
import { formatRelativeTime } from '../../utils/relativeTime';
import ConfirmInline from '../ConfirmInline';

interface VersionHistoryProps {
  filename: string;
  /** The workspace the card belongs to. */
  workspace?: string;
  disabled?: boolean;
  onRestore: (filename: string, version: string) => void;
  onClose: () => void;
}

/** A document's earlier originals (RAW_DOCUMENT_VERSIONS), newest first, each one
 *  restorable: it is indexed again as an upload of those bytes would be. */
export default function VersionHistory({
  filename,
  workspace,
  disabled,
  onRestore,
  onClose,
}: VersionHistoryProps) {
  const [versions, setVersions] = useState<DocumentVersionResponse[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api
      .listDocumentVersions(filename, controller.signal, workspace)
      .then((response) => setVersions(response.versions))
      .catch((err) => {
        if (!controller.signal.aborted) setError(apiErrorMessage(err, 'Could not list the versions.'));
      });
    return () => controller.abort();
  }, [filename, workspace]);

  return (
    <div className="corpus-panel__versions" data-testid="corpus-versions">
      <div className="corpus-panel__versions-head">
        <span>Earlier versions</span>
        <button type="button" onClick={onClose} aria-label={`Close the versions of ${filename}`}>
          ✕
        </button>
      </div>
      {error && <p role="alert">{error}</p>}
      {!error && versions === null && <p>Loading…</p>}
      {versions !== null && versions.length === 0 && <p>No earlier versions are kept.</p>}
      {versions !== null && versions.length > 0 && (
        <ul>
          {versions.map((item) => (
            <li key={item.version}>
              <span className="mono">
                replaced {formatRelativeTime(item.saved_at * 1000)} · {formatBytes(item.size)}
              </span>
              {confirming === item.version ? (
                <ConfirmInline
                  prompt="Restore?"
                  confirmLabel="Restore"
                  onCancel={() => setConfirming(null)}
                  onConfirm={() => {
                    setConfirming(null);
                    onRestore(filename, item.version);
                    onClose();
                  }}
                />
              ) : (
                <button
                  type="button"
                  onClick={() => setConfirming(item.version)}
                  disabled={disabled}
                  data-testid="corpus-version-restore"
                >
                  Restore
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
