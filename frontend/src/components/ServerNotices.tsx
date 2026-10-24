import { memo } from 'react';
import type { IndexProblem } from '../hooks/useServerHealth';
import './ServerNotices.css';

interface ServerNoticesProps {
  indexProblem: IndexProblem | null;
  /** Models that failed to load ("embedding", "reranker", "tokenizer"). */
  failedModels?: string[];
  updatedTo: string | null;
  /** Workspaces are on: the index check covers the default workspace only. */
  workspacesEnabled?: boolean;
  onReload: () => void;
}

/** What the server reported about itself that the user has to act on. */
const MODEL_NAMES: Record<string, string> = {
  embedding: 'embedding models',
  reranker: 'reranker',
  tokenizer: 'tokenizer',
};

function ServerNotices({
  indexProblem,
  failedModels = [],
  updatedTo,
  workspacesEnabled,
  onReload,
}: ServerNoticesProps) {
  if (!indexProblem && !updatedTo && failedModels.length === 0) return null;
  const models = failedModels.map((name) => MODEL_NAMES[name] ?? name).join(' and ');
  return (
    <div className="server-notices">
      {failedModels.length > 0 && (
        <div className="server-notice server-notice--bad" role="alert" data-testid="models-failed">
          <p className="server-notice__title">
            The server couldn’t load its {models}: questions and searches will fail.
          </p>
          <p className="server-notice__body">
            It keeps retrying in the background. If this stays, check the model settings and the
            server log.
          </p>
        </div>
      )}
      {indexProblem && (
        <div className="server-notice server-notice--bad" role="alert" data-testid="index-mismatch">
          <p className="server-notice__title">
            The index doesn’t match this server’s embedding setup
            {workspacesEnabled ? ' (default workspace)' : ''}: questions are refused.
          </p>
          <p className="server-notice__body">{indexProblem.detail}</p>
          {/* The server's detail already names the way out; with originals kept, this
              says where its steps are in the app. */}
          {indexProblem.originalsKept && (
            <p className="server-notice__body">
              In this app: ⌘K → “Download a backup of all documents”, delete every document, then
              ⌘K → “Restore documents from a backup”.
            </p>
          )}
        </div>
      )}
      {updatedTo && (
        <div className="server-notice" role="status" data-testid="server-updated">
          <p className="server-notice__title">DocRAG was updated to {updatedTo}.</p>
          <button type="button" className="server-notice__action" onClick={onReload}>
            Reload
          </button>
        </div>
      )}
    </div>
  );
}

export default memo(ServerNotices);
