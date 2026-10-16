import { useState } from 'react';
import ConfirmInline from '../ConfirmInline';
import { WORKSPACE_NAME_PATTERN } from '../../utils/limits';

interface WorkspaceBarProps {
  /** The workspace in use; '' is the default one. */
  current: string;
  /** Workspaces besides the default. */
  workspaces: string[];
  disabled?: boolean;
  /** The read-only key: switching only, no create or delete. */
  readOnly?: boolean;
  onSwitch: (name: string) => void;
  /** Rejects with a message to show (a bad name, one that exists). */
  onCreate: (name: string) => Promise<void>;
  /** Rejects with a message to show (409 while it still holds documents). */
  onDelete: (name: string) => Promise<void>;
}

/** Which corpus the panel, questions and searches use (WORKSPACES_ENABLED). */
export default function WorkspaceBar({
  current,
  workspaces,
  disabled,
  readOnly = false,
  onSwitch,
  onCreate,
  onDelete,
}: WorkspaceBarProps) {
  const [creating, setCreating] = useState(false);
  const [draft, setDraft] = useState('');
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async (work: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await work();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'That did not work.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="corpus-panel__workspace" data-testid="workspace-bar">
      <label className="corpus-panel__workspace-label mono" htmlFor="workspace-select">
        Workspace
      </label>
      <select
        id="workspace-select"
        className="corpus-panel__workspace-select"
        value={current}
        onChange={(event) => {
          setConfirmingDelete(false);
          onSwitch(event.target.value);
        }}
        disabled={disabled || busy}
        data-testid="workspace-select"
      >
        <option value="">Default</option>
        {workspaces.map((name) => (
          <option key={name} value={name}>
            {name}
          </option>
        ))}
      </select>
      {!readOnly && !creating && !confirmingDelete && (
        <>
          <button
            type="button"
            className="corpus-panel__bulk-button"
            onClick={() => setCreating(true)}
            disabled={disabled || busy}
            data-testid="workspace-new"
          >
            New
          </button>
          {current && (
            <button
              type="button"
              className="corpus-panel__bulk-button"
              onClick={() => setConfirmingDelete(true)}
              disabled={disabled || busy}
              data-testid="workspace-delete"
            >
              Delete
            </button>
          )}
        </>
      )}
      {confirmingDelete && (
        <ConfirmInline
          className="corpus-panel__confirm"
          prompt={`Delete empty workspace ${current}?`}
          confirmLabel="Delete"
          onCancel={() => setConfirmingDelete(false)}
          onConfirm={() =>
            void run(async () => {
              await onDelete(current);
              setConfirmingDelete(false);
            })
          }
          busy={busy}
        />
      )}
      {creating && (
        <form
          className="corpus-panel__workspace-form"
          onSubmit={(event) => {
            event.preventDefault();
            const name = draft.trim();
            if (!WORKSPACE_NAME_PATTERN.test(name)) {
              setError('Use 1–32 lowercase letters, digits or hyphens.');
              return;
            }
            void run(async () => {
              await onCreate(name);
              setDraft('');
              setCreating(false);
            });
          }}
        >
          <input
            type="text"
            className="corpus-panel__url-input"
            value={draft}
            onChange={(event) => setDraft(event.target.value.toLowerCase())}
            placeholder="new-workspace"
            aria-label="New workspace name"
            disabled={busy}
            data-testid="workspace-name-input"
            autoFocus
          />
          <button type="submit" className="corpus-panel__bulk-button" disabled={busy || !draft.trim()}>
            Create
          </button>
          <button
            type="button"
            className="corpus-panel__bulk-button"
            onClick={() => {
              setCreating(false);
              setError(null);
            }}
            disabled={busy}
          >
            Cancel
          </button>
        </form>
      )}
      {error && (
        <p className="corpus-panel__reject" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
