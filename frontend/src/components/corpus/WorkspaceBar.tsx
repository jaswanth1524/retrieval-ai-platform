import { useEffect, useId, useRef, useState } from 'react';
import ConfirmInline from '../ConfirmInline';
import { isImeComposing } from '../../utils/keyboard';
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

/** How long the select waits, after the last key that moved it, before switching. */
export const WORKSPACE_BROWSE_COMMIT_MS = 600;

// Keys that move a closed <select>'s value one option at a time: Windows and Linux
// browsers fire `change` on each, so arrowing past two workspaces switched twice
// (re-listing the corpus and possibly opening a new chat each time). Letters too: they
// jump to the matching option.
const BROWSE_KEYS = new Set([
  'ArrowUp',
  'ArrowDown',
  'ArrowLeft',
  'ArrowRight',
  'Home',
  'End',
  'PageUp',
  'PageDown',
]);

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
  // A workspace picked with the keyboard but not switched to yet (see BROWSE_KEYS):
  // shown in the select (it is controlled — without this React snapped it back to
  // `current` and the arrows could never get past the next option), switched to on
  // Enter, on blur, or once the keys stop.
  const [pending, setPending] = useState<string | null>(null);
  const browsingRef = useRef(false);
  const browseTimerRef = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(browseTimerRef.current), []);
  const errorId = useId();
  const selectRef = useRef<HTMLSelectElement>(null);
  const newRef = useRef<HTMLButtonElement>(null);
  const deleteRef = useRef<HTMLButtonElement>(null);
  // The control that had focus (the name field, Create, Cancel, Confirm) unmounts when
  // the form or the confirm closes; focus goes here next instead of to <body>.
  const [refocus, setRefocus] = useState<'select' | 'new' | 'delete' | null>(null);
  useEffect(() => {
    if (refocus === null || busy) return;
    setRefocus(null);
    const target = { select: selectRef, new: newRef, delete: deleteRef }[refocus].current;
    // The Delete button is gone once the default workspace is in use.
    (target ?? newRef.current ?? selectRef.current)?.focus();
  }, [refocus, busy]);

  // The workspace in use is always an option: missing from the list (created while the
  // list couldn't be read, or deleted elsewhere), the select showed "Default" while
  // requests went to it.
  const options = current && !workspaces.includes(current) ? [...workspaces, current] : workspaces;

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

  const commitSwitch = (value: string) => {
    window.clearTimeout(browseTimerRef.current);
    browseTimerRef.current = undefined;
    setPending(null);
    if (value !== current) onSwitch(value);
  };

  const cancelCreate = () => {
    setCreating(false);
    setError(null);
    setRefocus('new');
  };

  return (
    <div className="corpus-panel__workspace" data-testid="workspace-bar">
      <label className="corpus-panel__workspace-label mono" htmlFor="workspace-select">
        Workspace
      </label>
      <select
        ref={selectRef}
        id="workspace-select"
        className="corpus-panel__workspace-select"
        value={pending ?? current}
        onPointerDown={() => {
          browsingRef.current = false;
        }}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && pending !== null) {
            commitSwitch(pending);
            return;
          }
          const typeahead =
            event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey;
          if (BROWSE_KEYS.has(event.key) || typeahead) browsingRef.current = true;
        }}
        onChange={(event) => {
          setConfirmingDelete(false);
          const value = event.target.value;
          if (!browsingRef.current) {
            commitSwitch(value);
            return;
          }
          setPending(value);
          window.clearTimeout(browseTimerRef.current);
          browseTimerRef.current = window.setTimeout(
            () => commitSwitch(value),
            WORKSPACE_BROWSE_COMMIT_MS,
          );
        }}
        onBlur={() => {
          browsingRef.current = false;
          if (pending !== null) commitSwitch(pending);
        }}
        disabled={disabled || busy}
        aria-describedby={error && !creating ? errorId : undefined}
        data-testid="workspace-select"
      >
        <option value="">Default</option>
        {options.map((name) => (
          <option key={name} value={name}>
            {name}
          </option>
        ))}
      </select>
      {!readOnly && !creating && !confirmingDelete && (
        <>
          <button
            ref={newRef}
            type="button"
            className="corpus-panel__bulk-button"
            onClick={() => setCreating(true)}
            disabled={disabled || busy}
            aria-label="New workspace"
            data-testid="workspace-new"
          >
            New
          </button>
          {current && (
            <button
              ref={deleteRef}
              type="button"
              className="corpus-panel__bulk-button"
              onClick={() => setConfirmingDelete(true)}
              disabled={disabled || busy}
              aria-label={`Delete workspace ${current}`}
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
          onCancel={() => {
            setConfirmingDelete(false);
            setRefocus('delete');
          }}
          onConfirm={() =>
            void run(async () => {
              await onDelete(current);
              setConfirmingDelete(false);
              setRefocus('select');
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
              setRefocus('select');
            });
          }}
        >
          <input
            type="text"
            className="corpus-panel__url-input"
            value={draft}
            onChange={(event) => setDraft(event.target.value.toLowerCase())}
            onKeyDown={(event) => {
              if (isImeComposing(event) || event.key !== 'Escape') return;
              // Cancels the new name only — not the drawer this panel may sit in.
              event.stopPropagation();
              cancelCreate();
            }}
            placeholder="new-workspace"
            aria-label="New workspace name"
            aria-invalid={error !== null}
            aria-describedby={error ? errorId : undefined}
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
            onClick={cancelCreate}
            disabled={busy}
          >
            Cancel
          </button>
        </form>
      )}
      {error && (
        <p id={errorId} className="corpus-panel__reject" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
