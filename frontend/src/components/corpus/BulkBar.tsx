import { useState } from 'react';
import ConfirmInline from '../ConfirmInline';

interface BulkBarProps {
  selected: string[];
  disabled?: boolean;
  /** Names among the selection with a stored original (re-indexable). */
  reindexable: string[];
  onReindex?: (filenames: string[]) => void;
  onAddTag?: (tag: string) => Promise<void>;
  /** Compare the selection, offered when exactly two documents are selected. */
  onCompare?: (filenames: string[]) => void;
  onDelete: () => Promise<void>;
  onClear: () => void;
}

/** What to do with every selected document at once. */
export default function BulkBar({
  selected,
  disabled,
  reindexable,
  onReindex,
  onAddTag,
  onCompare,
  onDelete,
  onClear,
}: BulkBarProps) {
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [busy, setBusy] = useState(false);
  const [tagDraft, setTagDraft] = useState('');
  const count = selected.length;
  const label = `${count} document${count === 1 ? '' : 's'}`;

  const run = async (work: () => Promise<void>) => {
    setBusy(true);
    try {
      await work();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="corpus-panel__bulk" role="group" aria-label="Selected documents" data-testid="corpus-bulk">
      <span className="corpus-panel__bulk-count mono">{count} selected</span>
      {confirmingDelete ? (
        <ConfirmInline
          className="corpus-panel__confirm"
          prompt={`Delete ${label}?`}
          confirmLabel="Delete"
          onCancel={() => setConfirmingDelete(false)}
          onConfirm={() =>
            void run(async () => {
              await onDelete();
              setConfirmingDelete(false);
            })
          }
          busy={busy}
          confirmTestId="corpus-bulk-delete-confirm"
        />
      ) : (
        <>
          {onAddTag && (
            <form
              className="corpus-panel__bulk-tag"
              onSubmit={(event) => {
                event.preventDefault();
                const tag = tagDraft.trim();
                if (!tag) return;
                void run(async () => {
                  await onAddTag(tag);
                  setTagDraft('');
                });
              }}
            >
              <input
                type="text"
                className="corpus-panel__bulk-input"
                value={tagDraft}
                onChange={(event) => setTagDraft(event.target.value)}
                placeholder="tag"
                aria-label={`Tag to add to ${label}`}
                disabled={disabled || busy}
                data-testid="corpus-bulk-tag-input"
              />
              <button
                type="submit"
                className="corpus-panel__bulk-button"
                disabled={disabled || busy || !tagDraft.trim()}
                data-testid="corpus-bulk-tag"
              >
                + tag
              </button>
            </form>
          )}
          {onCompare && count === 2 && (
            <button
              type="button"
              className="corpus-panel__bulk-button"
              onClick={() => onCompare(selected)}
              disabled={disabled || busy}
              title="Compare the two selected documents"
              data-testid="corpus-bulk-compare"
            >
              Compare
            </button>
          )}
          {onReindex && reindexable.length > 0 && (
            <button
              type="button"
              className="corpus-panel__bulk-button"
              onClick={() => onReindex(reindexable)}
              disabled={disabled || busy}
              data-testid="corpus-bulk-reindex"
            >
              ↻ Re-index {reindexable.length}
            </button>
          )}
          <button
            type="button"
            className="corpus-panel__bulk-button corpus-panel__bulk-delete"
            onClick={() => setConfirmingDelete(true)}
            disabled={disabled || busy}
            data-testid="corpus-bulk-delete"
          >
            Delete {count}
          </button>
          <button
            type="button"
            className="corpus-panel__bulk-button"
            onClick={onClear}
            disabled={busy}
            data-testid="corpus-bulk-clear"
          >
            Clear
          </button>
        </>
      )}
    </div>
  );
}
