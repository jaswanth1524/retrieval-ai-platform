import { useEffect, useRef, useState } from 'react';
import { isImeComposing } from '../../utils/keyboard';

interface TagEditorProps {
  filename: string;
  tags: string[];
  disabled?: boolean;
  onSave: (filename: string, tags: string[]) => Promise<void>;
}

/** A document's tags as chips, edited inline as a comma-separated list. */
export default function TagEditor({ filename, tags, disabled, onSave }: TagEditorProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Closing the editor unmounts the control that had focus (the field, Save, Cancel);
  // focus goes back to the button that opened it rather than to <body>.
  const editButtonRef = useRef<HTMLButtonElement>(null);
  const refocusRef = useRef(false);
  const close = () => {
    refocusRef.current = true;
    setEditing(false);
  };
  useEffect(() => {
    if (editing || !refocusRef.current) return;
    refocusRef.current = false;
    editButtonRef.current?.focus();
  }, [editing]);

  const save = async () => {
    const next = draft
      .split(',')
      .map((tag) => tag.trim())
      .filter(Boolean);
    setSaving(true);
    setError(null);
    try {
      await onSave(filename, next);
      close();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not save tags.');
    } finally {
      setSaving(false);
    }
  };

  if (editing) {
    return (
      <div className="corpus-panel__tags-edit">
        <input
          type="text"
          className="corpus-panel__tags-input"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (isImeComposing(event)) return;
            if (event.key === 'Enter') {
              event.preventDefault();
              void save();
            } else if (event.key === 'Escape') {
              // Cancels the edit only — not the drawer this panel may sit in.
              event.stopPropagation();
              close();
            }
          }}
          placeholder="legal, 2026"
          aria-label={`Tags for ${filename}, comma separated`}
          disabled={saving}
          autoFocus
          data-testid="corpus-tags-input"
        />
        {/* Enter and Esc did this alone: a mouse or touch user had no way to save. */}
        <div className="corpus-panel__tags-actions">
          <button
            type="button"
            className="corpus-panel__tags-button"
            onClick={() => void save()}
            disabled={saving}
            data-testid="corpus-tags-save"
          >
            Save
          </button>
          <button
            type="button"
            className="corpus-panel__tags-button"
            onClick={close}
            disabled={saving}
            data-testid="corpus-tags-cancel"
          >
            Cancel
          </button>
        </div>
        {error && (
          <span className="corpus-panel__tags-error" role="alert">
            {error}
          </span>
        )}
      </div>
    );
  }
  return (
    <div className="corpus-panel__tags" data-testid="corpus-tags">
      {tags.map((tag) => (
        <span key={tag} className="corpus-panel__tag">
          #{tag}
        </span>
      ))}
      <button
        ref={editButtonRef}
        type="button"
        className="corpus-panel__tags-button"
        onClick={() => {
          setDraft(tags.join(', '));
          setError(null);
          setEditing(true);
        }}
        disabled={disabled}
        aria-label={`Edit tags for ${filename}`}
        data-testid="corpus-tags-edit"
      >
        {tags.length > 0 ? '✎' : '+ tag'}
      </button>
    </div>
  );
}
