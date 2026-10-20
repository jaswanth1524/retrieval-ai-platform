import { useId, useState } from 'react';
import { useDialog } from '../hooks/useDialog';
import { isImeComposing } from '../utils/keyboard';
import { MAX_QUESTION_CHARS } from '../utils/limits';
import './SummaryDialog.css';

interface SummaryDialogProps {
  /** One document to summarize, or two to compare. */
  filenames: string[];
  onRun: (instruction: string | undefined) => void;
  onClose: () => void;
}

/** Asks what a summary or comparison should concentrate on (optional) before it runs:
 *  "deadlines", "what changed for contractors". Enter with it empty is a general one. */
export default function SummaryDialog({ filenames, onRun, onClose }: SummaryDialogProps) {
  const dialogRef = useDialog(true, onClose);
  const [focus, setFocus] = useState('');
  const inputId = useId();
  const compare = filenames.length === 2;
  const brief = filenames.length > 2;
  const title = compare
    ? `Compare ${filenames[0]} and ${filenames[1]}`
    : brief
      ? `Brief on ${filenames.length} documents`
      : `Summarize ${filenames[0]}`;
  const run = () => {
    const instruction = focus.trim();
    onRun(instruction || undefined);
  };
  return (
    <div
      className="summary-dialog__backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        className="summary-dialog"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        ref={dialogRef}
        data-testid="summary-dialog"
      >
        <form
          onSubmit={(event) => {
            event.preventDefault();
            run();
          }}
        >
          <h2 className="summary-dialog__title">{title}</h2>
          <label className="summary-dialog__label" htmlFor={inputId}>
            Focus on (optional)
          </label>
          <input
            id={inputId}
            className="summary-dialog__input"
            value={focus}
            onChange={(event) => setFocus(event.target.value)}
            onKeyDown={(event) => {
              // An input method's Enter confirms its candidate, not the form.
              if (event.key === 'Enter' && isImeComposing(event)) event.preventDefault();
            }}
            placeholder={compare ? 'e.g. what changed for contractors' : 'e.g. deadlines and penalties'}
            maxLength={MAX_QUESTION_CHARS}
            autoFocus
            data-testid="summary-focus-input"
          />
          <div className="summary-dialog__actions">
            <button type="button" className="summary-dialog__cancel" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="summary-dialog__primary" data-testid="summary-run">
              {compare ? 'Compare' : brief ? 'Write the brief' : 'Summarize'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
