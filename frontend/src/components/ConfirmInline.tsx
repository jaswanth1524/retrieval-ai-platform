import type { KeyboardEvent } from 'react';
import './ConfirmInline.css';

interface ConfirmInlineProps {
  /** Shown before the buttons ("Delete?"); left out where the button it replaced said enough. */
  prompt?: string;
  confirmLabel: string;
  onConfirm: () => void;
  onCancel: () => void;
  /** The confirmed action is running: Confirm shows "…" and can't be pressed twice. */
  busy?: boolean;
  /** Confirm is unavailable for now (a question is in flight); Cancel always works. */
  disabled?: boolean;
  /** Colours for the side panel ('rail') or the main column ('main'). */
  tone?: 'rail' | 'main';
  /** Layout from the caller (the box it fills); colours and buttons come from here. */
  className?: string;
  confirmTestId?: string;
}

/** An "are you sure?" that takes the place of a destructive button.
 *
 *  Focus lands on Cancel, the safe choice: the button that opened this has just
 *  unmounted, which dropped focus to <body>. Escape backs out of the confirm only — not
 *  the drawer or dialog it may sit in. */
function ConfirmInline({
  prompt,
  confirmLabel,
  onConfirm,
  onCancel,
  busy = false,
  disabled = false,
  tone = 'rail',
  className,
  confirmTestId,
}: ConfirmInlineProps) {
  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'Escape') return;
    event.stopPropagation();
    onCancel();
  };

  return (
    <div
      className={`confirm-inline confirm-inline--${tone}${className ? ` ${className}` : ''}`}
      onKeyDown={handleKeyDown}
    >
      {prompt && <span className="confirm-inline__prompt">{prompt}</span>}
      <span className="confirm-inline__actions">
        <button type="button" className="confirm-inline__no" autoFocus onClick={onCancel}>
          Cancel
        </button>
        <button
          type="button"
          className="confirm-inline__yes"
          onClick={onConfirm}
          disabled={busy || disabled}
          data-testid={confirmTestId}
        >
          {busy ? '…' : confirmLabel}
        </button>
      </span>
    </div>
  );
}

export default ConfirmInline;
