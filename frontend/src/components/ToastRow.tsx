import { memo } from 'react';
import type { Toast } from '../hooks/useToasts';
import './ToastRow.css';

interface ToastRowProps {
  toasts: Toast[];
  onDismiss: (id: string) => void;
  /** Pause and resume a toast's auto-dismiss while the pointer or focus is on it. */
  onHold?: (id: string) => void;
  onRelease?: (id: string) => void;
}

/** Notices rendered in normal flow, between the thread and the composer.
 *
 * Deliberately not an overlay: these replace banners that reported real problems
 * (chat history failed to save), and a floating toast over the composer would cover
 * the control the user is reaching for. In flow it displaces instead.
 */
function ToastRow({ toasts, onDismiss, onHold, onRelease }: ToastRowProps) {
  const renderToast = (toast: Toast) => (
    <div
      key={toast.id}
      className={`toast toast--${toast.tone}`}
      onMouseEnter={() => onHold?.(toast.id)}
      onMouseLeave={() => onRelease?.(toast.id)}
      onFocus={() => onHold?.(toast.id)}
      onBlur={() => onRelease?.(toast.id)}
      data-testid="toast"
    >
      <span className="toast__dot" aria-hidden="true" />
      <span className="toast__text">
        <span className="toast__title">{toast.title}</span>
        {toast.body && <span className="toast__body">{toast.body}</span>}
      </span>
      <button
        type="button"
        className="toast__dismiss"
        onClick={() => onDismiss(toast.id)}
        aria-label={`Dismiss: ${toast.title}`}
        data-testid="toast-dismiss"
      >
        <span aria-hidden="true">✕</span>
      </button>
    </div>
  );
  // Two regions, both always mounted, even empty: screen readers often skip a live
  // region that appears together with its first message. Failures interrupt
  // (assertive); confirmations wait their turn. One role="alert" nested inside a polite
  // region was read twice by some screen readers. Not role="alert" itself: an empty
  // alert would sit in the accessibility tree all the time. Failures are listed first.
  return (
    <div
      className={`toast-row${toasts.length === 0 ? ' toast-row--empty' : ''}`}
      data-testid={toasts.length === 0 ? undefined : 'toast-row'}
    >
      <div className="toast-row__region" aria-live="assertive" data-testid="toast-failures">
        {toasts.filter((toast) => toast.tone === 'bad').map(renderToast)}
      </div>
      <div className="toast-row__region" aria-live="polite">
        {toasts.filter((toast) => toast.tone !== 'bad').map(renderToast)}
      </div>
    </div>
  );
}

// Memoized: App re-renders on every streamed frame, and none of this depends on it.
export default memo(ToastRow);
