import type { Toast } from '../hooks/useToasts';
import './ToastRow.css';

interface ToastRowProps {
  toasts: Toast[];
  onDismiss: (id: string) => void;
}

/** Notices rendered in normal flow, between the thread and the composer.
 *
 * Deliberately not an overlay: these replace banners that reported real problems
 * (chat history failed to save), and a floating toast over the composer would cover
 * the control the user is reaching for. In flow it displaces instead.
 */
function ToastRow({ toasts, onDismiss }: ToastRowProps) {
  // Always mounted, even empty: screen readers often skip a live region that appears
  // together with its first message, so a toast rendered into a fresh region could go
  // unannounced. Confirmations are announced politely through this region; failures
  // interrupt (role="alert").
  return (
    <div
      className={`toast-row${toasts.length === 0 ? ' toast-row--empty' : ''}`}
      aria-live="polite"
      data-testid={toasts.length === 0 ? undefined : 'toast-row'}
    >
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`toast toast--${toast.tone}`}
          role={toast.tone === 'bad' ? 'alert' : undefined}
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
      ))}
    </div>
  );
}

export default ToastRow;
