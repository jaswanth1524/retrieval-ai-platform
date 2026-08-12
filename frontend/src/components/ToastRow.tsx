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
  if (toasts.length === 0) return null;

  return (
    <div className="toast-row" data-testid="toast-row">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`toast toast--${toast.tone}`}
          // Failures must interrupt a screen reader; confirmations must not.
          role={toast.tone === 'bad' ? 'alert' : 'status'}
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
