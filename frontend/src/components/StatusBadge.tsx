import './StatusBadge.css';

export type ApiStatus = 'checking' | 'ok' | 'error';

interface StatusBadgeProps {
  status: ApiStatus;
  message?: string;
}

const LABELS: Record<ApiStatus, string> = {
  checking: 'Checking...',
  ok: 'API online',
  error: 'API unreachable',
};

function StatusBadge({ status, message }: StatusBadgeProps) {
  return (
    <div className="status-badge" data-testid="status-badge" data-status={status}>
      <span className={`status-badge__dot status-badge__dot--${status}`} />
      <span className="status-badge__label">{LABELS[status]}</span>
      {message && status === 'error' && <p className="status-badge__message mono">{message}</p>}
    </div>
  );
}

export default StatusBadge;
