import type { ReactNode } from 'react';
import type { ApiStatus } from '../api/types';
import './ContextPanel.css';

// A Record, not a ternary chain: adding a status now fails to compile until it has a
// label, instead of silently falling through to "checking".
const API_STATUS_LABELS: Record<ApiStatus, string> = {
  ok: 'api ok',
  error: 'api unreachable',
  checking: 'checking',
  unauthorized: 'api key required',
};

interface ContextPanelProps {
  title: string;
  actionLabel: string;
  onAction: () => void;
  apiStatus: ApiStatus;
  chunkTotal: number;
  children: ReactNode;
}

function ContextPanel({ title, actionLabel, onAction, apiStatus, chunkTotal, children }: ContextPanelProps) {
  return (
    <aside className="context-panel">
      <header className="context-panel__header">
        <h2 className="context-panel__title">{title}</h2>
        <button type="button" className="context-panel__action" onClick={onAction} data-testid="context-panel-action">
          {actionLabel}
        </button>
      </header>
      <div className="context-panel__body">{children}</div>
      <footer className="context-panel__footer" data-testid="context-panel-footer">
        <span
          className={`context-panel__status-dot context-panel__status-dot--${apiStatus}`}
          aria-hidden="true"
        />
        <span>{API_STATUS_LABELS[apiStatus]}</span>
        <span className="context-panel__spacer" />
        <span>{chunkTotal} chunks</span>
      </footer>
    </aside>
  );
}

export default ContextPanel;
