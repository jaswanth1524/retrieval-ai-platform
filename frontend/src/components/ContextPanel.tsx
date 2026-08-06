import type { ReactNode } from 'react';
import type { ApiStatus } from '../api/types';
import './ContextPanel.css';

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
        <span>{apiStatus === 'ok' ? 'api ok' : apiStatus === 'error' ? 'api unreachable' : 'checking'}</span>
        <span className="context-panel__spacer" />
        <span>{chunkTotal} chunks</span>
      </footer>
    </aside>
  );
}

export default ContextPanel;
