import type { ReactNode } from 'react';
import type { ApiStatus } from '../api/types';
import { useDialog } from '../hooks/useDialog';
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
  /** Narrow screens: the panel is an overlay drawer instead of a grid column. */
  drawer?: boolean;
  /** Whether the drawer is showing (ignored when not a drawer — the column always is). */
  drawerOpen?: boolean;
  onCloseDrawer?: () => void;
}

const NO_OP = () => {};

function ContextPanel({
  title,
  actionLabel,
  onAction,
  apiStatus,
  chunkTotal,
  children,
  drawer = false,
  drawerOpen = false,
  onCloseDrawer = NO_OP,
}: ContextPanelProps) {
  const showingDrawer = drawer && drawerOpen;
  const panelRef = useDialog<HTMLElement>(showingDrawer, onCloseDrawer);
  return (
    <>
      {showingDrawer && (
        <div className="drawer-backdrop" onClick={onCloseDrawer} data-testid="context-panel-backdrop" />
      )}
      <aside
        ref={panelRef}
        className={`context-panel${showingDrawer ? ' context-panel--drawer-open' : ''}`}
        data-testid="context-panel"
        {...(showingDrawer ? { role: 'dialog', 'aria-modal': true, 'aria-label': title } : {})}
      >
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
    </>
  );
}

export default ContextPanel;
