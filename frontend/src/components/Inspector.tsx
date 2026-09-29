import { useEffect, useId } from 'react';
import type { KeyboardEvent } from 'react';
import type { CitationResponse } from '../api/types';
import { useDialog } from '../hooks/useDialog';
import { useTrace } from '../hooks/useTrace';
import RetrievalTab from './inspector/RetrievalTab';
import SourcesTab from './inspector/SourcesTab';
import TraceTab from './inspector/TraceTab';
import './Inspector.css';

export type InspectorTab = 'sources' | 'retrieval' | 'trace';

/** Reader mode is deliberately Sources-only: Retrieval and Trace exist to debug
 *  retrieval, which is not what a Reader-mode user is doing. */
const READER_TABS: InspectorTab[] = ['sources'];
const ENGINEER_TABS: InspectorTab[] = ['sources', 'retrieval', 'trace'];

interface InspectorProps {
  engineerMode: boolean;
  tab: InspectorTab;
  onTabChange: (tab: InspectorTab) => void;
  onClose: () => void;
  /** Sources of the turn being inspected; empty before the first answer. */
  sources: CitationResponse[];
  /** Trace id of the turn being inspected, or null when none was recorded. */
  traceId: string | null;
  /** False while the inspected turn is still streaming — see useTrace's docstring for
   *  why fetching before this flips true reliably 404s. Defaults true: a trace pinned
   *  from the trace browser is always for an already-completed turn. */
  traceReady?: boolean;
  onOpenSource?: (filename: string, chunkId: string) => void;
  /** Too narrow for a fourth column: show as a drawer over the chat instead. */
  overlay?: boolean;
}

function Inspector({
  engineerMode,
  tab,
  onTabChange,
  onClose,
  sources,
  traceId,
  traceReady = true,
  onOpenSource,
  overlay = false,
}: InspectorProps) {
  const tabs = engineerMode ? ENGINEER_TABS : READER_TABS;
  const state = useTrace(traceId, traceReady);
  const panelRef = useDialog<HTMLElement>(overlay, onClose);
  const idPrefix = useId();
  const tabId = (name: InspectorTab) => `${idPrefix}-tab-${name}`;
  const panelId = `${idPrefix}-panel`;

  // Leaving Engineer mode with Retrieval or Trace open would strand the panel on a tab
  // its own tab bar no longer offers.
  useEffect(() => {
    if (!tabs.includes(tab)) onTabChange('sources');
  }, [tabs, tab, onTabChange]);

  const activeTab = tabs.includes(tab) ? tab : 'sources';

  const renderBody = () => {
    if (activeTab === 'sources') {
      return (
        <SourcesTab
          sources={sources}
          candidates={state.status === 'ready' ? state.trace.candidates : undefined}
          onOpenSource={onOpenSource}
        />
      );
    }
    // Retrieval and Trace are trace-backed, so they share its loading/absent states.
    if (state.status === 'pending') {
      return (
        <p className="inspector__empty" data-testid="inspector-trace-pending">
          {state.reason}
        </p>
      );
    }
    if (state.status === 'loading') {
      return <p className="inspector__empty">Loading trace…</p>;
    }
    if (state.status === 'unavailable') {
      return (
        <p className="inspector__empty" data-testid="inspector-trace-unavailable">
          {state.reason}
        </p>
      );
    }
    if (state.status === 'error') {
      return (
        <p className="inspector__empty inspector__empty--error" role="alert">
          {state.message}
        </p>
      );
    }
    return activeTab === 'retrieval' ? (
      <RetrievalTab trace={state.trace} />
    ) : (
      <TraceTab trace={state.trace} />
    );
  };

  // WAI-ARIA tabs: one tab stop for the whole tablist, arrows move between tabs.
  const handleTabKey = (event: KeyboardEvent<HTMLButtonElement>) => {
    const index = tabs.indexOf(activeTab);
    let next: InspectorTab | undefined;
    if (event.key === 'ArrowRight') next = tabs[(index + 1) % tabs.length];
    else if (event.key === 'ArrowLeft') next = tabs[(index - 1 + tabs.length) % tabs.length];
    else if (event.key === 'Home') next = tabs[0];
    else if (event.key === 'End') next = tabs[tabs.length - 1];
    if (!next) return;
    event.preventDefault();
    onTabChange(next);
    document.getElementById(tabId(next))?.focus();
  };

  return (
    <>
      {overlay && (
        <div className="drawer-backdrop" onClick={onClose} data-testid="inspector-backdrop" />
      )}
      <aside
        ref={panelRef}
        className={`inspector${overlay ? ' inspector--overlay' : ''}`}
        data-testid="inspector"
        aria-label="Inspector"
        {...(overlay ? { role: 'dialog', 'aria-modal': true } : {})}
      >
        <div className="inspector__tabs" role="tablist" aria-label="Inspector views">
          {tabs.map((name) => (
            <button
              key={name}
              id={tabId(name)}
              type="button"
              role="tab"
              aria-selected={activeTab === name}
              aria-controls={panelId}
              tabIndex={activeTab === name ? 0 : -1}
              className={`inspector__tab${activeTab === name ? ' inspector__tab--active' : ''}`}
              onClick={() => onTabChange(name)}
              onKeyDown={handleTabKey}
              data-testid={`inspector-tab-${name}`}
            >
              {name}
            </button>
          ))}
          <span className="inspector__tab-spacer" />
          <button
            type="button"
            className="inspector__close"
            onClick={onClose}
            aria-label="Close inspector"
            data-testid="inspector-close"
          >
            <span aria-hidden="true">✕</span>
          </button>
        </div>
        <div
          className="inspector__body"
          role="tabpanel"
          id={panelId}
          aria-labelledby={tabId(activeTab)}
        >
          {renderBody()}
        </div>
      </aside>
    </>
  );
}

export default Inspector;
