import { useEffect } from 'react';
import type { CitationResponse } from '../api/types';
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
}: InspectorProps) {
  const tabs = engineerMode ? ENGINEER_TABS : READER_TABS;
  const state = useTrace(traceId, traceReady);

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

  return (
    <aside className="inspector" data-testid="inspector" aria-label="Inspector">
      <div className="inspector__tabs" role="tablist" aria-label="Inspector views">
        {tabs.map((name) => (
          <button
            key={name}
            type="button"
            role="tab"
            aria-selected={activeTab === name}
            className={`inspector__tab${activeTab === name ? ' inspector__tab--active' : ''}`}
            onClick={() => onTabChange(name)}
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
      <div className="inspector__body" role="tabpanel">
        {renderBody()}
      </div>
    </aside>
  );
}

export default Inspector;
