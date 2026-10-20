import { memo, useState } from 'react';
import type { TraceSummaryResponse } from '../api/types';
import { formatRelativeTime } from '../utils/relativeTime';
import './TracesPanel.css';

interface TracesPanelProps {
  traces: TraceSummaryResponse[] | null;
  error: string | null;
  onSelect: (traceId: string) => void;
}

function latencyClass(totalMs: number | null): string {
  if (totalMs === null) return 'traces-panel__latency--dim';
  if (totalMs < 3000) return 'traces-panel__latency--good';
  if (totalMs <= 6000) return 'traces-panel__latency--warn';
  return 'traces-panel__latency--bad';
}

function formatLatency(totalMs: number | null): string {
  if (totalMs === null) return '—';
  return `${(totalMs / 1000).toFixed(2)}s`;
}

type StatusFilter = 'all' | 'problems';

function TracesPanel({ traces, error, onSelect }: TracesPanelProps) {
  const [query, setQuery] = useState('');
  const [status, setStatus] = useState<StatusFilter>('all');
  if (error) {
    return (
      <p className="traces-panel__message" role="alert">
        {error}
      </p>
    );
  }
  if (traces === null) {
    return <p className="traces-panel__message">Loading…</p>;
  }
  if (traces.length === 0) {
    return <p className="traces-panel__message">No traces yet.</p>;
  }

  // Client-side: the server lists its recent traces (a ring buffer) unfiltered.
  const needle = query.trim().toLowerCase();
  const shown = traces.filter(
    (trace) =>
      (status === 'all' || trace.status !== 'ok') &&
      (!needle || trace.question.toLowerCase().includes(needle)),
  );
  return (
    <>
      <div className="traces-panel__filters">
        <input
          type="search"
          className="traces-panel__search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Filter by question…"
          aria-label="Filter traces by question"
          data-testid="traces-filter"
        />
        <label className="traces-panel__problems mono">
          <input
            type="checkbox"
            checked={status === 'problems'}
            onChange={(event) => setStatus(event.target.checked ? 'problems' : 'all')}
            data-testid="traces-problems-only"
          />
          errors &amp; no context only
        </label>
      </div>
      {shown.length === 0 && (
        <p className="traces-panel__message">No traces match.</p>
      )}
      {shown.map((trace) => (
        <button
          key={trace.trace_id}
          type="button"
          className="traces-panel__row"
          onClick={() => onSelect(trace.trace_id)}
          data-testid="traces-panel-row"
        >
          <span className="traces-panel__query">{trace.question}</span>
          <span className="traces-panel__meta mono">
            <span className={latencyClass(trace.total_ms)}>{formatLatency(trace.total_ms)}</span>
            <span>{trace.llm_provider ?? '—'}</span>
            {trace.status !== 'ok' && (
              <span
                className={`traces-panel__status traces-panel__status--${trace.status}`}
                data-testid="traces-panel-status"
              >
                {trace.status === 'error' ? 'error' : 'no context'}
              </span>
            )}
            <span className="traces-panel__spacer" />
            <span>{formatRelativeTime(trace.created_at * 1000)}</span>
          </span>
        </button>
      ))}
    </>
  );
}

// Memoized: App re-renders on every streamed frame, and none of this depends on it.
export default memo(TracesPanel);
