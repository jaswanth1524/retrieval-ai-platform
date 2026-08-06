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

function TracesPanel({ traces, error, onSelect }: TracesPanelProps) {
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

  return (
    <>
      {traces.map((trace) => (
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
            <span className="traces-panel__spacer" />
            <span>{formatRelativeTime(trace.created_at * 1000)}</span>
          </span>
        </button>
      ))}
    </>
  );
}

export default TracesPanel;
