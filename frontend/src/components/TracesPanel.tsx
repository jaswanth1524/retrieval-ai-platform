import { memo, useState } from 'react';
import type { TraceSummaryResponse } from '../api/types';
import { formatRelativeTime } from '../utils/relativeTime';
import { ANSWER_FILTER_STATUSES, type AnswerFilter } from '../utils/traceFilters';
import './TracesPanel.css';

interface TracesPanelProps {
  traces: TraceSummaryResponse[] | null;
  error: string | null;
  onSelect: (traceId: string) => void;
  /** The answer-status filter, when the list is fetched with it (App). */
  answerFilter?: AnswerFilter;
  onAnswerFilter?: (filter: AnswerFilter) => void;
  /** Download every trace of this workspace as JSON Lines (full key only). */
  onExport?: () => void;
  exporting?: boolean;
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

function TracesPanel({
  traces,
  error,
  onSelect,
  onExport,
  exporting = false,
  answerFilter = 'any',
  onAnswerFilter,
}: TracesPanelProps) {
  const [query, setQuery] = useState('');
  const [status, setStatus] = useState<StatusFilter>('all');
  const filterControl = onAnswerFilter && (
    <label className="traces-panel__problems mono">
      answers
      <select
        value={answerFilter}
        onChange={(event) => onAnswerFilter(event.target.value as AnswerFilter)}
        aria-label="Show traces answered this way"
        data-testid="traces-answer-filter"
      >
        <option value="any">any</option>
        <option value="answered">answered</option>
        <option value="uncited">uncited</option>
        <option value="unanswered">not answered</option>
      </select>
    </label>
  );
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
  if (traces.length === 0 && answerFilter === 'any') {
    return <p className="traces-panel__message">No traces yet.</p>;
  }

  // Client-side: the server lists its recent traces (a ring buffer) unfiltered.
  const needle = query.trim().toLowerCase();
  const wanted = ANSWER_FILTER_STATUSES[answerFilter];
  const shown = traces.filter(
    (trace) =>
      (status === 'all' || trace.status !== 'ok') &&
      (wanted.length === 0 || wanted.includes(trace.answer_status ?? '')) &&
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
        {filterControl}
        {onExport && (
          <button
            type="button"
            className="traces-panel__export mono"
            onClick={onExport}
            disabled={exporting}
            title="Download every trace as JSON Lines"
            data-testid="traces-export"
          >
            {exporting ? 'exporting…' : 'export JSONL'}
          </button>
        )}
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
            {trace.status === 'ok' && trace.answer_status === 'uncited' && (
              <span className="traces-panel__status" data-testid="traces-panel-status">
                uncited
              </span>
            )}
            {trace.top_rerank_score != null && (
              <span title="Best match score">{trace.top_rerank_score.toFixed(2)}</span>
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
