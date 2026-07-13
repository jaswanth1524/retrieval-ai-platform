import { useEffect, useRef, useState } from 'react';
import { ApiClientError, api } from '../api/client';
import type { TimingsResponse, TraceCandidateResponse, TraceDetailResponse } from '../api/types';
import './TraceDrawer.css';

interface TraceDrawerProps {
  traceId: string;
  timings: TimingsResponse | null;
}

const DROP_REASON_LABEL: Record<string, string> = {
  below_min_score: 'below threshold',
  top_k_cut: 'top-k cut',
  not_scored: 'not scored',
};

function candidateStatusLabel(candidate: TraceCandidateResponse): string {
  if (candidate.selected_for_context) return 'in context';
  if (candidate.kept) return 'kept';
  return DROP_REASON_LABEL[candidate.drop_reason ?? ''] ?? 'dropped';
}

function candidateStatusModifier(candidate: TraceCandidateResponse): 'selected' | 'kept' | 'dropped' {
  if (candidate.selected_for_context) return 'selected';
  if (candidate.kept) return 'kept';
  return 'dropped';
}

function TraceDrawer({ traceId, timings }: TraceDrawerProps) {
  const [open, setOpen] = useState(false);
  const [trace, setTrace] = useState<TraceDetailResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Guards the fetch-once-per-traceId behavior: re-collapsing and re-expanding the
  // drawer must not refetch, but a failed fetch should still be retryable.
  const fetchedRef = useRef(false);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      controllerRef.current?.abort();
    };
  }, []);

  const handleToggle = () => {
    const next = !open;
    setOpen(next);
    if (!next || fetchedRef.current) return;

    fetchedRef.current = true;
    setLoading(true);
    setError(null);
    const controller = new AbortController();
    controllerRef.current = controller;
    api
      .getTrace(traceId, controller.signal)
      .then(setTrace)
      .catch((err: unknown) => {
        fetchedRef.current = false; // allow a retry on the next expand
        setError(
          err instanceof ApiClientError && err.statusCode === 404
            ? 'Trace no longer available (server restarted or trace evicted).'
            : 'Failed to load trace.',
        );
      })
      .finally(() => setLoading(false));
  };

  const effectiveTimings = trace?.timings ?? timings;

  return (
    <div className="trace-drawer" data-testid="trace-drawer">
      <button
        type="button"
        className="trace-drawer__toggle"
        aria-expanded={open}
        data-testid="trace-drawer-toggle"
        onClick={handleToggle}
      >
        {open ? 'Hide debug trace' : 'Debug trace'}
      </button>
      {open && (
        <div className="trace-drawer__panel" data-testid="trace-drawer-panel">
          {loading && <p className="trace-drawer__status">Loading trace...</p>}
          {error && (
            <p className="trace-drawer__status trace-drawer__status--error" data-testid="trace-drawer-error">
              {error}
            </p>
          )}

          {effectiveTimings && (
            <div className="trace-drawer__section">
              <h4 className="trace-drawer__heading">Timings</h4>
              <div className="trace-drawer__timings">
                {(
                  [
                    ['embed', effectiveTimings.embed_ms],
                    ['search', effectiveTimings.search_ms],
                    ['rerank', effectiveTimings.rerank_ms],
                    ['generate', effectiveTimings.generate_ms],
                    ['total', effectiveTimings.total_ms],
                  ] as const
                ).map(([label, ms]) => (
                  <span key={label} className="trace-drawer__badge mono">
                    {label} {ms.toFixed(0)}ms
                  </span>
                ))}
              </div>
            </div>
          )}

          {trace?.config && (
            <div className="trace-drawer__section">
              <h4 className="trace-drawer__heading">Config</h4>
              <dl className="trace-drawer__config mono">
                <dt>provider</dt>
                <dd>{trace.config.llm_provider}</dd>
                <dt>rerank_top_k</dt>
                <dd>{trace.config.rerank_top_k}</dd>
                <dt>max_context_chunks</dt>
                <dd>{trace.config.max_context_chunks}</dd>
                <dt>temperature</dt>
                <dd>{trace.config.llm_temperature}</dd>
                <dt>rerank_min_score</dt>
                <dd>{trace.config.rerank_min_score}</dd>
                <dt>scope</dt>
                <dd>{trace.config.filenames ? trace.config.filenames.join(', ') : 'all documents'}</dd>
              </dl>
            </div>
          )}

          {trace && trace.candidates.length > 0 && (
            <div className="trace-drawer__section">
              <h4 className="trace-drawer__heading">Candidates ({trace.candidates.length})</h4>
              <table className="trace-drawer__table">
                <thead>
                  <tr>
                    <th>chunk</th>
                    <th>retrieval</th>
                    <th>rerank</th>
                    <th>status</th>
                  </tr>
                </thead>
                <tbody>
                  {trace.candidates.map((candidate) => (
                    <tr
                      key={candidate.point_id}
                      className={`trace-drawer__row trace-drawer__row--${candidateStatusModifier(candidate)}`}
                      data-testid="trace-drawer-candidate"
                    >
                      <td className="mono">
                        {candidate.filename} &middot; p.{candidate.page} &middot; {candidate.section}
                        {candidate.neighbor_expanded ? ' (expanded)' : ''}
                      </td>
                      <td className="mono">{candidate.retrieval_score.toFixed(4)}</td>
                      <td className="mono">
                        {candidate.rerank_score === null ? '—' : candidate.rerank_score.toFixed(4)}
                      </td>
                      <td>{candidateStatusLabel(candidate)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {trace?.prompt_messages && (
            <div className="trace-drawer__section">
              <h4 className="trace-drawer__heading">Prompt</h4>
              {trace.prompt_messages.map((message, index) => (
                <details key={index} className="trace-drawer__prompt-message">
                  <summary className="mono">{message.role}</summary>
                  <pre className="trace-drawer__prompt-content mono">{message.content}</pre>
                </details>
              ))}
            </div>
          )}

          {trace && (
            <div className="trace-drawer__section">
              <h4 className="trace-drawer__heading">Answer</h4>
              {trace.status === 'error' ? (
                <p className="trace-drawer__status trace-drawer__status--error">{trace.error}</p>
              ) : (
                <>
                  <p className="trace-drawer__answer">{trace.answer}</p>
                  {trace.cited_source_numbers.length > 0 && (
                    <p className="trace-drawer__status mono">
                      cited: {trace.cited_source_numbers.map((n) => `[${n}]`).join(' ')}
                    </p>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default TraceDrawer;
