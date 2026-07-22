import { useCallback, useEffect, useState } from 'react';
import { ApiClientError, api } from '../api/client';
import type { TraceDetailResponse, TraceSummaryResponse } from '../api/types';
import './TraceBrowser.css';

interface TraceBrowserProps {
  onClose: () => void;
}

function TraceBrowser({ onClose }: TraceBrowserProps) {
  const [traces, setTraces] = useState<TraceSummaryResponse[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<TraceDetailResponse | null>(null);

  const loadList = useCallback((signal?: AbortSignal) => {
    setTraces(null);
    setError(null);
    api
      .listTraces(signal)
      .then((response) => setTraces(response.traces))
      .catch((err) => {
        if (signal?.aborted) return;
        setError(err instanceof ApiClientError ? err.message : 'Could not load traces.');
      });
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    loadList(controller.signal);
    return () => controller.abort();
  }, [loadList]);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  const openDetail = (traceId: string) => {
    setSelected(null);
    api
      .getTrace(traceId)
      .then((detail) => setSelected(detail))
      .catch(() => setSelected(null));
  };

  return (
    <div className="trace-browser" role="dialog" aria-modal="true" aria-label="Trace history">
      <div className="trace-browser__backdrop" data-testid="trace-browser-backdrop" onClick={onClose} />
      <div className="trace-browser__panel">
        <div className="trace-browser__header">
          <span className="trace-browser__title">Recent traces</span>
          <div className="trace-browser__header-actions">
            <button
              type="button"
              className="trace-browser__refresh"
              onClick={() => loadList()}
              data-testid="trace-browser-refresh"
            >
              Refresh
            </button>
            <button
              type="button"
              className="trace-browser__close"
              onClick={onClose}
              aria-label="Close trace browser"
              data-testid="trace-browser-close"
            >
              &times;
            </button>
          </div>
        </div>
        <div className="trace-browser__body">
          <ul className="trace-browser__list">
            {error && <li className="trace-browser__error" role="alert">{error}</li>}
            {!error && traces === null && <li className="trace-browser__loading">Loading…</li>}
            {traces?.length === 0 && <li className="trace-browser__empty">No traces yet.</li>}
            {traces?.map((trace) => (
              <li key={trace.trace_id}>
                <button
                  type="button"
                  className={`trace-browser__row${
                    selected?.trace_id === trace.trace_id ? ' trace-browser__row--active' : ''
                  }`}
                  onClick={() => openDetail(trace.trace_id)}
                  data-testid="trace-browser-row"
                >
                  <span className="trace-browser__question">{trace.question}</span>
                  <span className="trace-browser__row-meta mono">
                    {trace.status} &middot; {trace.mode} &middot; {trace.kept_count}/
                    {trace.candidate_count}
                    {trace.total_ms !== null ? ` · ${Math.round(trace.total_ms)}ms` : ''}
                  </span>
                </button>
              </li>
            ))}
          </ul>
          {selected && (
            <div className="trace-browser__detail" data-testid="trace-browser-detail">
              <div className="trace-browser__detail-line mono">status: {selected.status}</div>
              {selected.condensed_question && (
                <div className="trace-browser__detail-line">
                  Condensed: {selected.condensed_question}
                </div>
              )}
              {selected.query_variants && selected.query_variants.length > 0 && (
                <div className="trace-browser__detail-line">
                  Variants: {selected.query_variants.join(' | ')}
                </div>
              )}
              {selected.answer && (
                <p className="trace-browser__answer">{selected.answer}</p>
              )}
              <div className="trace-browser__candidates">
                {selected.candidates.map((candidate) => (
                  <div key={candidate.point_id} className="trace-browser__candidate mono">
                    {candidate.kept ? '✓' : '·'} {candidate.filename} p.{candidate.page}
                    {candidate.rerank_score !== null
                      ? ` (${candidate.rerank_score.toFixed(2)})`
                      : ''}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default TraceBrowser;
