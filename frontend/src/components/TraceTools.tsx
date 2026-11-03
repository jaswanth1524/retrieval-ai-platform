import { useEffect, useRef, useState } from 'react';
import { api, apiErrorMessage } from '../api/client';
import type { ScoreReportResponse, TraceDiffResponse } from '../api/types';

/** One request's state: loading, the answer, or the reason it failed. */
function useFetched<T>(load: (signal: AbortSignal) => Promise<T>, key: string) {
  const [state, setState] = useState<{ key: string; value?: T; error?: string } | null>(null);
  // Built inline by the caller: `key` says when it asks something new.
  const loadRef = useRef(load);
  loadRef.current = load;
  useEffect(() => {
    const controller = new AbortController();
    loadRef.current(controller.signal).then(
      (value) => setState({ key, value }),
      (err) => {
        if (!controller.signal.aborted) setState({ key, error: apiErrorMessage(err, 'Try again.') });
      },
    );
    return () => controller.abort();
  }, [key]);
  return state?.key === key ? state : null;
}

/** The best rerank score of each traced question, answered against not answered, and
 *  the RERANK_MIN_SCORE that would separate them best on this corpus. */
export function ScoreReportView({ onClose }: { onClose: () => void }) {
  const fetched = useFetched<ScoreReportResponse>((signal) => api.scoreReport(signal), 'report');
  const report = fetched?.value;
  const tallest = report ? Math.max(1, ...report.answered, ...report.unanswered) : 1;
  return (
    <section className="trace-tools" aria-label="Relevance score report" data-testid="score-report">
      <div className="trace-tools__head">
        <span>Score report</span>
        <button type="button" onClick={onClose} aria-label="Close the score report">
          ✕
        </button>
      </div>
      {!fetched && <p>Loading…</p>}
      {fetched?.error && <p role="alert">{fetched.error}</p>}
      {report && (
        <>
          <p className="mono">
            {report.answered_count} answered · {report.unanswered_count} not answered · the
            bar now {report.current_min_score.toFixed(2)}
          </p>
          <div className="trace-tools__bars" role="img" aria-label="Best match scores, answered against not answered">
            {report.answered.map((yes, index) => {
              const no = report.unanswered[index];
              if (!yes && !no) return null;
              return (
                <div key={index} className="trace-tools__bar-row mono" data-testid="score-report-bin">
                  <span>{report.bin_edges[index].toFixed(2)}</span>
                  <span className="trace-tools__bar trace-tools__bar--yes" style={{ width: `${(yes / tallest) * 45}%` }} />
                  <span className="trace-tools__bar trace-tools__bar--no" style={{ width: `${(no / tallest) * 45}%` }} />
                  <span>
                    {yes}/{no}
                  </span>
                </div>
              );
            })}
          </div>
          {report.suggested_min_score != null && (
            <p data-testid="score-report-suggestion">
              Suggested <span className="mono">RERANK_MIN_SCORE={report.suggested_min_score}</span>
              {report.would_change ? ` — ${report.would_change} traced question(s) decided differently` : ''}
            </p>
          )}
          <p className="trace-tools__note">{report.note}</p>
        </>
      )}
    </section>
  );
}

function shown(value: unknown): string {
  if (value === null || value === undefined) return '—';
  return typeof value === 'object' ? JSON.stringify(value) : String(value);
}

/** What differs between two traces: the settings, and the passages that moved. */
export function TraceDiffView({ a, b, onClose }: { a: string; b: string; onClose: () => void }) {
  const fetched = useFetched<TraceDiffResponse>((signal) => api.diffTraces(a, b, signal), `${a}|${b}`);
  const diff = fetched?.value;
  return (
    <section className="trace-tools" aria-label="Two traces compared" data-testid="trace-diff">
      <div className="trace-tools__head">
        <span>Compared</span>
        <button type="button" onClick={onClose} aria-label="Close the comparison">
          ✕
        </button>
      </div>
      {!fetched && <p>Loading…</p>}
      {fetched?.error && <p role="alert">{fetched.error}</p>}
      {diff && (
        <>
          <p data-testid="trace-diff-answer">
            {diff.same_answer ? 'Same answer.' : 'Different answers.'}
            {diff.question_a !== diff.question_b ? ' (Different questions.)' : ''}
          </p>
          {diff.settings.length > 0 ? (
            <ul className="mono">
              {diff.settings.map((change) => (
                <li key={change.name} data-testid="trace-diff-setting">
                  {change.name}: {shown(change.a)} → {shown(change.b)}
                </li>
              ))}
            </ul>
          ) : (
            <p>Same settings.</p>
          )}
          {diff.candidates.length === 0 ? (
            <p>The same passages, ranked the same.</p>
          ) : (
            <ul className="mono">
              {diff.candidates.map((change) => (
                <li key={change.point_id} data-testid="trace-diff-candidate">
                  {change.filename} p.{change.page}: rank {shown(change.rank_a)} → {shown(change.rank_b)}
                  {change.selected_a !== change.selected_b &&
                    (change.selected_b ? ' · now in the prompt' : ' · no longer in the prompt')}
                  {change.drop_reason_b ? ` · ${change.drop_reason_b}` : ''}
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  );
}
