import { useState } from 'react';
import { api, apiErrorMessage } from '../../api/client';
import type { TraceDetailResponse, TraceRerunResponse } from '../../api/types';

interface TraceRerunProps {
  trace: TraceDetailResponse;
}

/** "Search again with…": the traced question searched again under another relevance
 *  bar or passage count — would it have been answered? — with how the ranking moved.
 *  Retrieval only: no answer is written and nothing is stored. */
export default function TraceRerun({ trace }: TraceRerunProps) {
  const config = trace.config;
  const [minScore, setMinScore] = useState(String(config?.rerank_min_score ?? ''));
  const [topK, setTopK] = useState(String(config?.rerank_top_k ?? ''));
  const [result, setResult] = useState<TraceRerunResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  if (!config || (trace.kind ?? 'question') !== 'question') return null;

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      const score = Number.parseFloat(minScore);
      const passages = Number.parseInt(topK, 10);
      setResult(
        await api.rerunTrace(trace.trace_id, {
          ...(Number.isFinite(score) ? { rerank_min_score: score } : {}),
          ...(Number.isInteger(passages) ? { rerank_top_k: passages } : {}),
        }),
      );
    } catch (err) {
      setError(apiErrorMessage(err, 'Could not search again.'));
    } finally {
      setRunning(false);
    }
  };

  return (
    <section className="trace-rerun" aria-label="Search again with other settings" data-testid="trace-rerun">
      <form
        className="trace-rerun__form"
        onSubmit={(event) => {
          event.preventDefault();
          void run();
        }}
      >
        <label>
          Relevance bar
          <input
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={minScore}
            onChange={(event) => setMinScore(event.target.value)}
            data-testid="trace-rerun-min-score"
          />
        </label>
        <label>
          Passages
          <input
            type="number"
            min={1}
            max={50}
            value={topK}
            onChange={(event) => setTopK(event.target.value)}
            data-testid="trace-rerun-top-k"
          />
        </label>
        <button type="submit" disabled={running} data-testid="trace-rerun-submit">
          {running ? 'Searching…' : 'Search again'}
        </button>
      </form>
      {error && (
        <p className="trace-tab__error" role="alert">
          {error}
        </p>
      )}
      {result && (
        <div className="trace-rerun__result" role="status" data-testid="trace-rerun-result">
          <p>
            {result.would_answer
              ? 'Passages would reach the model: it would be answered from them.'
              : result.weak_matches
                ? 'Nothing clears that bar: only the closest passages would be sent.'
                : 'Nothing would be found.'}
          </p>
          {result.diff.candidates.length > 0 && (
            <ul className="trace-rerun__changes">
              {result.diff.candidates.slice(0, 12).map((change) => (
                <li key={change.point_id} className="mono" data-testid="trace-rerun-change">
                  {change.filename} p.{change.page}
                  {change.section ? ` · ${change.section}` : ''}: rank {change.rank_a ?? '–'} → {change.rank_b ?? '–'}
                  {change.selected_a !== change.selected_b
                    ? change.selected_b
                      ? ' · now in the prompt'
                      : ' · no longer in the prompt'
                    : ''}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
