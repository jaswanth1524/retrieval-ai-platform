import type { TimingsResponse, TraceCandidateResponse, TraceDetailResponse } from '../../api/types';

interface RetrievalTabProps {
  trace: TraceDetailResponse;
}

/** How far each candidate moved between vector rank and rerank rank.
 *
 * Positive means the cross-encoder promoted it. Computed here rather than server-side
 * because both orderings are already in the payload — see the mapping note in the
 * redesign plan. Candidates the reranker never scored have no rerank rank, so they get
 * no delta rather than a fabricated one.
 */
function rankDeltas(candidates: TraceCandidateResponse[]): Map<string, number | null> {
  const byRetrieval = [...candidates].sort((a, b) => b.retrieval_score - a.retrieval_score);
  const scored = candidates.filter((c) => c.rerank_score !== null);
  const byRerank = [...scored].sort((a, b) => (b.rerank_score ?? 0) - (a.rerank_score ?? 0));

  const retrievalRank = new Map(byRetrieval.map((c, i) => [c.point_id, i]));
  const rerankRank = new Map(byRerank.map((c, i) => [c.point_id, i]));

  return new Map(
    candidates.map((c) => {
      const before = retrievalRank.get(c.point_id);
      const after = rerankRank.get(c.point_id);
      if (before === undefined || after === undefined) return [c.point_id, null];
      return [c.point_id, before - after];
    }),
  );
}

function DeltaMarker({ delta }: { delta: number | null }) {
  if (delta === null) return <span className="retrieval-tab__delta mono">·</span>;
  if (delta > 0)
    return (
      <span className="retrieval-tab__delta retrieval-tab__delta--up mono" title={`up ${delta}`}>
        ▲{delta}
      </span>
    );
  if (delta < 0)
    return (
      <span className="retrieval-tab__delta retrieval-tab__delta--down mono" title={`down ${-delta}`}>
        ▼{-delta}
      </span>
    );
  return <span className="retrieval-tab__delta mono">—</span>;
}

/** Latency rows in true pipeline order.
 *
 * query expansion runs before embedding; context expansion after rerank. They were one
 * ambiguous `expand_ms` that named the former and was ordered like the latter, so this
 * list is the reason that field was split before the tab was built.
 */
function latencyRows(timings: TimingsResponse): Array<[string, number, 'io' | 'model']> {
  return [
    ['condense', timings.condense_ms, 'model'],
    ['query expand', timings.query_expansion_ms ?? 0, 'model'],
    ['embed', timings.embed_ms, 'io'],
    ['search', timings.search_ms, 'io'],
    ['rerank', timings.rerank_ms, 'model'],
    ['ctx expand', timings.context_expansion_ms ?? 0, 'io'],
    ['generate', timings.generate_ms, 'model'],
  ];
}

function RetrievalTab({ trace }: RetrievalTabProps) {
  const { candidates, timings } = trace;
  const deltas = rankDeltas(candidates);
  const keptCount = candidates.filter((c) => c.kept).length;
  const topScore = candidates.reduce<number | null>(
    (best, c) => (c.rerank_score !== null && (best === null || c.rerank_score > best) ? c.rerank_score : best),
    null,
  );
  const rows = timings ? latencyRows(timings) : [];
  // Bars are relative to the slowest stage, so the breakdown stays readable when
  // generation dwarfs everything else (routinely 100x the retrieval stages).
  const slowest = rows.reduce((max, [, ms]) => Math.max(max, ms), 0);

  return (
    <div className="retrieval-tab" data-testid="retrieval-tab">
      <div className="retrieval-tab__tiles">
        <div className="retrieval-tab__tile">
          <span className="retrieval-tab__tile-label mono">latency</span>
          <span className="retrieval-tab__tile-value mono">
            {timings ? `${(timings.total_ms / 1000).toFixed(2)}s` : '—'}
          </span>
        </div>
        <div className="retrieval-tab__tile">
          <span className="retrieval-tab__tile-label mono">chunks</span>
          <span className="retrieval-tab__tile-value mono">
            {keptCount}/{candidates.length}
          </span>
        </div>
        <div className="retrieval-tab__tile">
          <span className="retrieval-tab__tile-label mono">top score</span>
          <span className="retrieval-tab__tile-value mono">
            {topScore === null ? '—' : topScore.toFixed(2)}
          </span>
        </div>
      </div>

      {rows.length > 0 && (
        <div className="retrieval-tab__latency">
          {rows.map(([label, ms, kind]) => (
            <div key={label} className="retrieval-tab__latency-row">
              <span className="retrieval-tab__latency-label mono">{label}</span>
              <span className="retrieval-tab__latency-track">
                <span
                  className={`retrieval-tab__latency-bar retrieval-tab__latency-bar--${kind}`}
                  style={{ width: slowest > 0 ? `${(ms / slowest) * 100}%` : '0%' }}
                />
              </span>
              <span className="retrieval-tab__latency-ms mono">{ms.toFixed(0)}ms</span>
            </div>
          ))}
        </div>
      )}

      <div className="retrieval-tab__candidates">
        {candidates.map((candidate) => (
          <div
            key={candidate.point_id}
            className={`retrieval-tab__candidate${
              candidate.selected_for_context ? '' : ' retrieval-tab__candidate--dropped'
            }`}
            data-testid="retrieval-candidate"
          >
            <div className="retrieval-tab__candidate-head">
              <DeltaMarker delta={deltas.get(candidate.point_id) ?? null} />
              <span className="retrieval-tab__candidate-doc">
                {candidate.filename} &middot; {candidate.section}
              </span>
              <span className="retrieval-tab__candidate-scores mono">
                <span className="retrieval-tab__vec">{candidate.retrieval_score.toFixed(3)}</span>
                {' → '}
                <span className="retrieval-tab__rerank">
                  {candidate.rerank_score === null ? '—' : candidate.rerank_score.toFixed(3)}
                </span>
              </span>
            </div>
            <div className="retrieval-tab__candidate-meta mono">
              p.{candidate.page}
              {candidate.neighbor_expanded ? ' · expanded' : ''}
              {candidate.drop_reason ? ` · ${candidate.drop_reason.replace(/_/g, ' ')}` : ''}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default RetrievalTab;
