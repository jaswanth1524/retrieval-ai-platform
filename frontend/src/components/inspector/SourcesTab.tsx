import type { CitationResponse, TraceCandidateResponse } from '../../api/types';

interface SourcesTabProps {
  sources: CitationResponse[];
  /** Trace candidates, when loaded — supplies each source's rerank score for the track. */
  candidates?: TraceCandidateResponse[];
  onOpenSource?: (filename: string, chunkId: string) => void;
}

/** Score for a source, matched to its trace candidate by chunk id.
 *
 * The citation payload carries no score — only the trace does — so this is null until
 * the trace loads, and stays null when tracing is off. The track is hidden rather than
 * drawn at zero, which would read as "scored 0.00" instead of "not known".
 */
function scoreFor(
  source: CitationResponse,
  candidates: TraceCandidateResponse[] | undefined,
): number | null {
  const match = candidates?.find((candidate) => candidate.chunk_id === source.chunk_id);
  return match?.rerank_score ?? null;
}

function SourcesTab({ sources, candidates, onOpenSource }: SourcesTabProps) {
  if (sources.length === 0) {
    return (
      <p className="inspector__empty" data-testid="sources-empty">
        No sources for this turn yet. Ask a question to see what it retrieved.
      </p>
    );
  }

  return (
    <div className="sources-tab" data-testid="sources-tab">
      {sources.map((source) => {
        const score = scoreFor(source, candidates);
        return (
          <button
            key={`${source.source_number}-${source.chunk_id}`}
            type="button"
            className="sources-tab__card"
            onClick={onOpenSource ? () => onOpenSource(source.filename, source.chunk_id) : undefined}
            disabled={!onOpenSource}
            data-testid="sources-tab-card"
          >
            <span className="sources-tab__head mono">
              <span className="sources-tab__number" aria-hidden="true">
                {source.source_number}
              </span>
              <span className="sources-tab__filename">{source.filename}</span>
              <span className="sources-tab__location">
                p.{source.page} &middot; {source.section}
              </span>
            </span>
            <span className="sources-tab__quote">{source.text}</span>
            {score !== null && (
              <span className="sources-tab__score-row">
                <span className="sources-tab__track">
                  <span
                    className="sources-tab__fill"
                    style={{ width: `${Math.max(0, Math.min(1, score)) * 100}%` }}
                  />
                </span>
                <span className="sources-tab__score mono">{score.toFixed(2)}</span>
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

export default SourcesTab;
