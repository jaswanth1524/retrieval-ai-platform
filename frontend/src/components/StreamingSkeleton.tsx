import { formatElapsed, useNow } from '../utils/elapsed';
import './StreamingSkeleton.css';

interface StreamingSkeletonProps {
  stage: string;
  /** When the question was asked (epoch ms): shows the time waited so far. */
  startedAt?: number;
  /** How long the local model should take to read its prompt, from the server. */
  etaMs?: number | null;
}

function StreamingSkeleton({ stage, startedAt, etaMs }: StreamingSkeletonProps) {
  const now = useNow(startedAt !== undefined);
  return (
    <div className="streaming-skeleton" data-testid="streaming-skeleton">
      <div className="streaming-skeleton__bar streaming-skeleton__bar--1" />
      <div className="streaming-skeleton__bar streaming-skeleton__bar--2" />
      <div className="streaming-skeleton__bar streaming-skeleton__bar--3" />
      <div className="streaming-skeleton__stage">
        <span className="streaming-skeleton__spinner" aria-hidden="true" />
        {stage}
        {/* Not announced: the thread's status line says the stage; a ticking clock read
            out every second would drown everything else. */}
        {startedAt !== undefined && (
          <span className="streaming-skeleton__elapsed" aria-hidden="true" data-testid="streaming-elapsed">
            {formatElapsed(now - startedAt)}
            {typeof etaMs === 'number' && etaMs >= 5000 && ` · reading takes about ${formatElapsed(etaMs)}`}
          </span>
        )}
      </div>
    </div>
  );
}

export default StreamingSkeleton;
