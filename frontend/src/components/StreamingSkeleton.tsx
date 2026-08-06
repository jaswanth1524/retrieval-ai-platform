import './StreamingSkeleton.css';

interface StreamingSkeletonProps {
  stage: string;
}

function StreamingSkeleton({ stage }: StreamingSkeletonProps) {
  return (
    <div className="streaming-skeleton" data-testid="streaming-skeleton">
      <div className="streaming-skeleton__bar streaming-skeleton__bar--1" />
      <div className="streaming-skeleton__bar streaming-skeleton__bar--2" />
      <div className="streaming-skeleton__bar streaming-skeleton__bar--3" />
      <div className="streaming-skeleton__stage">
        <span className="streaming-skeleton__spinner" aria-hidden="true" />
        {stage}
      </div>
    </div>
  );
}

export default StreamingSkeleton;
