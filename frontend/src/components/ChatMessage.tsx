import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { CitationResponse, TimingsResponse } from '../api/types';
import CitationCard from './CitationCard';
import StreamingSkeleton from './StreamingSkeleton';
import TraceDrawer from './TraceDrawer';
import './ChatMessage.css';

export interface ChatTurn {
  id: string;
  role: 'user' | 'assistant' | 'error';
  content: string;
  sources: CitationResponse[];
  timestamp: number;
  timings: TimingsResponse | null;
  traceId: string | null;
  // The question that produced this turn — set on error turns only, so a failed
  // question's text isn't lost and can be resubmitted via the Retry button.
  question?: string;
}

interface ChatMessageProps {
  turn: ChatTurn;
  engineerMode: boolean;
  // Set only on the single turn actively streaming; undefined means "not this one".
  streamStage?: string;
  currentModelLabel?: string;
  onRetry?: (question: string) => void;
  onOpenSource?: (filename: string, chunkId: string) => void;
  onCitationHover?: (citation: CitationResponse) => void;
  onCitationLeave?: () => void;
}

function formatDuration(totalMs: number): string {
  return `${(totalMs / 1000).toFixed(2)}s`;
}

function ChatMessage({
  turn,
  engineerMode,
  streamStage,
  currentModelLabel,
  onRetry,
  onOpenSource,
  onCitationHover,
  onCitationLeave,
}: ChatMessageProps) {
  const [copied, setCopied] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);

  const handleCopy = () => {
    if (!navigator.clipboard) return;
    navigator.clipboard.writeText(turn.content).then(
      () => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      },
      () => {
        setCopyFailed(true);
        setTimeout(() => setCopyFailed(false), 1500);
      },
    );
  };

  const showSkeleton = streamStage !== undefined && turn.content === '';
  const timings = turn.timings;
  const showMeta = engineerMode && turn.role === 'assistant' && !showSkeleton && timings !== null;

  return (
    <div className="chat-message" data-testid="chat-message" data-role={turn.role}>
      {turn.role === 'user' && (
        <div className="chat-message__row">
          <span className="chat-message__gutter" aria-hidden="true">
            Q
          </span>
          <p className="chat-message__question">{turn.content}</p>
        </div>
      )}

      {turn.role === 'assistant' && (
        <div className="chat-message__row">
          <span className="chat-message__gutter chat-message__gutter--answer" aria-hidden="true">
            A
          </span>
          <div className="chat-message__answer">
            {showSkeleton ? (
              <StreamingSkeleton stage={streamStage} />
            ) : (
              <div className="chat-message__markdown">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{turn.content}</ReactMarkdown>
              </div>
            )}
            <div className="chat-message__meta-row">
              <time
                className="chat-message__timestamp mono"
                dateTime={new Date(turn.timestamp).toISOString()}
              >
                {new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit' }).format(
                  turn.timestamp,
                )}
              </time>
              {turn.content && navigator.clipboard && (
                <button
                  type="button"
                  className="chat-message__copy"
                  onClick={handleCopy}
                  data-testid="chat-message-copy"
                >
                  {copied ? 'Copied' : copyFailed ? 'Copy failed' : 'Copy'}
                </button>
              )}
            </div>
            {turn.sources.length > 0 && (
              <div className="chat-message__citations" data-testid="chat-message-sources">
                {turn.sources.map((source) => (
                  <CitationCard
                    key={`${turn.id}-${source.source_number}`}
                    sourceNumber={source.source_number}
                    filename={source.filename}
                    page={source.page}
                    section={source.section}
                    onOpen={onOpenSource ? () => onOpenSource(source.filename, source.chunk_id) : undefined}
                    onHoverStart={onCitationHover ? () => onCitationHover(source) : undefined}
                    onHoverEnd={onCitationLeave}
                  />
                ))}
              </div>
            )}
            {showMeta && timings && (
              <div className="chat-message__engineer-meta mono">
                <span>{formatDuration(timings.total_ms)}</span>
                {currentModelLabel && <span>{currentModelLabel}</span>}
              </div>
            )}
            {turn.traceId && <TraceDrawer traceId={turn.traceId} timings={turn.timings} />}
          </div>
        </div>
      )}

      {turn.role === 'error' && (
        <div className="chat-message__row">
          <span className="chat-message__gutter chat-message__gutter--error" aria-hidden="true">
            !
          </span>
          <div className="chat-message__error" role="alert">
            {turn.content}
            {turn.question && onRetry && (
              <button
                type="button"
                className="chat-message__retry"
                onClick={() => onRetry(turn.question!)}
                data-testid="chat-message-retry"
              >
                Retry
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default ChatMessage;
