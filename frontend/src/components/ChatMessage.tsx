import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { CitationResponse, TimingsResponse } from '../api/types';
import CitationCard from './CitationCard';
import StreamingSkeleton from './StreamingSkeleton';
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

  /** Copy without the async Clipboard API, which browsers gate behind a secure context.
   *
   *  DocRAG is self-hosted, so plain HTTP on a LAN address is a first-class deployment
   *  — and there `navigator.clipboard` is undefined. Hiding the button there (what used
   *  to happen) removed the feature from exactly the users the project targets. */
  const legacyCopy = (text: string): boolean => {
    const area = document.createElement('textarea');
    area.value = text;
    // Keep it out of view and off the tab order, but still selectable.
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.top = '-9999px';
    document.body.appendChild(area);
    area.select();
    try {
      return document.execCommand('copy');
    } catch {
      return false;
    } finally {
      document.body.removeChild(area);
    }
  };

  const settle = (ok: boolean) => {
    if (ok) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } else {
      setCopyFailed(true);
      setTimeout(() => setCopyFailed(false), 1500);
    }
  };

  const handleCopy = () => {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(turn.content).then(
        () => settle(true),
        // Even in a secure context the write can be denied by permissions policy;
        // fall back rather than reporting failure outright.
        () => settle(legacyCopy(turn.content)),
      );
      return;
    }
    settle(legacyCopy(turn.content));
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
              {turn.content && (
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
