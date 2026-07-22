import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { CitationResponse, TimingsResponse } from '../api/types';
import CitationCard from './CitationCard';
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
  onRetry?: (question: string) => void;
  onOpenSource?: (filename: string, chunkId: string) => void;
}

const TIME_FORMATTER = new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit' });

function ChatMessage({ turn, onRetry, onOpenSource }: ChatMessageProps) {
  const [copied, setCopied] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState(false);

  const handleCopy = () => {
    if (!navigator.clipboard) return;
    navigator.clipboard.writeText(turn.content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <div
      className={`chat-message chat-message--${turn.role}`}
      data-testid="chat-message"
      data-role={turn.role}
    >
      {turn.role === 'error' ? (
        <div className="chat-message__bubble chat-message__bubble--error" role="alert">
          <span className="chat-message__error-icon" aria-hidden="true">
            ⚠
          </span>
          <div className="chat-message__error-body">
            <div className="chat-message__error-lead">Generation provider unavailable</div>
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
      ) : turn.role === 'assistant' ? (
        <div className="chat-message__bubble chat-message__bubble--markdown">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{turn.content}</ReactMarkdown>
        </div>
      ) : (
        <div className="chat-message__bubble">{turn.content}</div>
      )}
      <div className="chat-message__meta">
        <time className="chat-message__timestamp mono" dateTime={new Date(turn.timestamp).toISOString()}>
          {TIME_FORMATTER.format(turn.timestamp)}
        </time>
        {turn.role === 'assistant' && turn.content && navigator.clipboard && (
          <button
            type="button"
            className="chat-message__copy"
            onClick={handleCopy}
            data-testid="chat-message-copy"
          >
            {copied ? 'Copied' : 'Copy'}
          </button>
        )}
        {turn.role === 'assistant' && turn.sources.length > 0 && (
          <button
            type="button"
            className="chat-message__sources-toggle"
            onClick={() => setSourcesOpen((prev) => !prev)}
            aria-expanded={sourcesOpen}
            data-testid="chat-message-sources-toggle"
          >
            {sourcesOpen ? 'Hide sources' : `Sources (${turn.sources.length})`}
          </button>
        )}
      </div>
      {turn.role === 'assistant' && turn.sources.length > 0 && sourcesOpen && (
        <div className="chat-message__sources" data-testid="chat-message-sources">
          {turn.sources.map((source) => (
            <CitationCard
              key={`${turn.id}-${source.source_number}`}
              sourceNumber={source.source_number}
              filename={source.filename}
              page={source.page}
              section={source.section}
              chunkId={source.chunk_id}
              text={source.text}
              onOpen={
                onOpenSource ? () => onOpenSource(source.filename, source.chunk_id) : undefined
              }
            />
          ))}
        </div>
      )}
      {turn.role === 'assistant' && turn.traceId && (
        <TraceDrawer traceId={turn.traceId} timings={turn.timings} />
      )}
    </div>
  );
}

export default ChatMessage;
