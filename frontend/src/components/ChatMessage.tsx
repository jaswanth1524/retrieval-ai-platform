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
}

interface ChatMessageProps {
  turn: ChatTurn;
}

function ChatMessage({ turn }: ChatMessageProps) {
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
          </div>
        </div>
      ) : (
        <div className="chat-message__bubble">{turn.content}</div>
      )}
      {turn.role === 'assistant' && turn.sources.length > 0 && (
        <div className="chat-message__sources">
          {turn.sources.map((source) => (
            <CitationCard
              key={`${turn.id}-${source.source_number}`}
              sourceNumber={source.source_number}
              filename={source.filename}
              page={source.page}
              section={source.section}
              chunkId={source.chunk_id}
              text={source.text}
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
