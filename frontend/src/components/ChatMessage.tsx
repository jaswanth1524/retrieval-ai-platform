import type { CitationResponse } from '../api/types';
import CitationCard from './CitationCard';
import './ChatMessage.css';

export interface ChatTurn {
  id: string;
  role: 'user' | 'assistant' | 'error';
  content: string;
  sources: CitationResponse[];
  timestamp: number;
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
      <div className="chat-message__bubble">{turn.content}</div>
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
    </div>
  );
}

export default ChatMessage;
