import { useEffect, useRef } from 'react';
import type { CitationResponse } from '../api/types';
import ChatMessage, { type ChatTurn } from './ChatMessage';
import StreamingSkeleton from './StreamingSkeleton';
import './ChatThread.css';

interface ChatThreadProps {
  turns: ChatTurn[];
  pending: boolean;
  engineerMode: boolean;
  currentModelLabel?: string;
  onRetry?: (question: string) => void;
  onOpenSource?: (filename: string, chunkId: string) => void;
  onCitationHover?: (citation: CitationResponse) => void;
  onCitationLeave?: () => void;
}

function ChatThread({
  turns,
  pending,
  engineerMode,
  currentModelLabel,
  onRetry,
  onOpenSource,
  onCitationHover,
  onCitationLeave,
}: ChatThreadProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [turns.length, pending]);

  if (turns.length === 0 && !pending) {
    return (
      <div className="chat-thread chat-thread--empty">
        <p>Upload a document, then ask a question about it.</p>
      </div>
    );
  }

  const lastTurn = turns[turns.length - 1];
  // The assistant turn is created lazily on the first `sources`/`delta` event (see
  // useChat), so while pending is true but the last turn is still the user's question,
  // there's no ChatTurn yet to attach a skeleton to — render a placeholder in its slot.
  const awaitingFirstEvent = pending && lastTurn?.role !== 'assistant';

  return (
    <div className="chat-thread">
      <div className="chat-thread__rail">
        {turns.map((turn) => (
          <ChatMessage
            key={turn.id}
            turn={turn}
            engineerMode={engineerMode}
            currentModelLabel={currentModelLabel}
            streamStage={pending && turn.id === lastTurn?.id && turn.role === 'assistant' ? 'generating answer…' : undefined}
            onRetry={onRetry}
            onOpenSource={onOpenSource}
            onCitationHover={onCitationHover}
            onCitationLeave={onCitationLeave}
          />
        ))}
        {awaitingFirstEvent && (
          <div className="chat-message" data-testid="chat-pending">
            <div className="chat-message__row">
              <span className="chat-message__gutter chat-message__gutter--answer" aria-hidden="true">
                A
              </span>
              <div className="chat-message__answer">
                <StreamingSkeleton stage="retrieving…" />
              </div>
            </div>
          </div>
        )}
      </div>
      <div ref={bottomRef} />
    </div>
  );
}

export default ChatThread;
