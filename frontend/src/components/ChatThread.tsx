import { useEffect, useRef } from 'react';
import ChatMessage, { type ChatTurn } from './ChatMessage';
import './ChatThread.css';

interface ChatThreadProps {
  turns: ChatTurn[];
  pending: boolean;
}

function ChatThread({ turns, pending }: ChatThreadProps) {
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

  return (
    <div className="chat-thread">
      {turns.map((turn) => (
        <ChatMessage key={turn.id} turn={turn} />
      ))}
      {pending && (
        <div className="chat-thread__pending" data-testid="chat-pending">
          Generating answer...
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}

export default ChatThread;
