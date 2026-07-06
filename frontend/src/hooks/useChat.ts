import { useState } from 'react';
import { ApiClientError, api } from '../api/client';
import type { ChatTurn } from '../components/ChatMessage';

export interface UseChatResult {
  turns: ChatTurn[];
  pending: boolean;
  ask: (question: string) => Promise<void>;
}

function makeTurn(role: ChatTurn['role'], content: string, sources: ChatTurn['sources'] = []): ChatTurn {
  return {
    id: crypto.randomUUID(),
    role,
    content,
    sources,
    timestamp: Date.now(),
  };
}

export function useChat(): UseChatResult {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [pending, setPending] = useState(false);

  const ask = async (question: string) => {
    setTurns((prev) => [...prev, makeTurn('user', question)]);
    setPending(true);
    try {
      const result = await api.askQuestion(question);
      setTurns((prev) => [...prev, makeTurn('assistant', result.answer, result.sources)]);
    } catch (err) {
      const message = err instanceof ApiClientError ? err.message : 'Something went wrong.';
      setTurns((prev) => [...prev, makeTurn('error', message)]);
    } finally {
      setPending(false);
    }
  };

  return { turns, pending, ask };
}
