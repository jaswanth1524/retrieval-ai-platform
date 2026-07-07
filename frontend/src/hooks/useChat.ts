import { useEffect, useRef, useState } from 'react';
import { ApiClientError, api } from '../api/client';
import type { LlmProvider } from '../api/types';
import type { ChatTurn } from '../components/ChatMessage';

export interface UseChatResult {
  turns: ChatTurn[];
  pending: boolean;
  ask: (question: string, provider?: LlmProvider) => Promise<void>;
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
  // Tracks the in-flight question's controller: doubles as a reentrancy guard (a
  // second `ask` while one is running would fire a second ~70s LLM call whose
  // response could land in any order) and lets unmount cancel a hung request.
  const inflightRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      inflightRef.current?.abort();
    };
  }, []);

  const ask = async (question: string, provider?: LlmProvider) => {
    if (inflightRef.current) return;
    const controller = new AbortController();
    inflightRef.current = controller;

    setTurns((prev) => [...prev, makeTurn('user', question)]);
    setPending(true);
    try {
      const result = await api.askQuestion(question, provider, controller.signal);
      setTurns((prev) => [...prev, makeTurn('assistant', result.answer, result.sources)]);
    } catch (err) {
      const message = err instanceof ApiClientError ? err.message : 'Something went wrong.';
      setTurns((prev) => [...prev, makeTurn('error', message)]);
    } finally {
      inflightRef.current = null;
      setPending(false);
    }
  };

  return { turns, pending, ask };
}
