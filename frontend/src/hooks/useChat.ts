import { useEffect, useRef, useState } from 'react';
import { ApiClientError, api } from '../api/client';
import type { LlmProvider, QuestionOverrides } from '../api/types';
import type { ChatTurn } from '../components/ChatMessage';

export interface UseChatResult {
  turns: ChatTurn[];
  pending: boolean;
  ask: (
    question: string,
    provider?: LlmProvider,
    overrides?: QuestionOverrides,
    filenames?: string[],
  ) => Promise<void>;
  cancel: () => void;
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

  const ask = async (
    question: string,
    provider?: LlmProvider,
    overrides?: QuestionOverrides,
    filenames?: string[],
  ) => {
    if (inflightRef.current) return;
    const controller = new AbortController();
    inflightRef.current = controller;

    setTurns((prev) => [...prev, makeTurn('user', question)]);
    setPending(true);

    // The assistant turn is created lazily on the first `sources` or `delta` event
    // (not up front) so a mid-stream failure before any event arrives still renders
    // as a clean error turn instead of leaving a stray empty assistant bubble.
    const assistantTurnId = crypto.randomUUID();
    // Whether the turn already exists is derived from `prev` itself (not a mutated
    // outer flag) — React 18 (StrictMode, concurrent features) may invoke a setState
    // updater more than once per commit, and a mutated closure variable would then
    // see "already created" on a replay whose append never actually committed,
    // silently dropping every subsequent update.
    const updateAssistantTurn = (update: (turn: ChatTurn) => ChatTurn) => {
      setTurns((prev) => {
        const existing = prev.find((turn) => turn.id === assistantTurnId);
        if (!existing) {
          return [...prev, update({ ...makeTurn('assistant', ''), id: assistantTurnId })];
        }
        return prev.map((turn) => (turn.id === assistantTurnId ? update(turn) : turn));
      });
    };

    try {
      await api.askQuestionStream(
        question,
        provider,
        overrides,
        filenames,
        {
          onSources: (sources) => {
            updateAssistantTurn((turn) => ({ ...turn, sources }));
          },
          onDelta: (text) => {
            updateAssistantTurn((turn) => ({ ...turn, content: turn.content + text }));
          },
          onDone: (answer, sources) => {
            updateAssistantTurn((turn) => ({ ...turn, content: answer, sources }));
          },
        },
        controller.signal,
      );
    } catch (err) {
      const message = err instanceof ApiClientError ? err.message : 'Something went wrong.';
      setTurns((prev) => [...prev, makeTurn('error', message)]);
    } finally {
      inflightRef.current = null;
      setPending(false);
    }
  };

  const cancel = () => {
    // The abort surfaces through askQuestionStream's fetch as an AbortError, which
    // the catch block above already turns into a clean error turn — no separate
    // cancellation path needed.
    inflightRef.current?.abort();
  };

  return { turns, pending, ask, cancel };
}
