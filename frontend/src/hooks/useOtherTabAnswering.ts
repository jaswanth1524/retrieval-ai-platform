import { useEffect, useState } from 'react';
import { announceAnswering, onOtherTabAnswering } from '../utils/chatChannel';

// While answering, a tab says so this often; another tab's word older than the grace
// below no longer counts (that tab was closed, or its answer ended unannounced).
export const ANSWERING_HEARTBEAT_MS = 5000;
const ANSWERING_GRACE_MS = 12_000;

/** Whether another tab is answering in ``conversationId`` now — and this tab's own
 *  answering told to the others. Two tabs on one conversation each asked and each
 *  wrote an answer onto it; the second now sees the first is on it. */
export function useOtherTabAnswering(
  conversationId: string | null,
  answeringHere: string | null,
): boolean {
  const [others, setOthers] = useState<ReadonlyMap<string, { conversationId: string; at: number }>>(
    () => new Map(),
  );

  useEffect(
    () =>
      onOtherTabAnswering((tab, answering) =>
        setOthers((prev) => {
          const next = new Map(prev);
          if (answering === null) next.delete(tab);
          else next.set(tab, { conversationId: answering, at: Date.now() });
          return next;
        }),
      ),
    [],
  );

  // Old words expire.
  useEffect(() => {
    if (others.size === 0) return;
    const timer = window.setInterval(() => {
      const now = Date.now();
      setOthers((prev) => {
        const kept = [...prev].filter(([, word]) => now - word.at < ANSWERING_GRACE_MS);
        return kept.length === prev.size ? prev : new Map(kept);
      });
    }, ANSWERING_HEARTBEAT_MS);
    return () => window.clearInterval(timer);
  }, [others.size]);

  useEffect(() => {
    if (answeringHere === null) return;
    announceAnswering(answeringHere);
    const timer = window.setInterval(() => announceAnswering(answeringHere), ANSWERING_HEARTBEAT_MS);
    return () => {
      window.clearInterval(timer);
      announceAnswering(null);
    };
  }, [answeringHere]);

  return conversationId !== null && [...others.values()].some((word) => word.conversationId === conversationId);
}
