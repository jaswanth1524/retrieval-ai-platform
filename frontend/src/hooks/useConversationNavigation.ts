import { useCallback, useEffect, useRef } from 'react';
import type { ChatTurn } from '../components/ChatMessage';
import type { RailPanel } from '../components/IconRail';
import type { ConversationSummary } from './chat/model';
import { useStableHandler } from './useStableHandler';

interface ConversationNavigationOptions {
  rail: RailPanel;
  conversations: ConversationSummary[];
  activeConversationId: string | null;
  switchConversation: (id: string) => void;
  turns: ChatTurn[];
  handleOpenSource: (filename: string, chunkId: string, highlights?: string[]) => void;
}

/**
 * Keyboard movement through the chat: `]`/`[` between conversations, `j`/`k` between
 * answers, and 1-9 to open a source of the answer in focus.
 */
export function useConversationNavigation({
  rail,
  conversations,
  activeConversationId,
  switchConversation,
  turns,
  handleOpenSource,
}: ConversationNavigationOptions) {
  // The conversation list's order (pinned first, its filters and search applied) while
  // it is shown: ] and [ walk what the reader sees. Else pinned first, by recent use.
  const listedOrderRef = useRef<string[] | null>(null);
  const noteListedOrder = useCallback((ids: string[]) => {
    listedOrderRef.current = ids;
  }, []);
  useEffect(() => {
    if (rail !== 'chat') listedOrderRef.current = null;
  }, [rail]);
  const cycleConversation = useStableHandler((step: 1 | -1) => {
    const order =
      listedOrderRef.current ??
      [
        ...conversations.filter((conversation) => conversation.pinned),
        ...conversations.filter((conversation) => !conversation.pinned),
      ].map((conversation) => conversation.id);
    if (order.length < 2) return;
    const index = order.indexOf(activeConversationId ?? '');
    // Not among those listed (filtered out): the first (or last) one.
    const nextIndex = index < 0 ? (step === 1 ? 0 : order.length - 1) : (index + step + order.length) % order.length;
    const next = order[nextIndex];
    if (next && next !== activeConversationId) switchConversation(next);
  });
  // 1-9: that source of the answer that has focus (j/k move it), else of the latest.
  const openAnswerSource = useStableHandler((number: number) => {
    const answers = turns.filter((turn) => turn.role === 'assistant' && turn.content);
    const focusedId = (document.activeElement as HTMLElement | null)?.closest<HTMLElement>(
      '[role="article"][data-role="assistant"]',
    )?.dataset.turnId;
    const answer = answers.find((turn) => turn.id === focusedId) ?? answers[answers.length - 1];
    const source = answer?.sources.find((candidate) => candidate.source_number === number);
    if (!source) return;
    handleOpenSource(
      source.filename,
      source.chunk_id,
      source.spans?.length ? source.spans.map((span) => span.quote) : undefined,
    );
  });
  const moveBetweenAnswers = useStableHandler((step: 1 | -1) => {
    const answers = [
      ...document.querySelectorAll<HTMLElement>('[role="article"][data-role="assistant"]'),
    ];
    if (answers.length === 0) return;
    const current = answers.findIndex((answer) => answer.contains(document.activeElement));
    let target: HTMLElement | undefined;
    if (current >= 0) {
      target = answers[current + step];
    } else {
      // Nothing focused yet: the first answer in view (j) or the last (k).
      const inView = answers.filter((answer) => {
        const box = answer.getBoundingClientRect();
        return box.bottom > 0 && box.top < window.innerHeight;
      });
      const pool = inView.length > 0 ? inView : answers;
      target = step === 1 ? pool[0] : pool[pool.length - 1];
    }
    if (!target) return;
    target.focus({ preventScroll: true });
    // The CSS reduced-motion rule doesn't reach a scripted smooth scroll.
    const still = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
    target.scrollIntoView({ block: 'start', behavior: still ? 'auto' : 'smooth' });
  });

  return { noteListedOrder, cycleConversation, openAnswerSource, moveBetweenAnswers };
}
