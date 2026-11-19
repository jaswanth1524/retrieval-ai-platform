import { useCallback, useMemo, useRef, useState } from 'react';
import type { RailPanel } from '../components/IconRail';
import type { SourceChange } from '../utils/citationFreshness';
import { unreviewedCount } from '../utils/review';
import type { ConversationSummary } from './chat/model';
import type { SavedAnswer } from './useChat';

interface ReviewQueueOptions {
  rail: RailPanel;
  workspace: string;
  pending: boolean;
  conversations: ConversationSummary[];
  answersIn: (workspace: string) => SavedAnswer[];
  setTurnReviewed: (conversationId: string, turnId: string, reviewed: boolean) => void;
  sourceChangeOf: (filename: string, answeredAtMs: number) => SourceChange | null;
}

/** The "Needs review" panel's answers and the rail's count of those still to review. */
export function useReviewQueue({
  rail,
  workspace,
  pending,
  conversations,
  answersIn,
  setTurnReviewed,
  sourceChangeOf,
}: ReviewQueueOptions) {
  // The "Needs review" panel: this workspace's answers, walked only while it is open
  // (conversations change on every streamed frame).
  // Marking an answer reviewed changes a turn, not the conversation summaries: counted
  // here so the list (and the rail's count) are read again.
  const [reviewTick, setReviewTick] = useState(0);
  const markReviewed = useCallback(
    (conversationId: string, turnId: string, reviewed: boolean) => {
      setTurnReviewed(conversationId, turnId, reviewed);
      setReviewTick((tick) => tick + 1);
    },
    [setTurnReviewed],
  );
  const reviewAnswers = useMemo(
    () => (rail === 'review' ? answersIn(workspace) : []),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [rail, workspace, answersIn, conversations, reviewTick],
  );
  // The rail's count of answers still to review: read between answers (walking every
  // turn on each streamed frame would cost the stream), kept while one streams.
  const reviewCountRef = useRef(0);
  const reviewCount = useMemo(() => {
    if (!pending) {
      reviewCountRef.current = unreviewedCount(
        answersIn(workspace).map((item) => item.turn),
        sourceChangeOf,
      );
    }
    return reviewCountRef.current;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pending, workspace, answersIn, conversations, sourceChangeOf, reviewTick]);

  return { markReviewed, reviewAnswers, reviewCount };
}
