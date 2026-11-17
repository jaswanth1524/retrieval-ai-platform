import { useCallback, type Dispatch, type SetStateAction } from 'react';
import type { FeedbackRating } from '../../api/types';
import { withVersionRestored } from '../../utils/answerVersions';
import { updateTurn } from '../chatPersistence';
import type { ChatStorageV2 } from '../chatPersistence';
import { MAX_NOTE_CHARS } from './sanitize';

/**
 * The marks a reader puts on one turn — saved, a note, reviewed, feedback, a source
 * marked not relevant, an earlier version restored — each a single stored update.
 * Part of ``useChat``.
 */
export function useTurnMarks(setStorage: Dispatch<SetStateAction<ChatStorageV2>>) {
  const setSourceIrrelevant = useCallback(
    (turnId: string, sourceNumber: number, marked: boolean, conversationId?: string) => {
      setStorage((prev) =>
        updateTurn(
          prev,
          conversationId ?? prev.activeConversationId ?? '',
          turnId,
          (turn) => {
            const others = (turn.irrelevantSources ?? []).filter((n) => n !== sourceNumber);
            const next = marked ? [...others, sourceNumber] : others;
            const { irrelevantSources: _previous, ...rest } = turn;
            return next.length > 0 ? { ...rest, irrelevantSources: next } : rest;
          },
          Date.now(),
        ),
      );
    },
    [setStorage],
  );

  const setTurnSaved = useCallback((turnId: string, saved: boolean, conversationId?: string) => {
    // One conversation's turn: an edited question's copy shares the earlier turns' ids
    // (follow-ups, feedback and the in-flight answer key on them), so matching the id
    // alone saved the answer in both and listed it twice.
    setStorage((prev) =>
      updateTurn(
        prev,
        conversationId ?? prev.activeConversationId ?? '',
        turnId,
        (turn) => {
          const { saved: _previous, ...rest } = turn;
          return saved ? { ...rest, saved: true } : rest;
        },
        Date.now(),
      ),
    );
  }, [setStorage]);

  // The reader's note on a saved answer ("" takes it away).
  const setTurnNote = useCallback((conversationId: string, turnId: string, note: string) => {
    setStorage((prev) =>
      updateTurn(
        prev,
        conversationId,
        turnId,
        (turn) => {
          const { note: _previous, ...rest } = turn;
          const text = note.trim().slice(0, MAX_NOTE_CHARS);
          return text ? { ...rest, note: text } : rest;
        },
        Date.now(),
      ),
    );
  }, [setStorage]);

  // Marked looked at in the Needs review panel (or unmarked).
  const setTurnReviewed = useCallback(
    (conversationId: string, turnId: string, reviewed: boolean) => {
      setStorage((prev) =>
        updateTurn(
          prev,
          conversationId,
          turnId,
          (turn) => {
            const { reviewedAt: _previous, ...rest } = turn;
            return reviewed ? { ...rest, reviewedAt: Date.now() } : rest;
          },
          Date.now(),
        ),
      );
    },
    [setStorage],
  );

  // An earlier version of an answer made the answer again (the one shown becomes a
  // version, so nothing is lost).
  const restoreAnswerVersion = useCallback((turnId: string, index: number) => {
    setStorage((prev) =>
      updateTurn(
        prev,
        prev.activeConversationId ?? '',
        turnId,
        (turn) => withVersionRestored(turn, index),
        Date.now(),
      ),
    );
  }, [setStorage]);

  const setTurnFeedback = useCallback(
    (turnId: string, rating: FeedbackRating | undefined, conversationId?: string) => {
      // Stored on the turn, so a rating survives a reload or a conversation switch and
      // the buttons can't send a duplicate row for the same answer.
      setStorage((prev) =>
        updateTurn(
          prev,
          conversationId ?? prev.activeConversationId ?? '',
          turnId,
          (turn) => ({ ...turn, feedback: rating }),
          Date.now(),
        ),
      );
    },
    [setStorage],
  );

  return {
    setSourceIrrelevant,
    setTurnSaved,
    setTurnNote,
    setTurnReviewed,
    restoreAnswerVersion,
    setTurnFeedback,
  };
}
