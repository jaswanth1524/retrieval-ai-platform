import { useCallback, useMemo, useRef, type RefObject } from 'react';
import type { PassageNarrowing } from '../../api/types';
import type { ChatTurn } from '../../components/ChatMessage';
import type { ChatStorageV2, Conversation } from '../chatPersistence';

/** A saved answer and where it is (the Saved answers panel). */
export interface SavedAnswer {
  conversationId: string;
  conversationTitle: string;
  /** The workspace its conversation asked in ("" = the default). */
  workspace: string;
  turn: ChatTurn;
  /** The question it answered (the user turn before it), if there is one. */
  question: string | null;
  /** How that question was narrowed, and whether it was a whole-document task (a
   *  summary's "question" is a label, not something to search for). */
  narrowing?: PassageNarrowing;
  task?: boolean;
}

interface TurnCounts {
  turnCount: number;
  savedCount?: number;
  firstSavedTurnId?: string;
}

// Per turns array: only the conversation being answered gets a new one per streamed
// frame, so every other conversation's counts are reused instead of recounted.
const turnCountsCache = new WeakMap<readonly ChatTurn[], TurnCounts>();

export function turnCounts(turns: readonly ChatTurn[]): TurnCounts {
  const cached = turnCountsCache.get(turns);
  if (cached) return cached;
  let turnCount = 0;
  let savedCount = 0;
  let firstSavedTurnId: string | undefined;
  for (const turn of turns) {
    if (turn.role === 'user') turnCount += 1;
    if (turn.saved) {
      savedCount += 1;
      firstSavedTurnId ??= turn.id;
    }
  }
  const counts: TurnCounts =
    savedCount > 0 ? { turnCount, savedCount, firstSavedTurnId } : { turnCount };
  turnCountsCache.set(turns, counts);
  return counts;
}

/**
 * The answers other panels list: saved ones (kept up to date), and a workspace's or a
 * document's (read when asked). Part of ``useChat``.
 */
export function useAnswerQueries(
  conversations: Conversation[],
  storageRef: RefObject<ChatStorageV2>,
) {
  const savedAnswersRef = useRef<SavedAnswer[]>([]);
  const savedAnswers = useMemo(() => {
    const found: SavedAnswer[] = [];
    for (const conversation of conversations) {
      // The cached counts skip every conversation with nothing saved, unwalked.
      if (!turnCounts(conversation.turns).savedCount) continue;
      conversation.turns.forEach((turn, index) => {
        if (!turn.saved) return;
        const asked = conversation.turns
          .slice(0, index)
          .reverse()
          .find((earlier) => earlier.role === 'user');
        found.push({
          conversationId: conversation.id,
          conversationTitle: conversation.title,
          workspace: conversation.workspace ?? '',
          turn,
          question: asked?.content ?? null,
          ...(asked?.narrowing ? { narrowing: asked.narrowing } : {}),
          ...(asked?.task ? { task: true } : {}),
        });
      });
    }
    found.sort((a, b) => b.turn.timestamp - a.turn.timestamp);
    const previous = savedAnswersRef.current;
    // The same list (a streamed frame changed something else): the same array.
    if (
      previous.length === found.length &&
      previous.every(
        (item, index) =>
          item.turn === found[index].turn &&
          item.conversationTitle === found[index].conversationTitle,
      )
    ) {
      return previous;
    }
    savedAnswersRef.current = found;
    return found;
  }, [conversations]);

  // Every answer of a workspace's conversations, newest first, with its question: what
  // the "Needs review" panel looks through (read when it is open).
  const answersIn = useCallback((workspace: string) => {
    const found: SavedAnswer[] = [];
    for (const conversation of storageRef.current.conversations) {
      if ((conversation.workspace ?? '') !== workspace) continue;
      let asked: ChatTurn | null = null;
      for (const turn of conversation.turns) {
        if (turn.role === 'user') asked = turn;
        else if (turn.role === 'assistant') {
          found.push({
            conversationId: conversation.id,
            conversationTitle: conversation.title,
            workspace,
            turn,
            question: asked?.content ?? null,
            // As the saved list: "Ask again" re-asks a summary as a summary, and a
            // narrowed question with its narrowing (both were dropped here).
            ...(asked?.narrowing ? { narrowing: asked.narrowing } : {}),
            ...(asked?.task ? { task: true } : {}),
          });
        }
      }
    }
    return found.sort((a, b) => b.turn.timestamp - a.turn.timestamp);
  }, [storageRef]);

  // Read when asked (a card's "Questions that cited this"), not kept up to date: it walks
  // every turn of every conversation.
  const answersCiting = useCallback((filename: string, workspace: string) => {
    const found: SavedAnswer[] = [];
    for (const conversation of storageRef.current.conversations) {
      // Another workspace's document of the same name is a different document.
      if ((conversation.workspace ?? '') !== workspace) continue;
      conversation.turns.forEach((turn, index) => {
        if (turn.role !== 'assistant' || !turn.sources.some((source) => source.filename === filename)) {
          return;
        }
        const asked = conversation.turns
          .slice(0, index)
          .reverse()
          .find((earlier) => earlier.role === 'user');
        found.push({
          conversationId: conversation.id,
          conversationTitle: conversation.title,
          workspace: conversation.workspace ?? '',
          turn,
          question: asked?.content ?? null,
        });
      });
    }
    return found.sort((a, b) => b.turn.timestamp - a.turn.timestamp);
  }, [storageRef]);

  return { savedAnswers, answersIn, answersCiting };
}
