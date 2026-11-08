import type { MutableRefObject } from 'react';
import type { QuestionStreamHandlers } from '../api/client';
import type { ChatTurn } from '../components/ChatMessage';
import { cleanFollowups } from './chatPersistence';

type DoneHandler = NonNullable<QuestionStreamHandlers['onDone']>;

/** Deltas coalesced into one update per animation frame. Each update re-renders the
 *  streaming message, whose markdown is re-parsed over the whole answer so far — per
 *  token that is quadratic in answer length on a fast provider. The text waiting is
 *  kept in `pending` (a ref: a pagehide snapshot reads it too). */
export function createDeltaBuffer(
  pending: MutableRefObject<string>,
  apply: (text: string) => void,
  schedule: (flush: () => void) => number = requestAnimationFrame,
  unschedule: (id: number) => void = cancelAnimationFrame,
  /** Apply the first text at once, not on the first scheduled flush: the wait between
   *  flushes shouldn't delay the first token shown. */
  firstAtOnce = false,
) {
  pending.current = '';
  let frameId: number | null = null;
  let started = !firstAtOnce;
  const flush = () => {
    if (frameId !== null) {
      unschedule(frameId);
      frameId = null;
    }
    if (!pending.current) return;
    const text = pending.current;
    pending.current = '';
    apply(text);
  };
  return {
    push(text: string) {
      pending.current += text;
      if (!started) {
        started = true;
        flush();
        return;
      }
      if (frameId === null) frameId = schedule(flush);
    },
    flush,
  };
}

export interface AnswerStreamActions {
  /** Show `label` as what the server is doing (`etaMs`: how long the model should
   *  take to read its prompt, when the server knows). */
  enterStage: (label: string, etaMs?: number | null) => void;
  /** Turn a server stage name (and a summary's step) into its label. */
  stageLabel: (stage: string, progress?: { step: number; steps: number }) => string;
  /** Change the answer's turn (created on first use); `touch` false: not a user edit. */
  updateAnswer: (update: (turn: ChatTurn) => ChatTurn, touch?: boolean) => void;
  /** Put suggested follow-ups on the answer, if it is still there. */
  setFollowups: (followups: string[]) => void;
  /** `done`: the answer's final state (the buffer is flushed first). */
  done: DoneHandler;
}

/** The stream handlers of one question: stages, sources, coalesced deltas, follow-ups
 *  and `done`, applied through `actions`. */
export function answerStreamHandlers(
  isTask: boolean,
  buffer: ReturnType<typeof createDeltaBuffer>,
  actions: AnswerStreamActions,
): QuestionStreamHandlers {
  const writing = isTask ? 'writing the summary…' : 'generating answer…';
  let streamedText = false;
  return {
    onStage: (serverStage, progress, reading) => {
      actions.enterStage(actions.stageLabel(serverStage, progress), reading?.etaMs ?? null);
    },
    onSources: (sources, traceId) => {
      // Sources land the moment retrieval finishes, which is when generation starts; a
      // newer server follows with "reading" (or "waiting").
      actions.enterStage(writing);
      actions.updateAnswer((turn) => ({ ...turn, sources, traceId }));
    },
    onDelta: (text) => {
      if (!streamedText) {
        // The model is writing: "reading the passages…" is over.
        streamedText = true;
        actions.enterStage(writing);
      }
      buffer.push(text);
    },
    onFollowups: (questions) => {
      const followups = cleanFollowups(questions);
      if (followups.length > 0) actions.setFollowups(followups);
    },
    onDone: (...args) => {
      buffer.flush();
      actions.done(...args);
    },
  };
}
