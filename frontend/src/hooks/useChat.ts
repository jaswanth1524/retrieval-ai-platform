import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ApiClientError, api, apiErrorMessage, getWorkspace } from '../api/client';
import type { QuestionStreamHandlers } from '../api/client';
import type {
  FeedbackRating,
  HistoryMessage,
  LlmProvider,
  QuestionOptions,
  QuestionOverrides,
} from '../api/types';
import type { ChatTurn } from '../components/ChatMessage';
import { newId } from '../utils/id';
import { useChatPersistence } from './useChatPersistence';
import { useConversationActions } from './useConversationActions';
import { answerStreamHandlers, createDeltaBuffer } from './streamHandlers';
import { mergeRegenerated, withVersionRestored } from '../utils/answerVersions';
import { sanitizeStyle } from './chat/sanitize';
import type { ImportItem } from './useConversationActions';

/** How often streamed answer text is applied (see createDeltaBuffer's use below). */
export const STREAM_FLUSH_MS = 50;
export type { ImportItem } from './useConversationActions';
import {
  EMPTY_SCOPE,
  TITLE_MAX_CHARS,
  applyDeleteHeldDuringStream,
  byRecentUse,
  capConversations,
  conversationWorkspace,
  coverageWorthShowing,
  deriveTitle,
  loadStorage,
  searchConversations as searchStoredConversations,
  updateTurn,
  withConversationRestored,
  withoutConversation,
} from './chatPersistence';
import type {
  ChatStorageV2,
  Conversation,
  ConversationMatch,
  ConversationScope,
  ConversationSummary,
} from './chatPersistence';

// Re-exported for the components and tests that import them from here.
export { mergeConversations } from './chatPersistence';
export type {
  Conversation,
  ConversationMatch,
  ConversationScope,
  ConversationSummary,
} from './chatPersistence';

// History sent with a question lives in utils/history; re-exported for the components
// and tests that import it from here.
import { buildHistory, historyForQuestion } from '../utils/history';
export { buildHistory, historyForQuestion };

// Server stage labels mapped to what the reader sees. Anything the server sends that
// isn't listed here falls back to the generic label, so a new backend stage can never
// surface a raw protocol string in the UI or break an older frontend.
const STAGE_LABELS: Record<string, string> = {
  condensing: 'reading the conversation…',
  searching: 'searching documents…',
  waiting_for_model: 'waiting for the model — another question is using it…',
  joining_identical_question: 'the same question is being answered — waiting for it…',
  reading: 'reading the passages…',
  verifying_citations: 'checking citations…',
  reading_documents: 'reading the document…',
  summarizing: 'taking notes…',
  combining: 'combining notes…',
};
const DEFAULT_STAGE_LABEL = 'retrieving…';
/** When the question in flight was asked and its current stage began (epoch ms), and how
 *  long the local model should take to read the prompt (the server's estimate). */
export interface StageTiming {
  askedAt: number;
  stageStartedAt: number;
  etaMs: number | null;
}

export function stageLabel(stage: string, progress?: { step: number; steps: number }): string {
  if (stage === 'summarizing' && progress && progress.steps > 1) {
    return `taking notes, part ${progress.step} of ${progress.steps}…`;
  }
  return STAGE_LABELS[stage] ?? DEFAULT_STAGE_LABEL;
}

export interface UseChatResult {
  turns: ChatTurn[];
  /** False until the full history has loaded from IndexedDB (true where there is none). */
  historyReady: boolean;
  /** Any answer in flight, in whichever conversation. */
  pending: boolean;
  /** The conversation the answer in flight is written to; null when none is. */
  streamingConversationId: string | null;
  /** The turns of the conversation asked in last (the active one's before any ask). */
  askedTurns: ChatTurn[];
  // Human-readable label for the stage the in-flight question is on, or null when
  // nothing is in flight. Driven by real server `stage` events rather than a hardcoded
  // string, so a 30s retrieval no longer looks identical to a frozen UI.
  stage: string | null;
  /** Timing of the question in flight (null when none is): drives the elapsed time. */
  stageTiming: StageTiming | null;
  persistError: boolean;
  persistPartial: boolean;
  conversations: ConversationSummary[];
  /** Every saved answer across the conversations, newest first. */
  savedAnswers: SavedAnswer[];
  /** Every answer in this browser's conversations citing ``filename``, newest first. */
  /** Answers citing `filename` in conversations asked in `workspace` ('' = default). */
  answersCiting: (filename: string, workspace: string) => SavedAnswer[];
  /** Every answer in ``workspace``'s conversations, newest first, with its question. */
  answersIn: (workspace: string) => SavedAnswer[];
  activeConversationId: string | null;
  // The active conversation's search scope.
  scope: ConversationScope;
  /** The workspace the active conversation's questions went to ('' = default), or null
   *  while it has none and so belongs to whichever workspace asks first. */
  conversationWorkspace: string | null;
  /** A scope, or an updater of the current one — updaters compose, so two setters
   *  called in the same tick (clear files, then clear tags) don't undo each other. */
  setScope: (scope: ConversationScope | ((prev: ConversationScope) => ConversationScope)) => void;
  /** Drop names that are no longer indexed, and tags no document carries any more
   *  (when ``indexedTags`` is given), from every conversation's scope. */
  pruneScopes: (indexedFilenames: string[], indexedTags?: string[]) => void;
  /** Drop just these filenames from every conversation's scope. Unlike pruneScopes it
   *  needs no snapshot of the whole corpus, which can be stale by the time it runs.
   *  `workspace`: the one they were deleted from (the one in use when absent). */
  removeFromScopes: (filenames: string[], workspace?: string) => void;
  ask: (
    question: string,
    provider?: LlmProvider,
    overrides?: QuestionOverrides,
    filenames?: string[],
    options?: QuestionOptions,
  ) => Promise<void>;
  /** Ask `question` in place of an earlier user turn, in a new conversation forked
   *  just before it — the original conversation stays as it was. */
  editAndResend: (
    turnId: string,
    question: string,
    provider?: LlmProvider,
    overrides?: QuestionOverrides,
    filenames?: string[],
    options?: QuestionOptions,
  ) => Promise<void>;
  /** Add a conversation from a JSON export; false when the file holds no turns. */
  importConversation: (raw: unknown, title?: string) => boolean;
  /** Add the conversations of an "Export all" file, with their own dates, into the
   *  room the history has left. Existing conversations are pushed out to make room only
   *  with ``makeRoom`` (the least recently used, never a pinned, the active or the
   *  answering one); ``removed`` counts those, ``skipped`` the valid ones that didn't fit. */
  importConversations: (
    items: readonly ImportItem[],
    options?: { makeRoom?: boolean },
  ) => { imported: number; skipped: number; removed: number };
  /** How many of ``items`` are valid conversations, and how many fit without removing
   *  any (asked before importing, to offer making room). */
  importPreview: (items: readonly ImportItem[]) => { valid: number; room: number };
  cancel: () => void;
  clear: () => void;
  newConversation: () => void;
  switchConversation: (id: string) => void;
  renameConversation: (id: string, title: string) => void;
  /** The deleted conversation (null when it couldn't be: an answer streams into it). */
  deleteConversation: (id: string) => Conversation | null;
  /** Put a just-deleted conversation back (the delete's Undo). */
  restoreConversation: (conversation: Conversation) => void;
  /** Pin or unpin; false (nothing changes) past MAX_PINNED_CONVERSATIONS pins. */
  setPinned: (id: string, pinned: boolean) => boolean;
  /** Replace a conversation's labels (cleaned; an empty list removes them). */
  setLabels: (id: string, labels: string[]) => void;
  /** Rename a label on every conversation (``to`` null: remove it everywhere). */
  renameLabel: (from: string, to: string | null) => void;
  /** Conversations whose question or answer text contains `query`, by id. Reads the
   *  latest history, so call it from an effect or a memo keyed on `conversations`. */
  searchConversations: (query: string) => Map<string, ConversationMatch>;
  /** `undefined` takes a rating back (it could not be sent). */
  setTurnFeedback: (turnId: string, rating: FeedbackRating | undefined) => void;
  /** Every conversation with turns, as one JSON document (Export all conversations);
   *  importConversations reads it back. */
  allConversationsJson: (label?: string) => string;
  /** Mark (or, after a failed send, unmark) one cited source of a turn not relevant. */
  setSourceIrrelevant: (turnId: string, sourceNumber: number, marked: boolean) => void;
  /** Save (or unsave) an answer of ``conversationId`` (the open one when omitted). */
  setTurnSaved: (turnId: string, saved: boolean, conversationId?: string) => void;
  /** Make an answer's earlier version (by index) the answer again. */
  restoreAnswerVersion: (turnId: string, index: number) => void;
  /** Mark an answer looked at in the Needs review panel (or not). */
  setTurnReviewed: (conversationId: string, turnId: string, reviewed: boolean) => void;
}

function makeTurn(role: ChatTurn['role'], content: string, sources: ChatTurn['sources'] = []): ChatTurn {
  return {
    id: newId(),
    role,
    content,
    sources,
    timestamp: Date.now(),
    timings: null,
    traceId: null,
  };
}

const NO_TURNS: ChatTurn[] = [];

/** A saved answer and where it is (the Saved answers panel). */
export interface SavedAnswer {
  conversationId: string;
  conversationTitle: string;
  /** The workspace its conversation asked in ("" = the default). */
  workspace: string;
  turn: ChatTurn;
  /** The question it answered (the user turn before it), if there is one. */
  question: string | null;
}

interface TurnCounts {
  turnCount: number;
  savedCount?: number;
  firstSavedTurnId?: string;
}

// Per turns array: only the conversation being answered gets a new one per streamed
// frame, so every other conversation's counts are reused instead of recounted.
const turnCountsCache = new WeakMap<readonly ChatTurn[], TurnCounts>();

function turnCounts(turns: readonly ChatTurn[]): TurnCounts {
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

/** Times a question refused with 429 is asked again on its own. */
export const BUSY_RETRY_ATTEMPTS = 3;
const BUSY_RETRY_MAX_SECONDS = 60;

/** How long to wait before asking again after ``err`` on ``attempt``, or null when it
 *  isn't a busy refusal (or the attempts are used up). */
export function busyRetrySeconds(err: unknown, attempt: number): number | null {
  if (!(err instanceof ApiClientError) || err.statusCode !== 429) return null;
  if (attempt > BUSY_RETRY_ATTEMPTS) return null;
  return Math.min(Math.max(1, Math.ceil(err.retryAfterSeconds ?? 5)), BUSY_RETRY_MAX_SECONDS);
}

/** Waits ``seconds``, calling ``onTick`` with the seconds left each second; ends early
 *  (resolving) when ``signal`` aborts. */
export function countDown(
  seconds: number,
  signal: AbortSignal,
  onTick: (left: number) => void,
): Promise<void> {
  return new Promise((resolve) => {
    let left = seconds;
    onTick(left);
    const timer = setInterval(() => {
      left -= 1;
      if (left <= 0) done();
      else onTick(left);
    }, 1000);
    const done = () => {
      clearInterval(timer);
      signal.removeEventListener('abort', done);
      resolve();
    };
    signal.addEventListener('abort', done);
  });
}

/** ``send()``, asked again after a busy refusal (429: every question slot taken, or the
 *  rate limit), up to BUSY_RETRY_ATTEMPTS times: ``onWait`` hears each second of the
 *  wait, ``onResume`` that it is asking again. ``signal`` aborting ends the wait and
 *  rethrows the refusal; any other error is rethrown at once. */
export async function withBusyRetry(
  send: () => Promise<void>,
  signal: AbortSignal,
  onWait: (left: number) => void,
  onResume: () => void,
): Promise<void> {
  for (let attempt = 1; ; attempt += 1) {
    try {
      await send();
      return;
    } catch (err) {
      const waitSeconds = busyRetrySeconds(err, attempt);
      if (waitSeconds === null || signal.aborted) throw err;
      await countDown(waitSeconds, signal, onWait);
      if (signal.aborted) throw err;
      onResume();
    }
  }
}

type DoneArgs = Parameters<NonNullable<QuestionStreamHandlers['onDone']>>;

/** The answer ``turn`` becomes at the stream's `done` frame. */
export function answeredTurn(
  turn: ChatTurn,
  answer: DoneArgs[0],
  sources: DoneArgs[1],
  timings: DoneArgs[2],
  traceId: DoneArgs[3],
  cached: DoneArgs[4],
  grounding: DoneArgs[5],
): ChatTurn {
  const status = grounding?.answerStatus;
  const notFound = status === 'insufficient_context' || status === 'no_context';
  return {
    ...turn,
    content: answer,
    sources,
    // The passages it was given, kept when it found nothing in them, so the reader can
    // see what was looked at (text dropped: CitationCard opens the passage by chunk id).
    ...(notFound && turn.sources.length > 0 && sources.length === 0
      ? {
          contextSources: turn.sources.map(({ spans: _spans, ...source }) => ({
            ...source,
            text: '',
          })),
        }
      : {}),
    ...(grounding?.autoCited ? { autoCited: true } : {}),
    ...(grounding?.passagesLeftOut ? { passagesLeftOut: grounding.passagesLeftOut } : {}),
    timings,
    traceId,
    ...(cached ? { cached: true } : {}),
    ...(notFound ? { notFound: true } : {}),
    ...(status === 'uncited' ? { uncited: true } : {}),
    ...(grounding?.weakMatches ? { weakMatch: true } : {}),
    ...(grounding?.truncated ? { truncated: true } : {}),
    ...(grounding?.coverage && coverageWorthShowing(grounding.coverage)
      ? { coverage: grounding.coverage }
      : {}),
    ...(typeof grounding?.topRerankScore === 'number'
      ? { topRerankScore: grounding.topRerankScore }
      : {}),
    ...(typeof grounding?.rerankMinScore === 'number'
      ? { rerankMinScore: grounding.rerankMinScore }
      : {}),
    ...(grounding?.usage
      ? {
          usage: {
            prompt: grounding.usage.prompt_tokens,
            completion: grounding.usage.completion_tokens,
            ...(typeof grounding.usage.cost_usd === 'number'
              ? { costUsd: grounding.usage.cost_usd }
              : {}),
          },
        }
      : {}),
  };
}

interface TrailingStream {
  conversationId: string | null;
  controller: AbortController;
}


export function useChat(): UseChatResult {
  const [storage, setStorage] = useState<ChatStorageV2>(loadStorage);
  const [pending, setPending] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [stageTiming, setStageTiming] = useState<StageTiming | null>(null);
  // Which conversation the answer in flight is written to (null when none is), and the
  // one asked in last: other conversations can be opened while an answer streams.
  const [streamingConversationId, setStreamingConversationId] = useState<string | null>(null);
  const [askedConversationId, setAskedConversationId] = useState<string | null>(null);

  const { conversations, activeConversationId } = storage;
  const activeConversation =
    conversations.find((conversation) => conversation.id === activeConversationId) ??
    conversations[0];
  const turns = activeConversation?.turns ?? [];
  // Kept in sync every render (not inside an effect — it must be current before ask's
  // first read, not one render behind) so `ask` below can be a stable useCallback
  // identity: reading through the ref instead of closing over `activeConversation`
  // directly means `ask`'s reference doesn't change on every delta, which matters
  // because it's passed down as ChatMessage's onRetry prop and an unstable onRetry
  // defeats React.memo(ChatMessage) — see App.tsx's askQuestion.
  const activeConversationRef = useRef(activeConversation);
  activeConversationRef.current = activeConversation;

  const storageRef = useRef(storage);
  storageRef.current = storage;
  // The conversation asked in last, for what the background notice says once it ends.
  const askedTurns = useMemo(
    () =>
      conversations.find((conversation) => conversation.id === askedConversationId)?.turns ??
      activeConversation?.turns ??
      NO_TURNS,
    [conversations, askedConversationId, activeConversation],
  );
  // The assistant turn being streamed right now, if any — see the pagehide flush.
  const streamingTurnIdRef = useRef<string | null>(null);
  // The conversation it streams into, and the text not yet applied to it (deltas are
  // applied once per animation frame, and a hidden tab runs no frames).
  const streamingConversationIdRef = useRef<string | null>(null);
  const bufferedDeltaRef = useRef('');
  const { persistError, persistPartial, historyReady, historyLoaded } = useChatPersistence(storage, setStorage, pending, {
    storageRef,
    streamingTurnIdRef,
    streamingConversationIdRef,
    bufferedDeltaRef,
  });
  const {
    importConversation,
    importPreview,
    importConversations,
    newConversation,
    switchConversation,
    renameConversation,
    setPinned,
    setLabels,
    renameLabel,
  } = useConversationActions(setStorage, storageRef, streamingConversationIdRef);

  // Tracks the in-flight question's controller: doubles as a reentrancy guard and lets
  // unmount cancel a hung request.
  const inflightRef = useRef<AbortController | null>(null);
  // Answered streams still open for their suggested follow-ups (see runAsk's finish).
  const trailingRef = useRef(new Set<TrailingStream>());

  useEffect(() => {
    const trailing = trailingRef.current;
    return () => {
      inflightRef.current?.abort();
      trailing.forEach((stream) => stream.controller.abort());
    };
  }, []);

  const pendingRef = useRef(pending);
  pendingRef.current = pending;

  // Apply a turns-updater to one conversation (null: the active one). `touch` bumps its
  // updatedAt; a streamed delta doesn't, so the conversation list (sorted by it) doesn't
  // re-render on every frame of an answer.
  const setConversationTurns = useCallback((
    conversationId: string | null,
    update: (turns: ChatTurn[]) => ChatTurn[],
    touch = true,
  ) => {
    setStorage((prev) => {
      const targetId =
        conversationId ?? prev.activeConversationId ?? prev.conversations[0]?.id ?? null;
      return {
        ...prev,
        conversations: prev.conversations.map((conversation) => {
          if (conversation.id !== targetId) return conversation;
          // Call the updater exactly once. It used to run a second time for the title,
          // doubling the work on the hottest path in the app (every streamed delta
          // rebuilds the turn list) and quietly requiring every updater to be pure.
          const nextTurns = update(conversation.turns);
          return {
            ...conversation,
            turns: nextTurns,
            title:
              conversation.title === 'New chat' ? deriveTitle(nextTurns) : conversation.title,
            updatedAt: touch ? Date.now() : conversation.updatedAt,
          };
        }),
      };
    });
  }, []);

  const runAsk = useCallback(async (
    // The conversation asked in. Every update of this answer goes to it by id: written
    // to "the active conversation", a merge from another tab that moved the active id
    // sent the rest of the stream into a different conversation.
    conversationId: string | null,
    question: string,
    history: HistoryMessage[],
    provider?: LlmProvider,
    overrides?: QuestionOverrides,
    filenames?: string[],
    options?: QuestionOptions,
  ) => {
    if (inflightRef.current) return;
    const controller = new AbortController();
    inflightRef.current = controller;
    streamingConversationIdRef.current = conversationId;
    setStreamingConversationId(conversationId);
    setAskedConversationId(conversationId);
    const startedAt = Date.now();
    const updateTurns = (update: (turns: ChatTurn[]) => ChatTurn[], touch = true) =>
      setConversationTurns(conversationId, update, touch);
    let trailing: TrailingStream | null = null;

    const task = options?.task;
    // Kept on the question's turns so Retry and Regenerate look in the same passages.
    const narrowing = options?.narrowing && Object.keys(options.narrowing).length > 0 ? options.narrowing : undefined;
    // The first question binds the conversation to the workspace it was asked in
    // (api/client sends that one's header); the default is recorded as no field.
    const workspace = getWorkspace();
    setStorage((prev) => ({
      ...prev,
      conversations: prev.conversations.map((conversation) => {
        if (conversation.id !== conversationId || conversation.turns.length > 0) return conversation;
        const { workspace: _previous, ...rest } = conversation;
        return workspace ? { ...rest, workspace } : rest;
      }),
    }));
    updateTurns((prev) => {
      // Asked again after it got no answer (stopped before the first frame, a reload
      // mid-retrieval): it takes the unanswered question's place, not a second copy.
      const last = prev[prev.length - 1];
      const unanswered =
        last?.role === 'user' &&
        last.content === question &&
        JSON.stringify(last.task ?? null) === JSON.stringify(task ?? null);
      return [
        ...(unanswered ? prev.slice(0, -1) : prev),
        { ...makeTurn('user', question), ...(task ? { task } : {}), ...(narrowing ? { narrowing } : {}) },
      ];
    });
    setPending(true);
    // Seeded before the first server frame arrives so there is never a gap with a
    // pending request and no label at all.
    setStage(DEFAULT_STAGE_LABEL);
    const askedAt = Date.now();
    setStageTiming({ askedAt, stageStartedAt: askedAt, etaMs: null });
    const enterStage = (label: string, etaMs: number | null = null) => {
      setStage(label);
      setStageTiming({ askedAt, stageStartedAt: Date.now(), etaMs });
    };
    // Searched within some documents or tags: a "not found" is about those, not all.
    const scoped = (filenames?.length ?? 0) > 0 || (options?.tags?.length ?? 0) > 0;

    // The assistant turn is created lazily on the first `sources`/`delta` event so a
    // mid-stream failure before any event arrives renders as a clean error turn.
    const assistantTurnId = newId();
    streamingTurnIdRef.current = assistantTurnId;
    const updateAssistantTurn = (update: (turn: ChatTurn) => ChatTurn, touch = true) => {
      updateTurns((prev) => {
        const existing = prev.find((turn) => turn.id === assistantTurnId);
        if (!existing) {
          const created: ChatTurn = { ...makeTurn('assistant', ''), id: assistantTurnId };
          if (options?.modelLabel) created.model = options.modelLabel;
          const style = sanitizeStyle({
            answerFormat: options?.answerFormat,
            answerLanguage: options?.answerLanguage,
          });
          if (style) created.style = style;
          // Asked with the conversation as history: rated as a follow-up (feedback kind).
          // A summary sends none.
          if (!task && history.length > 0) created.followUp = true;
          if (scoped && !task) created.scoped = true;
          return [...prev, update(created)];
        }
        return prev.map((turn) => (turn.id === assistantTurnId ? update(turn) : turn));
      }, touch);
    };

    // Streamed text is applied every STREAM_FLUSH_MS, not every animation frame.
    // Measured on the production build (400 deltas over ~5 s): main-thread scripting per
    // answer 197 → 67 ms and task time 436 → 186 ms, the view still following the end.
    // Each flush re-renders the answer, re-parses its last markdown block and lays the
    // thread out; 20 a second still reads as continuous text (a model sends ~30-80
    // tokens a second). The first text is shown at once.
    const deltas = createDeltaBuffer(
      bufferedDeltaRef,
      (text) => updateAssistantTurn((turn) => ({ ...turn, content: turn.content + text }), false),
      (flush) => window.setTimeout(flush, STREAM_FLUSH_MS),
      (id) => window.clearTimeout(id),
      true,
    );
    const flushDelta = deltas.flush;

    // The answer is finished at `done`, though the stream stays open for the suggested
    // follow-ups the server works out after it: until then the answer read as still
    // streaming (no Regenerate or rating, Stop on screen, no "Answer ready") and the next
    // question waited for suggestions nobody had asked for yet. Run once — at `done`, or
    // when the stream ends without one — so a question asked in between keeps its state.
    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      flushDelta();
      streamingTurnIdRef.current = null;
      streamingConversationIdRef.current = null;
      setStreamingConversationId(null);
      if (inflightRef.current === controller) inflightRef.current = null;
      if (conversationId) {
        setStorage((prev) => applyDeleteHeldDuringStream(prev, conversationId, startedAt));
      }
      setPending(false);
      setStage(null);
      setStageTiming(null);
    };
    // A newer question in this conversation makes these suggestions moot (only the last
    // answer shows its own): stop waiting for them, freeing the server's question slot.
    trailingRef.current.forEach((trailing) => {
      if (trailing.conversationId === conversationId) trailing.controller.abort();
    });

    try {
      // A 429 (every question slot busy, or the rate limit) arrives before any frame:
      // waited out with a countdown and asked again, up to BUSY_RETRY_ATTEMPTS times,
      // instead of an error turn telling the user to do exactly that by hand. Stop
      // cancels the wait.
      await withBusyRetry(
        () =>
          api.askQuestionStream(
            question,
            provider,
            overrides,
            filenames,
            history,
            answerStreamHandlers(Boolean(task), deltas, {
              enterStage,
              stageLabel,
              updateAnswer: updateAssistantTurn,
              // Onto this answer by id, and never creating it: by now it may have been
              // cleared or deleted, and other questions may have been asked since.
              setFollowups: (followups) =>
                setConversationTurns(
                  conversationId,
                  (prev) =>
                    prev.some((turn) => turn.id === assistantTurnId)
                      ? prev.map((turn) => (turn.id === assistantTurnId ? { ...turn, followups } : turn))
                      : prev,
                  false,
                ),
              done: (answer, sources, timings, traceId, cached, grounding) => {
                updateAssistantTurn((turn) =>
                  answeredTurn(turn, answer, sources, timings, traceId, cached, grounding),
                );
                // Regenerated: the answer it replaces becomes its earlier version (only
                // now — a regenerate that failed leaves the old answer where it was).
                const replaces = options?.replacesAnswer;
                if (replaces) {
                  updateTurns((prev) => mergeRegenerated(prev, replaces, assistantTurnId));
                }
                finish();
                // Still open for follow-ups; aborted on unmount or by a newer question here.
                trailing = { conversationId, controller };
                trailingRef.current.add(trailing);
              },
            }),
            controller.signal,
            // The workspace it was asked in: a retry after a 429 countdown went to
            // whichever one the reader had switched to meanwhile.
            { ...options, workspace },
          ),
        controller.signal,
        (left) => setStage(`server busy — asking again in ${left}s…`),
        () => setStage(DEFAULT_STAGE_LABEL),
      );
    } catch (err) {
      flushDelta();
      // A user-initiated Stop aborts `controller`; a timeout aborts askQuestionStream's
      // own private controller instead, leaving this one untouched. So the flag is true
      // only for a deliberate cancel — where whatever streamed so far should simply
      // stand, with no error row claiming the request failed.
      if (controller.signal.aborted) {
        // Drop a content-less assistant turn: cancelling after `sources` but before the
        // first delta would otherwise leave an empty answer bubble behind. A partial one
        // is kept but marked, so after a reload it doesn't read as a finished answer.
        updateTurns((prev) =>
          prev
            .filter((turn) => turn.id !== assistantTurnId || turn.content !== '')
            .map((turn) => (turn.id === assistantTurnId ? { ...turn, stopped: true } : turn)),
        );
      } else {
        const message = apiErrorMessage(err, 'Something went wrong.');
        // Whatever streamed before the failure is a fragment. Left unmarked it read as a
        // finished answer: it offered Regenerate and feedback, and went back to the
        // model as history on the next question.
        // (An assistant turn with no text yet is kept as it was: it carries the trace
        // link to the partial server-side trace.)
        updateTurns((prev) => [
          ...prev.map((turn) =>
            turn.id === assistantTurnId && turn.content !== '' ? { ...turn, incomplete: true } : turn,
          ),
          {
            ...makeTurn('error', message),
            question,
            ...(task ? { task } : {}),
            ...(narrowing ? { narrowing } : {}),
          },
        ]);
      }
    } finally {
      if (trailing) trailingRef.current.delete(trailing);
      finish();
    }
    // Only refs and stable setters are closed over — see activeConversationRef above.
  }, [setConversationTurns]);

  const ask = useCallback(
    (
      question: string,
      provider?: LlmProvider,
      overrides?: QuestionOverrides,
      filenames?: string[],
      options?: QuestionOptions,
    ) => {
      const turns = activeConversationRef.current?.turns ?? [];
      // Regenerating (or reformatting) an answer: with the conversation as it stood
      // when its question was asked — an earlier answer's later turns are no part of
      // it — and in the style it was asked for, unless another is asked for now.
      const replacing = options?.replacesAnswer
        ? turns.findIndex((turn) => turn.id === options.replacesAnswer)
        : -1;
      const replaced = replacing >= 0 ? turns[replacing] : undefined;
      const history =
        replacing >= 1 ? buildHistory(turns.slice(0, replacing - 1)) : historyForQuestion(turns, question);
      const restyled =
        replaced?.style && options && !options.answerFormat && !options.answerLanguage
          ? { ...options, ...replaced.style }
          : options;
      return runAsk(
        activeConversationRef.current?.id ?? null,
        question,
        history,
        provider,
        overrides,
        filenames,
        restyled,
      );
    },
    [runAsk],
  );

  const editAndResend = useCallback(
    async (
      turnId: string,
      question: string,
      provider?: LlmProvider,
      overrides?: QuestionOverrides,
      filenames?: string[],
      options?: QuestionOptions,
    ) => {
      if (inflightRef.current) return;
      const source = activeConversationRef.current;
      const index = source?.turns.findIndex((turn) => turn.id === turnId) ?? -1;
      if (!source || index < 0 || source.turns[index].role !== 'user') return;
      // Passages the question was asked about (or without) stay with it when it is
      // edited; otherwise the conversation's scope applies, as for any new question.
      const asked = source.turns[index].narrowing;
      // Picked passages are the whole scope, as when they were first asked about.
      const picked = Boolean(asked?.chunk_ids?.length);
      const editedOptions =
        asked?.chunk_ids || asked?.exclude_chunk_ids
          ? { ...options, narrowing: asked, ...(picked ? { tags: [] } : {}) }
          : options;
      // The copy's earlier answers aren't saved again: a saved one stays saved where it
      // was, rather than listed once per edit.
      const prior = source.turns.slice(0, index).map((turn) => {
        if (!turn.saved) return turn;
        const { saved: _saved, ...rest } = turn;
        return rest;
      });
      const now = Date.now();
      const fork: Conversation = {
        id: newId(),
        title: prior.some((turn) => turn.role === 'user')
          ? `${source.title} (edited)`.slice(0, TITLE_MAX_CHARS)
          : 'New chat',
        createdAt: now,
        updatedAt: now,
        turns: prior,
        scope: source.scope,
        ...(source.workspace ? { workspace: source.workspace } : {}),
        // Filed with the conversation it was edited from.
        ...(source.labels?.length ? { labels: [...source.labels] } : {}),
      };
      setStorage((prev) => ({
        ...prev,
        activeConversationId: fork.id,
        conversations: capConversations([fork, ...prev.conversations], fork.id),
      }));
      await runAsk(
        fork.id,
        question,
        buildHistory(prior),
        provider,
        overrides,
        picked ? undefined : filenames,
        editedOptions,
      );
    },
    [runAsk],
  );

  const setScope = useCallback(
    (next: ConversationScope | ((prev: ConversationScope) => ConversationScope)) => {
      setStorage((prev) => ({
        ...prev,
        conversations: prev.conversations.map((conversation) =>
          conversation.id === prev.activeConversationId
            ? {
                ...conversation,
                scope: typeof next === 'function' ? next(conversation.scope ?? EMPTY_SCOPE) : next,
                editedAt: Date.now(),
              }
            : conversation,
        ),
      }));
    },
    [],
  );

  const keepInScopes = useCallback((
    keep: (filename: string) => boolean,
    keepTag: (tag: string) => boolean = () => true,
    current = getWorkspace(),
  ) => {
    // Only the conversations of the workspace the change was seen in: another
    // workspace's documents are simply not in its listing.
    setStorage((prev) => {
      let changed = false;
      const conversations = prev.conversations.map((conversation) => {
        const bound = conversationWorkspace(conversation);
        // One with no questions yet belongs to the workspace in use, so a delete that
        // ends after a switch away from its workspace leaves it alone.
        if (bound === null ? current !== getWorkspace() : bound !== current) return conversation;
        const filenames = conversation.scope?.filenames ?? [];
        const tags = conversation.scope?.tags ?? [];
        const kept = filenames.filter(keep);
        const keptTags = tags.filter(keepTag);
        if (kept.length === filenames.length && keptTags.length === tags.length) return conversation;
        changed = true;
        // The page range and upload date stay; a page range only while its one
        // document does.
        const { pageMin, pageMax, uploadedAfter, uploadedBefore } =
          conversation.scope ?? EMPTY_SCOPE;
        const pagesStay = kept.length === filenames.length;
        return {
          ...conversation,
          scope: {
            tags: keptTags,
            filenames: kept,
            ...(pagesStay && pageMin !== undefined ? { pageMin } : {}),
            ...(pagesStay && pageMax !== undefined ? { pageMax } : {}),
            ...(uploadedAfter !== undefined ? { uploadedAfter } : {}),
            ...(uploadedBefore !== undefined ? { uploadedBefore } : {}),
          },
          editedAt: Date.now(),
        };
      });
      return changed ? { ...prev, conversations } : prev;
    });
  }, []);

  // A listing that lands before the stored history has loaded is held, not applied:
  // pruning then stamped edits on the compact localStorage copies, which then won the
  // merge with IndexedDB's full ones (older turns and passage text lost).
  const historyLoadedRef = useRef(historyLoaded);
  historyLoadedRef.current = historyLoaded;
  const heldPruneRef = useRef<[string[], string[] | undefined] | null>(null);

  const pruneScopes = useCallback(
    (indexedFilenames: string[], indexedTags?: string[]) => {
      if (!historyLoadedRef.current) {
        heldPruneRef.current = [indexedFilenames, indexedTags];
        return;
      }
      const indexed = new Set(indexedFilenames);
      // A tag no document carries filtered every question down to nothing, while the
      // header still named it and the popover (which lists tags in use) couldn't untick
      // it. Matched like the server matches question tags: ignoring case.
      const tags = indexedTags && new Set(indexedTags.map((tag) => tag.toLowerCase()));
      keepInScopes(
        (name) => indexed.has(name),
        tags ? (tag) => tags.has(tag.toLowerCase()) : undefined,
      );
    },
    [keepInScopes],
  );

  useEffect(() => {
    if (!historyLoaded || heldPruneRef.current === null) return;
    const [indexedFilenames, indexedTags] = heldPruneRef.current;
    heldPruneRef.current = null;
    pruneScopes(indexedFilenames, indexedTags);
  }, [historyLoaded, pruneScopes]);

  const removeFromScopes = useCallback(
    (filenames: string[], workspace?: string) => {
      const removed = new Set(filenames);
      keepInScopes((name) => !removed.has(name), undefined, workspace);
    },
    [keepInScopes],
  );

  // Stable identities (pending is read through a ref) so components receiving these
  // don't re-render on every streamed delta.
  const cancel = useCallback(() => {
    inflightRef.current?.abort();
  }, []);

  // While an answer streams, only the conversation it is written to is held: the others
  // can be opened, started, cleared and deleted (the answer goes to its own by id).
  const isStreamingInto = (id: string | undefined | null) =>
    pendingRef.current && id != null && id === streamingConversationIdRef.current;

  const clear = useCallback(() => {
    if (isStreamingInto(activeConversationRef.current?.id)) return;
    setStorage((prev) => ({
      ...prev,
      conversations: prev.conversations.map((conversation) =>
        conversation.id === prev.activeConversationId
          ? // The title came from the first question, which is gone now. An edit (not
            // just activity), so the empty version wins over other tabs' copies.
            { ...conversation, turns: [], title: 'New chat', updatedAt: Date.now(), editedAt: Date.now() }
          : conversation,
      ),
    }));
  }, []);

  // Returns what was deleted, so the caller can offer it back (restoreConversation).
  const deleteConversation = useCallback((id: string): Conversation | null => {
    if (isStreamingInto(id)) return null;
    const removed = storageRef.current.conversations.find((conversation) => conversation.id === id);
    setStorage((prev) => withoutConversation(prev, id, Date.now()));
    return removed ?? null;
  }, []);

  const restoreConversation = useCallback((conversation: Conversation) => {
    setStorage((prev) => withConversationRestored(prev, conversation));
  }, []);

  const searchConversations = useCallback(
    (query: string) => searchStoredConversations(storageRef.current.conversations, query),
    [],
  );

  const allConversationsJson = useCallback(
    (label?: string) =>
      JSON.stringify(
        {
          docrag_conversations: storageRef.current.conversations
            .filter((conversation) => conversation.turns.length > 0)
            // Only those filed under ``label``, when one is given.
            .filter(
              (conversation) =>
                label === undefined ||
                (conversation.labels ?? []).some(
                  (each) => each.toLocaleLowerCase() === label.toLocaleLowerCase(),
                ),
            )
            .sort(byRecentUse)
            // With its dates: an import keeps them (importConversations), and without
            // them every imported conversation read as used just now.
            .map((conversation) => ({
              title: conversation.title,
              createdAt: conversation.createdAt,
              updatedAt: conversation.updatedAt,
              ...(conversation.labels?.length ? { labels: conversation.labels } : {}),
              turns: conversation.turns,
            })),
        },
        null,
        2,
      ),
    [],
  );

  const setSourceIrrelevant = useCallback(
    (turnId: string, sourceNumber: number, marked: boolean) => {
      setStorage((prev) =>
        updateTurn(
          prev,
          undefined,
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
    [],
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
  }, []);

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
    [],
  );

  // An earlier version of an answer made the answer again (the one shown becomes a
  // version, so nothing is lost).
  const restoreAnswerVersion = useCallback((turnId: string, index: number) => {
    setStorage((prev) =>
      updateTurn(prev, undefined, turnId, (turn) => withVersionRestored(turn, index), Date.now()),
    );
  }, []);

  const setTurnFeedback = useCallback((turnId: string, rating: FeedbackRating | undefined) => {
    // Stored on the turn, so a rating survives a reload or a conversation switch and
    // the buttons can't send a duplicate row for the same answer.
    setStorage((prev) =>
      updateTurn(prev, undefined, turnId, (turn) => ({ ...turn, feedback: rating }), Date.now()),
    );
  }, []);

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
      let question: string | null = null;
      for (const turn of conversation.turns) {
        if (turn.role === 'user') question = turn.content;
        else if (turn.role === 'assistant') {
          found.push({
            conversationId: conversation.id,
            conversationTitle: conversation.title,
            workspace,
            turn,
            question,
          });
        }
      }
    }
    return found.sort((a, b) => b.turn.timestamp - a.turn.timestamp);
  }, []);

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
  }, []);

  // Keeps the previous array when nothing the list shows changed: `conversations` is a
  // new array on every streamed frame, and a new summaries array re-rendered the list.
  const summariesRef = useRef<ConversationSummary[]>([]);
  const summaries: ConversationSummary[] = useMemo(() => {
    const next = [...conversations].sort(byRecentUse).map((conversation) => ({
      id: conversation.id,
      title: conversation.title,
      updatedAt: conversation.updatedAt,
      ...(conversation.pinned ? { pinned: true } : {}),
      ...(conversation.labels?.length ? { labels: conversation.labels } : {}),
      ...(conversation.workspace && conversation.turns.length > 0
        ? { workspace: conversation.workspace }
        : {}),
      ...turnCounts(conversation.turns),
    }));
    const prev = summariesRef.current;
    const same =
      prev.length === next.length &&
      prev.every(
        (summary, index) =>
          summary.id === next[index].id &&
          summary.title === next[index].title &&
          summary.updatedAt === next[index].updatedAt &&
          summary.turnCount === next[index].turnCount &&
          summary.pinned === next[index].pinned &&
          summary.labels === next[index].labels &&
          summary.workspace === next[index].workspace &&
          summary.savedCount === next[index].savedCount &&
          summary.firstSavedTurnId === next[index].firstSavedTurnId,
      );
    if (same) return prev;
    summariesRef.current = next;
    return next;
  }, [conversations]);

  return {
    turns,
    historyReady,
    streamingConversationId,
    askedTurns,
    scope: activeConversation?.scope ?? EMPTY_SCOPE,
    conversationWorkspace: activeConversation ? conversationWorkspace(activeConversation) : null,
    setScope,
    pruneScopes,
    removeFromScopes,
    editAndResend,
    importConversation,
    importConversations,
    importPreview,
    pending,
    stage,
    stageTiming,
    persistError,
    persistPartial,
    conversations: summaries,
    savedAnswers,
    answersCiting,
    answersIn,
    activeConversationId: activeConversation?.id ?? null,
    ask,
    cancel,
    clear,
    newConversation,
    switchConversation,
    renameConversation,
    deleteConversation,
    restoreConversation,
    setPinned,
    setLabels,
    renameLabel,
    searchConversations,
    setTurnFeedback,
    setSourceIrrelevant,
    setTurnSaved,
    restoreAnswerVersion,
    setTurnReviewed,
    allConversationsJson,
  };
}
