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
import { GENERATING_LABEL, answerStreamHandlers, createDeltaBuffer } from './streamHandlers';
import { ANSWER_RUN_FIELDS, mergeRegenerated } from '../utils/answerVersions';
import { sanitizeStyle } from './chat/sanitize';
import type { ImportItem } from './useConversationActions';
import { turnCounts, useAnswerQueries, type SavedAnswer } from './chat/answers';
import { useTurnMarks } from './chat/turnMarks';
// Re-exported for the components and tests that import it from here.
export type { SavedAnswer } from './chat/answers';

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
import { addToTrash } from '../utils/trash';
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
  continuing: 'continuing the answer…',
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
  /** Bumped when this hook put conversations into Recently deleted itself: one the
   *  history cap let go, or a conversation's turns before a Clear. */
  trashedVersion: number;
  /** Any answer in flight, in whichever conversation. */
  pending: boolean;
  /** The conversation the answer in flight is written to; null when none is. */
  streamingConversationId: string | null;
  /** The answer a Continue is writing into, while it runs (null otherwise). */
  continuingTurnId: string | null;
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
  restoreConversation: (conversation: Conversation, usedNow?: boolean) => void;
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
  setTurnFeedback: (
    turnId: string,
    rating: FeedbackRating | undefined,
    conversationId?: string,
  ) => void;
  /** Every conversation with turns, as one JSON document (Export all conversations);
   *  importConversations reads it back. */
  allConversationsJson: (label?: string) => string;
  /** Mark (or, after a failed send, unmark) one cited source of a turn not relevant. */
  setSourceIrrelevant: (
    turnId: string,
    sourceNumber: number,
    marked: boolean,
    conversationId?: string,
  ) => void;
  /** Save (or unsave) an answer of ``conversationId`` (the open one when omitted). */
  setTurnSaved: (turnId: string, saved: boolean, conversationId?: string) => void;
  /** The reader's note on a saved answer ("" removes it). */
  setTurnNote: (conversationId: string, turnId: string, note: string) => void;
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
  const emptyScope = status === 'empty_scope';
  const notFound = status === 'insufficient_context' || status === 'no_context' || emptyScope;
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
    ...(grounding?.unsupportedClaims ? { unsupportedClaims: grounding.unsupportedClaims } : {}),
    ...(grounding?.unsupportedNumbers ? { unsupportedNumbers: grounding.unsupportedNumbers } : {}),
    ...(grounding?.searchedAs ? { searchedAs: grounding.searchedAs } : {}),
    timings,
    traceId,
    ...(cached ? { cached: true } : {}),
    ...(notFound ? { notFound: true } : {}),
    ...(emptyScope ? { emptyScope: true } : {}),
    ...(status === 'uncited' ? { uncited: true } : {}),
    ...(grounding?.weakMatches ? { weakMatch: true } : {}),
    ...(grounding?.truncated ? { truncated: true } : {}),
    ...(grounding?.promptCut ? { promptCut: true } : {}),
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


/** ``turns`` without a stopped Regenerate's own pair: its answer (``answerId``, when it
 *  was created) and the copy of ``question`` asked last for it. */
export function withoutRegenerateAttempt(
  turns: ChatTurn[],
  answerId: string,
  question: string,
): ChatTurn[] {
  const kept = turns.filter((turn) => turn.id !== answerId);
  const last = kept[kept.length - 1];
  return last?.role === 'user' && last.content === question ? kept.slice(0, -1) : kept;
}

export function useChat(): UseChatResult {
  const [storage, setStorage] = useState<ChatStorageV2>(loadStorage);
  const [pending, setPending] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [stageTiming, setStageTiming] = useState<StageTiming | null>(null);
  // Which conversation the answer in flight is written to (null when none is), and the
  // one asked in last: other conversations can be opened while an answer streams.
  const [streamingConversationId, setStreamingConversationId] = useState<string | null>(null);
  // The answer a Continue is writing into (any answer, not only the last); null while
  // a new answer streams at the end of the thread.
  const [continuingTurnId, setContinuingTurnId] = useState<string | null>(null);
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
  // Conversations the history cap (MAX_CONVERSATIONS) let go go to Recently deleted, as
  // a delete does: dropped silently, a month-old conversation was simply gone. One that
  // left with a tombstone was deleted (here, in another tab, or to make room for an
  // import the user confirmed) and is the deleter's to trash. Watched only once the
  // full history has loaded: the first-paint copy is a subset of it.
  const [trashedVersion, setTrashedVersion] = useState(0);
  const watchedConversationsRef = useRef<Conversation[] | null>(null);
  useEffect(() => {
    if (!historyLoaded) {
      watchedConversationsRef.current = null;
      return;
    }
    const previous = watchedConversationsRef.current;
    watchedConversationsRef.current = conversations;
    if (previous === null || previous === conversations) return;
    const live = new Set(conversations.map((conversation) => conversation.id));
    const tombstoned = new Set((storage.deleted ?? []).map((entry) => entry.id));
    const dropped = previous.filter(
      (conversation) =>
        !live.has(conversation.id) && !tombstoned.has(conversation.id) && conversation.turns.length > 0,
    );
    if (dropped.length === 0) return;
    for (const conversation of dropped) addToTrash(conversation);
    setTrashedVersion((version) => version + 1);
  }, [conversations, historyLoaded, storage.deleted]);
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
    // "Continue": the rest of a cut-off answer is written into that answer itself — no
    // second copy of its question, no fragment at the end of the thread — and whatever
    // goes wrong (stopped, failed, its trace gone) leaves the answer as it was.
    const continued =
      options?.continueFrom && options.replacesAnswer
        ? activeConversationRef.current?.turns.find(
            (turn) => turn.id === options.replacesAnswer && turn.role === 'assistant',
          )
        : undefined;
    if (!continued) updateTurns((prev) => {
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
    const assistantTurnId = continued?.id ?? newId();
    // A Continue changes nothing on the answer before `done` (below): a reload or a
    // discarded tab meanwhile finds the answer as it was, not marked stopped with raw
    // continuation text on it.
    streamingTurnIdRef.current = continued ? null : assistantTurnId;
    if (continued) setContinuingTurnId(assistantTurnId);
    const updateAssistantTurn = (update: (turn: ChatTurn) => ChatTurn, touch = true) => {
      updateTurns((prev) => {
        const existing = prev.find((turn) => turn.id === assistantTurnId);
        // The answer being continued was deleted or cleared meanwhile: nothing to add to.
        if (!existing && continued) return prev;
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
    // A Continue's text isn't shown as it streams: the server joins it to the answer
    // (a space or none, a restart dropped) only at `done`, and the raw pieces read
    // "butbusiness" or repeated the answer until then. The answer stays as it was, with
    // "continuing the answer…", and `done` brings the whole text.
    const deltas = createDeltaBuffer(
      continued ? { current: '' } : bufferedDeltaRef,
      (text) => {
        if (continued) return;
        updateAssistantTurn((turn) => ({ ...turn, content: turn.content + text }), false);
      },
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
      setContinuingTurnId(null);
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
              // A Continue reads "continuing the answer…" while the model writes.
              enterStage: continued
                ? (label, etaMs) =>
                    enterStage(label === GENERATING_LABEL ? stageLabel('continuing') : label, etaMs)
                : enterStage,
              stageLabel,
              // A Continue's sources are the answer's own, and its new trace id is only
              // stored once written (`done`): set before, the Inspector looked it up
              // while the server had no such trace yet and said it was gone.
              updateAnswer: continued ? () => undefined : updateAssistantTurn,
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
                updateAssistantTurn((turn) => {
                  if (!continued) {
                    return answeredTurn(turn, answer, sources, timings, traceId, cached, grounding);
                  }
                  // The whole answer now: what described its first part (how it ended,
                  // its checks and notes, a rating of that text) no longer applies, and
                  // what the continuation cost is added to what that part did.
                  const whole: Partial<ChatTurn> = { ...turn };
                  for (const field of ANSWER_RUN_FIELDS) delete whole[field];
                  // Finished now, however its first part ended.
                  delete whole.stopped;
                  delete whole.incomplete;
                  const done = answeredTurn(
                    { ...(whole as ChatTurn), timings: null },
                    answer,
                    sources,
                    timings,
                    traceId,
                    cached,
                    grounding,
                  );
                  const before = continued.usage;
                  if (!before || !done.usage) return done;
                  const cost =
                    before.costUsd === undefined && done.usage.costUsd === undefined
                      ? {}
                      : { costUsd: (before.costUsd ?? 0) + (done.usage.costUsd ?? 0) };
                  return {
                    ...done,
                    usage: {
                      prompt: before.prompt + done.usage.prompt,
                      completion: before.completion + done.usage.completion,
                      ...cost,
                    },
                  };
                });
                // Regenerated: the answer it replaces becomes its earlier version (only
                // now — a regenerate that failed leaves the old answer where it was).
                const replaces = continued ? undefined : options?.replacesAnswer;
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
      if (continued) {
        // As it was before Continue. A trace the server no longer has (a restart, the
        // buffer moved on) or an older server: it can't be continued at all — said
        // once, the button gone. Anything else may work on another try.
        const status = err instanceof ApiClientError ? err.statusCode : undefined;
        const gone = !controller.signal.aborted && (status === 404 || status === 409);
        const failed = controller.signal.aborted ? undefined : apiErrorMessage(err, 'Something went wrong.');
        updateTurns((prev) =>
          prev.map((turn) =>
            turn.id === assistantTurnId
              ? {
                  ...continued,
                  ...(gone ? { cannotContinue: true } : {}),
                  ...(failed && !gone ? { continueFailed: failed } : {}),
                }
              : turn,
          ),
        );
      } else if (controller.signal.aborted && options?.replacesAnswer) {
        // A Regenerate stopped: the answer it was to replace stays the answer, and the
        // second copy of its question (with whatever streamed) goes, rather than ending
        // the thread with the same question asked twice.
        updateTurns((prev) => withoutRegenerateAttempt(prev, assistantTurnId, question));
      } else if (controller.signal.aborted) {
        // Drop a content-less assistant turn: cancelling after `sources` but before the
        // first delta would otherwise leave an empty answer bubble behind. A partial one
        // is kept but marked, so after a reload it doesn't read as a finished answer.
        updateTurns((prev) =>
          prev
            .filter((turn) => turn.id !== assistantTurnId || turn.content !== '')
            .map((turn) => (turn.id === assistantTurnId ? { ...turn, stopped: true } : turn)),
        );
      } else if (options?.replacesAnswer) {
        // A Regenerate failed: the answer it was to replace stays, with why, instead of
        // the same question asked twice and an error row whose Retry no longer
        // replaced anything.
        const message = apiErrorMessage(err, 'Something went wrong.');
        const replaced = options.replacesAnswer;
        updateTurns((prev) =>
          withoutRegenerateAttempt(prev, assistantTurnId, question).map((turn) =>
            turn.id === replaced ? { ...turn, regenerateFailed: message } : turn,
          ),
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
    const active = activeConversationRef.current;
    if (isStreamingInto(active?.id)) return;
    if (active && active.turns.length > 0) {
      // Its turns can be had back from Recently deleted, as a copy of their own (the
      // conversation itself stays, empty, so its id is live).
      addToTrash({ ...active, id: newId(), title: `${active.title} (cleared)`, pinned: false });
      setTrashedVersion((version) => version + 1);
    }
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

  const restoreConversation = useCallback((conversation: Conversation, usedNow = false) => {
    setStorage((prev) => withConversationRestored(prev, conversation, usedNow));
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

  // These change one conversation's turn (the active one unless named): an edited
  // question's copy shares the earlier turns' ids, so matching the id alone rated,
  // marked or swapped the answer in both conversations.
  const {
    setSourceIrrelevant,
    setTurnSaved,
    setTurnNote,
    setTurnReviewed,
    restoreAnswerVersion,
    setTurnFeedback,
  } = useTurnMarks(setStorage);

  const { savedAnswers, answersIn, answersCiting } = useAnswerQueries(conversations, storageRef);

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
    trashedVersion,
    streamingConversationId,
    continuingTurnId,
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
    setTurnNote,
    restoreAnswerVersion,
    setTurnReviewed,
    allConversationsJson,
  };
}
