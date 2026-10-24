import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ApiClientError, api, apiErrorMessage, getWorkspace } from '../api/client';
import type {
  FeedbackRating,
  HistoryMessage,
  LlmProvider,
  QuestionOptions,
  QuestionOverrides,
} from '../api/types';
import type { ChatTurn } from '../components/ChatMessage';
import { newId } from '../utils/id';
import { sliceWhole } from '../utils/text';
import { useChatPersistence } from './useChatPersistence';
import { useConversationActions } from './useConversationActions';
import type { ImportItem } from './useConversationActions';
export type { ImportItem } from './useConversationActions';
import {
  EMPTY_SCOPE,
  TITLE_MAX_CHARS,
  applyDeleteHeldDuringStream,
  byRecentUse,
  capConversations,
  cleanFollowups,
  conversationWorkspace,
  deriveTitle,
  loadStorage,
  searchConversations as searchStoredConversations,
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

// Mirrors the server's own REQUEST_HISTORY_MAX_MESSAGES cap (api/schemas.py) — kept
// in sync so a client-side trim never silently disagrees with what the server would
// accept anyway. The char budget is a client-only practical guard against crowding
// a small local model's context window; the server has its own count-based backstop.
const HISTORY_MAX_MESSAGES = 12;
const HISTORY_CHAR_BUDGET = 8000;
// Mirrors the server's REQUEST_HISTORY_MESSAGE_MAX_CHARS (api/schemas.py) per-message
// cap. Without this, one assistant answer over 4000 chars survives the total-budget
// trim below (it's the only message so far) and the server 422s the whole request.
const HISTORY_MESSAGE_MAX_CHARS = 4000;
// How far back from the cap a whitespace break is still worth taking. A hard cut lands
// mid-word and can sever a trailing qualifier or negation, which reads to the model as
// something the speaker never said; backing up to the last space avoids that. Bounded
// because text with no whitespace in its tail (a base64 blob, unspaced CJK) would
// otherwise back off arbitrarily far — past this, a clean cut beats a short message.
// A summary or comparison answer sent back as history: its gist is what a follow-up
// ("what does it say about deadlines?") needs, and at the full 4000 characters one
// summary took half of HISTORY_CHAR_BUDGET from the exchanges before it.
const SUMMARY_HISTORY_MAX_CHARS = 1500;

/** Truncate to HISTORY_MESSAGE_MAX_CHARS, preferring a nearby whitespace boundary.
 *
 * The result is always <= HISTORY_MESSAGE_MAX_CHARS: the ellipsis has to fit inside the
 * server's per-message limit too, so the budget it occupies is reserved up front rather
 * than appended after a full-width cut (which would land one char over and 422). */
function truncateForHistory(content: string, maxChars = HISTORY_MESSAGE_MAX_CHARS): string {
  if (content.length <= maxChars) return content;
  const ELLIPSIS = '…';
  const budget = maxChars - ELLIPSIS.length;
  const hardCut = sliceWhole(content, budget);
  const lastBreak = hardCut.lastIndexOf(' ');
  const backoff = Math.floor(maxChars * 0.12);
  const cut = lastBreak >= budget - backoff ? hardCut.slice(0, lastBreak) : hardCut;
  // The marker is for the model's benefit: without it a truncated answer looks complete,
  // so it may treat a severed sentence as the speaker's final word.
  return `${cut.trimEnd()}${ELLIPSIS}`;
}

function isCompleteAnswer(turn: ChatTurn | undefined): boolean {
  return (
    turn?.role === 'assistant' &&
    turn.content.trim().length > 0 &&
    // A stopped answer is a fragment; sent back as history it reads to the model as
    // what it actually concluded. Same for one cut short by a failed connection.
    !turn.stopped &&
    !turn.incomplete &&
    // "Not enough information" answered nothing: sent back, the server's rewrite of the
    // next question folded the refusal into the search and the model refused again.
    !turn.notFound
  );
}

/** Build bounded conversation history from prior turns: only answered questions and
 * their answers (a question that failed, was stopped or is being asked again has no
 * answer, and sent alone it read to the model as two questions in a row), most recent
 * first up to HISTORY_MAX_MESSAGES, each truncated to HISTORY_MESSAGE_MAX_CHARS, then
 * trimmed to a total char budget (dropping the oldest first), restored to
 * chronological order. */
export function buildHistory(turns: ChatTurn[]): HistoryMessage[] {
  const candidates = turns
    .filter((turn, index): turn is ChatTurn & { role: 'user' | 'assistant' } =>
      turn.role === 'user'
        ? turn.content.trim().length > 0 && isCompleteAnswer(turns[index + 1])
        : isCompleteAnswer(turn) && turns[index - 1]?.role === 'user',
    )
    .slice(-HISTORY_MAX_MESSAGES);

  const withinBudget: HistoryMessage[] = [];
  let charsUsed = 0;
  for (let i = candidates.length - 1; i >= 0; i -= 1) {
    const turn = candidates[i];
    const isSummary = turn.role === 'assistant' && Boolean(candidates[i - 1]?.task);
    const content = truncateForHistory(turn.content, isSummary ? SUMMARY_HISTORY_MAX_CHARS : undefined);
    charsUsed += content.length;
    if (charsUsed > HISTORY_CHAR_BUDGET && withinBudget.length > 0) break;
    withinBudget.unshift({ role: candidates[i].role, content });
  }
  // The count cap can split a pair: history never opens on an answer to a question
  // it doesn't carry.
  if (withinBudget[0]?.role === 'assistant') withinBudget.shift();
  return withinBudget;
}

/** History for asking ``question`` again (Retry, Regenerate): the exchange it repeats is
 *  left out, or the model saw the question twice, the second time already answered. */
export function historyForQuestion(turns: ChatTurn[], question: string): HistoryMessage[] {
  const history = buildHistory(turns);
  const last = history.length - 2;
  if (last >= 0 && history[last].role === 'user' && history[last].content === question) {
    return history.slice(0, last);
  }
  return history;
}

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
  deleteConversation: (id: string) => void;
  /** Pin or unpin; false (nothing changes) past MAX_PINNED_CONVERSATIONS pins. */
  setPinned: (id: string, pinned: boolean) => boolean;
  /** Conversations whose question or answer text contains `query`, by id. Reads the
   *  latest history, so call it from an effect or a memo keyed on `conversations`. */
  searchConversations: (query: string) => Map<string, ConversationMatch>;
  /** `undefined` takes a rating back (it could not be sent). */
  setTurnFeedback: (turnId: string, rating: FeedbackRating | undefined) => void;
  /** Every conversation with turns, as one JSON document (Export all conversations);
   *  importConversations reads it back. */
  allConversationsJson: () => string;
  /** Mark (or, after a failed send, unmark) one cited source of a turn not relevant. */
  setSourceIrrelevant: (turnId: string, sourceNumber: number, marked: boolean) => void;
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
function countDown(
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
  const { persistError, persistPartial, historyReady } = useChatPersistence(storage, setStorage, pending, {
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
    let streamedText = false;

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
          // Asked with the conversation as history: rated as a follow-up (feedback kind).
          // A summary sends none.
          if (!task && history.length > 0) created.followUp = true;
          if (scoped && !task) created.scoped = true;
          return [...prev, update(created)];
        }
        return prev.map((turn) => (turn.id === assistantTurnId ? update(turn) : turn));
      }, touch);
    };

    // Deltas are coalesced into one state update per animation frame. Each update
    // re-renders the streaming message, whose markdown is re-parsed over the whole
    // answer so far — per token that is quadratic in answer length on a fast provider.
    bufferedDeltaRef.current = '';
    let frameId: number | null = null;
    const flushDelta = () => {
      if (frameId !== null) {
        cancelAnimationFrame(frameId);
        frameId = null;
      }
      if (!bufferedDeltaRef.current) return;
      const text = bufferedDeltaRef.current;
      bufferedDeltaRef.current = '';
      updateAssistantTurn((turn) => ({ ...turn, content: turn.content + text }), false);
    };

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
      for (let attempt = 1; ; attempt += 1) {
        try {
          await api.askQuestionStream(
            question,
            provider,
            overrides,
            filenames,
            history,
            {
              onStage: (serverStage, progress, reading) => {
                enterStage(stageLabel(serverStage, progress), reading?.etaMs ?? null);
              },
              onSources: (sources, traceId) => {
                // Sources land the moment retrieval finishes, which is when generation
                // starts; a newer server follows with "reading" (or "waiting").
                enterStage(task ? 'writing the summary…' : 'generating answer…');
                updateAssistantTurn((turn) => ({ ...turn, sources, traceId }));
              },
              onDelta: (text) => {
                if (!streamedText) {
                  // The model is writing: "reading the passages…" is over.
                  streamedText = true;
                  enterStage(task ? 'writing the summary…' : 'generating answer…');
                }
                bufferedDeltaRef.current += text;
                if (frameId === null) frameId = requestAnimationFrame(flushDelta);
              },
              onFollowups: (questions) => {
                const followups = cleanFollowups(questions);
                if (followups.length === 0) return;
                // Onto this answer by id, and never creating it: by now it may have been
                // cleared or deleted, and other questions may have been asked since.
                setConversationTurns(
                  conversationId,
                  (prev) =>
                    prev.some((turn) => turn.id === assistantTurnId)
                      ? prev.map((turn) => (turn.id === assistantTurnId ? { ...turn, followups } : turn))
                      : prev,
                  false,
                );
              },
              onDone: (answer, sources, timings, traceId, cached, grounding) => {
                flushDelta();
                const status = grounding?.answerStatus;
                const notFound = status === 'insufficient_context' || status === 'no_context';
                updateAssistantTurn((turn) => ({
                  ...turn,
                  content: answer,
                  sources,
                  // The passages it was given, kept when it found nothing in them, so
                  // the reader can see what was looked at (text dropped: CitationCard
                  // opens the passage by chunk id).
                  ...(notFound && turn.sources.length > 0 && sources.length === 0
                    ? {
                        contextSources: turn.sources.map(({ spans: _spans, ...source }) => ({
                          ...source,
                          text: '',
                        })),
                      }
                    : {}),
                  ...(grounding?.autoCited ? { autoCited: true } : {}),
                  timings,
                  traceId,
                  ...(cached ? { cached: true } : {}),
                  ...(notFound ? { notFound: true } : {}),
                  ...(status === 'uncited' ? { uncited: true } : {}),
                  ...(grounding?.weakMatches ? { weakMatch: true } : {}),
                  ...(grounding?.truncated ? { truncated: true } : {}),
                  ...(grounding?.coverage && grounding.coverage.read < grounding.coverage.total
                    ? { coverage: grounding.coverage }
                    : {}),
                  ...(typeof grounding?.topRerankScore === 'number'
                    ? { topRerankScore: grounding.topRerankScore }
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
                }));
                finish();
                // Still open for follow-ups; aborted on unmount or by a newer question here.
                trailing = { conversationId, controller };
                trailingRef.current.add(trailing);
              },
            },
            controller.signal,
            options,
          );
          break;
        } catch (err) {
          const waitSeconds = busyRetrySeconds(err, attempt);
          if (waitSeconds === null || controller.signal.aborted) throw err;
          await countDown(waitSeconds, controller.signal, (left) =>
            setStage(`server busy — asking again in ${left}s…`),
          );
          if (controller.signal.aborted) throw err;
          setStage(DEFAULT_STAGE_LABEL);
        }
      }
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
    ) =>
      runAsk(
        activeConversationRef.current?.id ?? null,
        question,
        historyForQuestion(activeConversationRef.current?.turns ?? [], question),
        provider,
        overrides,
        filenames,
        options,
      ),
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
      const prior = source.turns.slice(0, index);
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
      };
      setStorage((prev) => ({
        ...prev,
        activeConversationId: fork.id,
        conversations: capConversations([fork, ...prev.conversations], fork.id),
      }));
      await runAsk(fork.id, question, buildHistory(prior), provider, overrides, filenames, options);
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
        return {
          ...conversation,
          scope: { tags: keptTags, filenames: kept },
          editedAt: Date.now(),
        };
      });
      return changed ? { ...prev, conversations } : prev;
    });
  }, []);

  const pruneScopes = useCallback(
    (indexedFilenames: string[], indexedTags?: string[]) => {
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

  const deleteConversation = useCallback((id: string) => {
    if (isStreamingInto(id)) return;
    setStorage((prev) => withoutConversation(prev, id, Date.now()));
  }, []);

  const searchConversations = useCallback(
    (query: string) => searchStoredConversations(storageRef.current.conversations, query),
    [],
  );

  const allConversationsJson = useCallback(
    () =>
      JSON.stringify(
        {
          docrag_conversations: storageRef.current.conversations
            .filter((conversation) => conversation.turns.length > 0)
            .sort(byRecentUse)
            .map((conversation) => ({ title: conversation.title, turns: conversation.turns })),
        },
        null,
        2,
      ),
    [],
  );

  const setSourceIrrelevant = useCallback(
    (turnId: string, sourceNumber: number, marked: boolean) => {
      setStorage((prev) => ({
        ...prev,
        conversations: prev.conversations.map((conversation) =>
          conversation.turns.some((turn) => turn.id === turnId)
            ? {
                ...conversation,
                turns: conversation.turns.map((turn) => {
                  if (turn.id !== turnId) return turn;
                  const others = (turn.irrelevantSources ?? []).filter((n) => n !== sourceNumber);
                  const next = marked ? [...others, sourceNumber] : others;
                  const { irrelevantSources: _previous, ...rest } = turn;
                  return next.length > 0 ? { ...rest, irrelevantSources: next } : rest;
                }),
                editedAt: Date.now(),
              }
            : conversation,
        ),
      }));
    },
    [],
  );

  const setTurnFeedback = useCallback((turnId: string, rating: FeedbackRating | undefined) => {
    // Stored on the turn, so a rating survives a reload or a conversation switch and
    // the buttons can't send a duplicate row for the same answer.
    setStorage((prev) => ({
      ...prev,
      conversations: prev.conversations.map((conversation) =>
        conversation.turns.some((turn) => turn.id === turnId)
          ? {
              ...conversation,
              turns: conversation.turns.map((turn) =>
                turn.id === turnId ? { ...turn, feedback: rating } : turn,
              ),
              editedAt: Date.now(),
            }
          : conversation,
      ),
    }));
  }, []);

  // Keeps the previous array when nothing the list shows changed: `conversations` is a
  // new array on every streamed frame, and a new summaries array re-rendered the list.
  const summariesRef = useRef<ConversationSummary[]>([]);
  const summaries: ConversationSummary[] = useMemo(() => {
    const next = [...conversations].sort(byRecentUse).map((conversation) => ({
      id: conversation.id,
      title: conversation.title,
      updatedAt: conversation.updatedAt,
      turnCount: conversation.turns.filter((turn) => turn.role === 'user').length,
      ...(conversation.pinned ? { pinned: true } : {}),
      ...(conversation.workspace && conversation.turns.length > 0
        ? { workspace: conversation.workspace }
        : {}),
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
          summary.workspace === next[index].workspace,
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
    activeConversationId: activeConversation?.id ?? null,
    ask,
    cancel,
    clear,
    newConversation,
    switchConversation,
    renameConversation,
    deleteConversation,
    setPinned,
    searchConversations,
    setTurnFeedback,
    setSourceIrrelevant,
    allConversationsJson,
  };
}
