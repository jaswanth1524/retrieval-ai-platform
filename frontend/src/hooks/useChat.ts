import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, apiErrorMessage } from '../api/client';
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
import { chatStoreAvailable, clearChatStore, loadChatStore, saveChatStore } from './chatStore';
import {
  CHAT_STORAGE_KEY,
  EMPTY_SCOPE,
  MAX_PINNED_CONVERSATIONS,
  TITLE_MAX_CHARS,
  applyDeleteHeldDuringStream,
  byRecentUse,
  capConversations,
  deriveTitle,
  emptyConversation,
  isOnlyEmpty,
  loadStorage,
  mergeConversations,
  parseConversations,
  parseDeleted,
  persistStorage,
  sanitizeTurn,
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
const HISTORY_TRUNCATION_BACKOFF_CHARS = Math.floor(HISTORY_MESSAGE_MAX_CHARS * 0.12);

/** Truncate to HISTORY_MESSAGE_MAX_CHARS, preferring a nearby whitespace boundary.
 *
 * The result is always <= HISTORY_MESSAGE_MAX_CHARS: the ellipsis has to fit inside the
 * server's per-message limit too, so the budget it occupies is reserved up front rather
 * than appended after a full-width cut (which would land one char over and 422). */
function truncateForHistory(content: string): string {
  if (content.length <= HISTORY_MESSAGE_MAX_CHARS) return content;
  const ELLIPSIS = '…';
  const budget = HISTORY_MESSAGE_MAX_CHARS - ELLIPSIS.length;
  const hardCut = sliceWhole(content, budget);
  const lastBreak = hardCut.lastIndexOf(' ');
  const cut = lastBreak >= budget - HISTORY_TRUNCATION_BACKOFF_CHARS ? hardCut.slice(0, lastBreak) : hardCut;
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
    !turn.incomplete
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
    const content = truncateForHistory(candidates[i].content);
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
  verifying_citations: 'checking citations…',
};
const DEFAULT_STAGE_LABEL = 'retrieving…';

export function stageLabel(stage: string): string {
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
  persistError: boolean;
  persistPartial: boolean;
  conversations: ConversationSummary[];
  activeConversationId: string | null;
  // The active conversation's search scope.
  scope: ConversationScope;
  /** A scope, or an updater of the current one — updaters compose, so two setters
   *  called in the same tick (clear files, then clear tags) don't undo each other. */
  setScope: (scope: ConversationScope | ((prev: ConversationScope) => ConversationScope)) => void;
  /** Drop names that are no longer indexed, and tags no document carries any more
   *  (when ``indexedTags`` is given), from every conversation's scope. */
  pruneScopes: (indexedFilenames: string[], indexedTags?: string[]) => void;
  /** Drop just these filenames from every conversation's scope. Unlike pruneScopes it
   *  needs no snapshot of the whole corpus, which can be stale by the time it runs. */
  removeFromScopes: (filenames: string[]) => void;
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

// How long IndexedDB writes are held back so a burst of changes is written once.
const CHAT_STORE_SAVE_DELAY_MS = 300;

export function useChat(): UseChatResult {
  const [storage, setStorage] = useState<ChatStorageV2>(loadStorage);
  const [pending, setPending] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  // Which conversation the answer in flight is written to (null when none is), and the
  // one asked in last: other conversations can be opened while an answer streams.
  const [streamingConversationId, setStreamingConversationId] = useState<string | null>(null);
  const [askedConversationId, setAskedConversationId] = useState<string | null>(null);
  // True when the last persist attempt failed even after the reduced-payload retry
  // below — the session is memory-only from that point on until a write succeeds.
  const [persistError, setPersistError] = useState(false);
  // True when the reduced-payload retry *succeeded*: the active conversation is saved
  // but the others aren't. Distinct from persistError because "some of this survives a
  // reload" and "none of this survives a reload" need different warnings.
  const [persistPartial, setPersistPartial] = useState(false);

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
  // The next IndexedDB write, held back by CHAT_STORE_SAVE_DELAY_MS (see the save effect).
  const pendingChatStoreRef = useRef<ChatStorageV2 | null>(null);
  const chatStoreTimerRef = useRef<number | undefined>(undefined);
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);
  const flushChatStore = () => {
    window.clearTimeout(chatStoreTimerRef.current);
    const pendingStorage = pendingChatStoreRef.current;
    pendingChatStoreRef.current = null;
    if (pendingStorage === null) return;
    // Mirrors persistStorage: a lone empty conversation clears the store rather than
    // being saved, so it can't come back next to a new one after a reload.
    void (isOnlyEmpty(pendingStorage) ? clearChatStore() : saveChatStore(pendingStorage)).then(
      (saved) => {
        if (!mountedRef.current) return;
        // The full history is safe in IndexedDB: a localStorage quota miss loses nothing.
        if (saved) {
          setPersistError(false);
          setPersistPartial(false);
        } else {
          // Only localStorage's compact copy (older passage text and turns dropped) is
          // saved now. This used to pass without a word.
          setPersistPartial(true);
        }
      },
    );
  };
  // IndexedDB holds the full history (see chatStore.ts). Nothing is written there until
  // it has been read: saving first would replace it with localStorage's compact copy.
  const [chatStoreReady, setChatStoreReady] = useState(() => !chatStoreAvailable());

  useEffect(() => {
    if (chatStoreReady) return;
    let cancelled = false;
    void loadChatStore().then((raw) => {
      if (cancelled) return;
      // An empty conversation holds nothing to restore, and merging one in next to the
      // fresh one this load already made showed two "New chat" rows.
      const incoming = parseConversations(raw)?.filter((c) => c.turns.length > 0);
      const deleted = parseDeleted(raw);
      if ((incoming && incoming.length > 0) || deleted?.length) {
        setStorage((prev) =>
          mergeConversations(
            prev,
            incoming ?? [],
            true,
            deleted,
            Date.now(),
            streamingConversationIdRef.current,
          ),
        );
      }
      setChatStoreReady(true);
    });
    return () => {
      cancelled = true;
    };
    // Once, on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  // The assistant turn being streamed right now, if any — see the pagehide flush.
  const streamingTurnIdRef = useRef<string | null>(null);
  // The conversation it streams into, and the text not yet applied to it (deltas are
  // applied once per animation frame, and a hidden tab runs no frames).
  const streamingConversationIdRef = useRef<string | null>(null);
  const bufferedDeltaRef = useRef('');

  // What a reload or a discarded tab would find: the answer being streamed is cut off
  // there, so it is saved as stopped — saved as-is it read as finished after the
  // reload, and went back as history.
  const persistStreamSnapshot = () => {
    const turnId = streamingTurnIdRef.current;
    const current = storageRef.current;
    const pendingText = bufferedDeltaRef.current;
    const snapshot =
      turnId === null
        ? current
        : {
            ...current,
            conversations: current.conversations.map((conversation) => ({
              ...conversation,
              turns: conversation.turns.map((turn) =>
                turn.id === turnId
                  ? { ...turn, content: turn.content + pendingText, stopped: true }
                  : turn,
              ),
            })),
          };
    persistStorage(snapshot);
    // Best effort: the page may be gone before this commits; localStorage has it.
    // This snapshot supersedes any write still waiting on its timer.
    window.clearTimeout(chatStoreTimerRef.current);
    pendingChatStoreRef.current = null;
    if (chatStoreAvailable()) void saveChatStore(snapshot);
  };

  // While an answer streams: the other conversations and the selection as last seen,
  // to tell a delta (skipped) from a rename, pin, delete or switch elsewhere (saved).
  const outsideStreamRef = useRef<{
    others: Conversation[];
    active: string | null;
    deleted: ChatStorageV2['deleted'];
  } | null>(null);

  useEffect(() => {
    // Skip writes while a stream is in flight — the assistant turn's content mutates
    // on every delta, and persisting each one would thrash localStorage. `pagehide`
    // below covers a reload or navigation that lands mid-stream. A change outside the
    // streaming conversation is saved at once, though: conversations stay usable
    // mid-answer, and a rename or delete made then used to wait for the answer to end
    // (other tabs didn't see it, and closing the tab lost it).
    if (pending) {
      const streamingId = streamingConversationIdRef.current;
      const seen = {
        others: storage.conversations.filter((conversation) => conversation.id !== streamingId),
        active: storage.activeConversationId,
        deleted: storage.deleted,
      };
      const before = outsideStreamRef.current;
      outsideStreamRef.current = seen;
      const changed =
        before !== null &&
        (before.active !== seen.active ||
          before.deleted !== seen.deleted ||
          before.others.length !== seen.others.length ||
          before.others.some((conversation, index) => conversation !== seen.others[index]));
      if (changed && chatStoreReady) persistStreamSnapshot();
      return;
    }
    outsideStreamRef.current = null;
    const outcome = persistStorage(storage);
    setPersistError(outcome === 'failed');
    setPersistPartial(outcome === 'partial');
    if (!chatStoreReady || !chatStoreAvailable()) return;
    // Coalesced: every scope click, rename or rating used to rewrite the whole history
    // in IndexedDB at once; a burst of changes now costs one write. The timer is not
    // cancelled on unmount (the write must still land), and a hidden tab or pagehide
    // flushes it early.
    pendingChatStoreRef.current = storage;
    window.clearTimeout(chatStoreTimerRef.current);
    chatStoreTimerRef.current = window.setTimeout(flushChatStore, CHAT_STORE_SAVE_DELAY_MS);
  }, [storage, pending, chatStoreReady]);

  useEffect(() => {
    const onVisibility = () => {
      if (document.visibilityState !== 'hidden') return;
      // Mid-answer nothing is saved until the answer ends, and a mobile browser can
      // discard a hidden tab without a pagehide: the question and the answer so far
      // were lost. The finished answer replaces this copy when it lands.
      if (streamingTurnIdRef.current !== null) persistStreamSnapshot();
      else flushChatStore();
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
    // flushChatStore and persistStreamSnapshot only read refs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    // A reload, a link followed in this tab, or a closed tab during a 1-2 minute answer
    // used to lose the question and everything streamed so far.
    window.addEventListener('pagehide', persistStreamSnapshot);
    return () => window.removeEventListener('pagehide', persistStreamSnapshot);
    // persistStreamSnapshot only reads refs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.key !== CHAT_STORAGE_KEY || !event.newValue) return;
      try {
        const payload: unknown = JSON.parse(event.newValue);
        const incoming = parseConversations(payload);
        if (!incoming) return;
        setStorage((prev) =>
          mergeConversations(
            prev,
            incoming,
            false,
            parseDeleted(payload),
            Date.now(),
            streamingConversationIdRef.current,
          ),
        );
      } catch {
        // Another tab wrote something unreadable; keep what this tab has.
      }
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  // Tracks the in-flight question's controller: doubles as a reentrancy guard and lets
  // unmount cancel a hung request.
  const inflightRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      inflightRef.current?.abort();
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

    updateTurns((prev) => [...prev, makeTurn('user', question)]);
    setPending(true);
    // Seeded before the first server frame arrives so there is never a gap with a
    // pending request and no label at all.
    setStage(DEFAULT_STAGE_LABEL);

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

    try {
      await api.askQuestionStream(
        question,
        provider,
        overrides,
        filenames,
        history,
        {
          onStage: (serverStage) => {
            setStage(stageLabel(serverStage));
          },
          onSources: (sources, traceId) => {
            // Sources land the moment retrieval finishes, which is exactly when
            // generation starts — no separate server frame needed for it.
            setStage('generating answer…');
            updateAssistantTurn((turn) => ({ ...turn, sources, traceId }));
          },
          onDelta: (text) => {
            bufferedDeltaRef.current += text;
            if (frameId === null) frameId = requestAnimationFrame(flushDelta);
          },
          onDone: (answer, sources, timings, traceId, cached, grounding) => {
            flushDelta();
            const status = grounding?.answerStatus;
            updateAssistantTurn((turn) => ({
              ...turn,
              content: answer,
              sources,
              timings,
              traceId,
              ...(cached ? { cached: true } : {}),
              ...(status === 'insufficient_context' || status === 'no_context'
                ? { notFound: true }
                : {}),
              ...(status === 'uncited' ? { uncited: true } : {}),
              ...(grounding?.truncated ? { truncated: true } : {}),
            }));
          },
        },
        controller.signal,
        options,
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
          { ...makeTurn('error', message), question },
        ]);
      }
    } finally {
      flushDelta();
      streamingTurnIdRef.current = null;
      streamingConversationIdRef.current = null;
      setStreamingConversationId(null);
      inflightRef.current = null;
      if (conversationId) {
        setStorage((prev) => applyDeleteHeldDuringStream(prev, conversationId, startedAt));
      }
      setPending(false);
      setStage(null);
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

  const importConversation = useCallback((raw: unknown, title?: string) => {
    // Fine mid-answer: the answer is written to the conversation it was asked in, by id.
    const candidate = Array.isArray(raw) ? { turns: raw } : raw;
    if (!candidate || typeof candidate !== 'object') return false;
    const record = candidate as Partial<Conversation>;
    if (!Array.isArray(record.turns)) return false;
    // Fresh turn ids: importing the same file twice must not produce clashing keys.
    const turns = record.turns
      .map(sanitizeTurn)
      .filter((turn): turn is ChatTurn => turn !== null)
      .map((turn) => ({ ...turn, id: newId() }));
    if (turns.length === 0) return false;
    const now = Date.now();
    const conversation: Conversation = {
      id: newId(),
      title: (title ?? '').trim().slice(0, TITLE_MAX_CHARS) || deriveTitle(turns),
      createdAt: now,
      updatedAt: now,
      turns,
    };
    setStorage((prev) => ({
      ...prev,
      activeConversationId: conversation.id,
      conversations: capConversations([conversation, ...prev.conversations], conversation.id),
    }));
    return true;
  }, []);

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
  ) => {
    setStorage((prev) => {
      let changed = false;
      const conversations = prev.conversations.map((conversation) => {
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
    (filenames: string[]) => {
      const removed = new Set(filenames);
      keepInScopes((name) => !removed.has(name));
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

  const newConversation = useCallback(() => {
    setStorage((prev) => {
      // Reuse an already-empty active conversation instead of stacking blanks.
      const active = prev.conversations.find((c) => c.id === prev.activeConversationId);
      if (active && active.turns.length === 0) return prev;
      const conversation = emptyConversation();
      // Past the cap, the least recently *used* unpinned conversation goes — not the
      // oldest created, which dropped a conversation still in daily use.
      const conversations = capConversations([conversation, ...prev.conversations], conversation.id);
      return { ...prev, activeConversationId: conversation.id, conversations };
    });
  }, []);

  const switchConversation = useCallback((id: string) => {
    setStorage((prev) =>
      prev.conversations.some((conversation) => conversation.id === id)
        ? { ...prev, activeConversationId: id }
        : prev,
    );
  }, []);

  const renameConversation = useCallback((id: string, title: string) => {
    const trimmed = title.replace(/\s+/g, ' ').trim().slice(0, TITLE_MAX_CHARS);
    if (!trimmed) return;
    setStorage((prev) => ({
      ...prev,
      conversations: prev.conversations.map((conversation) =>
        conversation.id === id ? { ...conversation, title: trimmed, editedAt: Date.now() } : conversation,
      ),
    }));
  }, []);

  const deleteConversation = useCallback((id: string) => {
    if (isStreamingInto(id)) return;
    setStorage((prev) => withoutConversation(prev, id, Date.now()));
  }, []);

  const setPinned = useCallback((id: string, pinned: boolean) => {
    const current = storageRef.current.conversations;
    const target = current.find((conversation) => conversation.id === id);
    if (!target || Boolean(target.pinned) === pinned) return Boolean(target);
    if (pinned && current.filter((c) => c.pinned).length >= MAX_PINNED_CONVERSATIONS) return false;
    setStorage((prev) => ({
      ...prev,
      conversations: prev.conversations.map((conversation) =>
        conversation.id === id
          ? { ...conversation, pinned: pinned || undefined, editedAt: Date.now() }
          : conversation,
      ),
    }));
    return true;
  }, []);

  const searchConversations = useCallback(
    (query: string) => searchStoredConversations(storageRef.current.conversations, query),
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
          summary.pinned === next[index].pinned,
      );
    if (same) return prev;
    summariesRef.current = next;
    return next;
  }, [conversations]);

  return {
    turns,
    historyReady: chatStoreReady,
    streamingConversationId,
    askedTurns,
    scope: activeConversation?.scope ?? EMPTY_SCOPE,
    setScope,
    pruneScopes,
    removeFromScopes,
    editAndResend,
    importConversation,
    pending,
    stage,
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
  };
}
