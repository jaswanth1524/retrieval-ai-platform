import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ApiClientError, api } from '../api/client';
import type {
  FeedbackRating,
  HistoryMessage,
  LlmProvider,
  QuestionOptions,
  QuestionOverrides,
} from '../api/types';
import type { ChatTurn } from '../components/ChatMessage';
import { newId } from '../utils/id';
import { chatStoreAvailable, clearChatStore, loadChatStore, saveChatStore } from './chatStore';

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
  const hardCut = content.slice(0, budget);
  const lastBreak = hardCut.lastIndexOf(' ');
  const cut = lastBreak >= budget - HISTORY_TRUNCATION_BACKOFF_CHARS ? hardCut.slice(0, lastBreak) : hardCut;
  // The marker is for the model's benefit: without it a truncated answer looks complete,
  // so it may treat a severed sentence as the speaker's final word.
  return `${cut.trimEnd()}${ELLIPSIS}`;
}

/** Build bounded conversation history from prior turns: only completed user/assistant
 * turns (error turns carry no useful context and are excluded), most recent first
 * up to HISTORY_MAX_MESSAGES, each truncated to HISTORY_MESSAGE_MAX_CHARS, then
 * trimmed to a total char budget (dropping the oldest first), restored to
 * chronological order. */
export function buildHistory(turns: ChatTurn[]): HistoryMessage[] {
  const candidates = turns
    .filter((turn): turn is ChatTurn & { role: 'user' | 'assistant' } =>
      (turn.role === 'user' || turn.role === 'assistant') &&
      turn.content.trim().length > 0 &&
      // A stopped answer is a fragment; sent back as history it reads to the model as
      // what it actually concluded. Same for one cut short by a failed connection.
      !turn.stopped &&
      !turn.incomplete,
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
  return withinBudget;
}

/** Which documents a conversation's questions search. Empty lists mean "everything". */
export interface ConversationScope {
  filenames: string[];
  tags: string[];
}

export const EMPTY_SCOPE: ConversationScope = { filenames: [], tags: [] };

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  turns: ChatTurn[];
  // Per conversation, so a follow-up in another conversation can't silently search
  // the documents this one was scoped to. Absent on conversations saved before it.
  scope?: ConversationScope;
}

export interface ConversationSummary {
  id: string;
  title: string;
  updatedAt: number;
  turnCount: number;
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
  pending: boolean;
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
  /** Drop names that are no longer indexed from every conversation's scope. */
  pruneScopes: (indexedFilenames: string[]) => void;
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
  setTurnFeedback: (turnId: string, rating: FeedbackRating) => void;
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

const CHAT_STORAGE_KEY = 'docrag-chat-history';
const CHAT_STORAGE_VERSION = 2;
// Cap persisted turns per conversation so a very long session doesn't grow
// localStorage unboundedly; older turns drop from persistence only, never from the
// in-memory session. Conversations beyond the cap are pruned oldest-updated first.
const CHAT_STORAGE_MAX_TURNS = 100;
const MAX_CONVERSATIONS = 20;
const TITLE_MAX_CHARS = 60;
// Each source carries its full chunk text (~2 KB), up to 8 per answer, and it was the
// bulk of what filled the ~5 MB quota. Only the most recent turns keep it on disk; older
// ones keep the citation (file, page, section, chunk id), which is what opens the viewer.
const PERSISTED_SOURCE_TEXT_TURNS = 6;

export interface ChatStorageV2 {
  version: 2;
  activeConversationId: string | null;
  conversations: Conversation[];
}

function isChatTurn(value: unknown): value is ChatTurn {
  if (!value || typeof value !== 'object') return false;
  const turn = value as Partial<ChatTurn>;
  return (
    typeof turn.id === 'string' &&
    (turn.role === 'user' || turn.role === 'assistant' || turn.role === 'error') &&
    typeof turn.content === 'string' &&
    Array.isArray(turn.sources) &&
    typeof turn.timestamp === 'number'
  );
}

function deriveTitle(turns: ChatTurn[]): string {
  const firstUser = turns.find((turn) => turn.role === 'user' && turn.content.trim());
  if (!firstUser) return 'New chat';
  const collapsed = firstUser.content.replace(/\s+/g, ' ').trim();
  return collapsed.slice(0, TITLE_MAX_CHARS) || 'New chat';
}

function emptyConversation(): Conversation {
  const now = Date.now();
  return { id: newId(), title: 'New chat', createdAt: now, updatedAt: now, turns: [] };
}

function byRecentUse(a: Conversation, b: Conversation): number {
  return b.updatedAt - a.updatedAt;
}

/** A conversation as written to disk: the last `maxTurns` turns, with source text kept
 * only on the most recent few (see PERSISTED_SOURCE_TEXT_TURNS). */
function compactForStorage(conversation: Conversation, maxTurns: number): Conversation {
  const turns = conversation.turns.slice(-maxTurns);
  const keepTextFrom = turns.length - PERSISTED_SOURCE_TEXT_TURNS;
  return {
    ...conversation,
    turns: turns.map((turn, index) =>
      index >= keepTextFrom || turn.sources.every((source) => !source.text)
        ? turn
        : { ...turn, sources: turn.sources.map((source) => ({ ...source, text: '' })) },
    ),
  };
}

type PersistOutcome = 'saved' | 'partial' | 'failed';

/** Write the store to localStorage, falling back to the active conversation alone. */
function isOnlyEmpty(storage: ChatStorageV2): boolean {
  return storage.conversations.length === 1 && storage.conversations[0].turns.length === 0;
}

function persistStorage(storage: ChatStorageV2): PersistOutcome {
  const { conversations, activeConversationId } = storage;
  if (isOnlyEmpty(storage)) {
    try {
      localStorage.removeItem(CHAT_STORAGE_KEY);
    } catch {
      // Nothing more to do if even a removeItem fails.
    }
    return 'saved';
  }
  const payloadFor = (convos: Conversation[], maxTurns: number) =>
    JSON.stringify({
      version: CHAT_STORAGE_VERSION,
      activeConversationId,
      conversations: convos.map((conversation) => compactForStorage(conversation, maxTurns)),
    });
  try {
    localStorage.setItem(CHAT_STORAGE_KEY, payloadFor(conversations, CHAT_STORAGE_MAX_TURNS));
    return 'saved';
  } catch {
    // Quota most likely exceeded. Retry once with only the active conversation and a
    // shorter tail rather than freezing at the last successful write with no signal —
    // this still drops older conversations from disk but keeps the current one saved.
    const reduced = conversations.filter((c) => c.id === activeConversationId);
    if (reduced.length === 0) {
      // No active match to fall back to. Writing the reduced payload here would
      // persist an empty conversations array, which loadStorage reads back as
      // "nothing saved" — a silent wipe reported as success. Fail loudly instead.
      return 'failed';
    }
    try {
      localStorage.setItem(CHAT_STORAGE_KEY, payloadFor(reduced, 20));
      // The write succeeded but every other conversation, and all but the last 20
      // turns of this one, are now absent from disk — its own signal, not a clean save.
      return 'partial';
    } catch {
      return 'failed';
    }
  }
}

/** ``incoming`` with passage text put back from ``mine`` where it was stripped.
 *
 * Another tab's localStorage copy drops old passage text; merged over this tab's full
 * copy as-is, the next IndexedDB save would have stored the stripped version. */
function keepPassageText(incoming: Conversation, mine: Conversation): Conversation {
  const mineById = new Map(mine.turns.map((turn) => [turn.id, turn]));
  let restored = false;
  const turns = incoming.turns.map((turn) => {
    const own = mineById.get(turn.id);
    if (!own || turn.sources.length !== own.sources.length) return turn;
    if (!turn.sources.some((source, index) => !source.text && own.sources[index].text)) return turn;
    restored = true;
    return {
      ...turn,
      sources: turn.sources.map((source, index) =>
        source.text ? source : { ...source, text: own.sources[index].text },
      ),
    };
  });
  return restored ? { ...incoming, turns } : incoming;
}

/** Fold another tab's saved conversations into this tab's, newest copy of each winning.
 *
 * Every tab used to rewrite the whole blob from its own memory, so asking in tab B
 * erased what tab A had just saved. A conversation deleted in the other tab survives
 * here if this tab still has it — losing a delete beats losing a conversation. */
export function mergeConversations(
  local: ChatStorageV2,
  incoming: Conversation[],
  // IndexedDB's copy of a conversation is the complete one (localStorage's drops old
  // passage text), so on equal timestamps it wins.
  preferIncomingOnTie = false,
): ChatStorageV2 {
  const byId = new Map(local.conversations.map((conversation) => [conversation.id, conversation]));
  let changed = false;
  for (const conversation of incoming) {
    const mine = byId.get(conversation.id);
    if (
      !mine ||
      conversation.updatedAt > mine.updatedAt ||
      (preferIncomingOnTie && conversation.updatedAt === mine.updatedAt)
    ) {
      byId.set(conversation.id, mine ? keepPassageText(conversation, mine) : conversation);
      changed = true;
    }
  }
  if (!changed) return local;
  const conversations = [...byId.values()].sort(byRecentUse).slice(0, MAX_CONVERSATIONS);
  const activeConversationId = conversations.some((c) => c.id === local.activeConversationId)
    ? local.activeConversationId
    : conversations[0].id;
  return { ...local, activeConversationId, conversations };
}

/** The conversations in a saved v2 payload (either store), or null when unreadable. */
function parseConversations(value: unknown): Conversation[] | null {
  if (!value || typeof value !== 'object') return null;
  const parsed = value as { version?: number; conversations?: unknown };
  if (parsed.version !== 2 || !Array.isArray(parsed.conversations)) return null;
  return parsed.conversations
    .map(sanitizeConversation)
    .filter((conversation): conversation is Conversation => conversation !== null);
}

function emptyStorage(): ChatStorageV2 {
  const conversation = emptyConversation();
  return { version: 2, activeConversationId: conversation.id, conversations: [conversation] };
}

function sanitizeConversation(value: unknown): Conversation | null {
  if (!value || typeof value !== 'object') return null;
  const raw = value as Partial<Conversation>;
  if (typeof raw.id !== 'string' || !Array.isArray(raw.turns)) return null;
  const turns = raw.turns.filter(isChatTurn);
  return {
    id: raw.id,
    title: typeof raw.title === 'string' && raw.title ? raw.title : deriveTitle(turns),
    createdAt: typeof raw.createdAt === 'number' ? raw.createdAt : Date.now(),
    updatedAt: typeof raw.updatedAt === 'number' ? raw.updatedAt : Date.now(),
    turns,
    scope: sanitizeScope(raw.scope),
  };
}

function sanitizeScope(value: unknown): ConversationScope | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const raw = value as Partial<ConversationScope>;
  const strings = (list: unknown) =>
    Array.isArray(list) ? list.filter((item): item is string => typeof item === 'string') : [];
  return { filenames: strings(raw.filenames), tags: strings(raw.tags) };
}

// Logs an unreadable or unrecognized-shape persisted value before it's replaced with an
// empty conversation, so the corruption is diagnosable from the console instead of
// vanishing silently. Deliberately does *not* copy the blob to a sibling storage key:
// nothing in the app ever reads such a copy back, so it would double the footprint of a
// value that may well have been what exceeded the quota, and pay that cost forever.
// Recovery would need a real UI to surface and discard it — a feature, not a side effect.
function reportUnreadableStorage(raw: string): void {
  console.warn(
    `Discarding unreadable chat history in localStorage["${CHAT_STORAGE_KEY}"] ` +
      `(${raw.length} chars); starting a fresh conversation.`,
  );
}

function loadStorage(): ChatStorageV2 {
  let raw: string | null = null;
  try {
    raw = localStorage.getItem(CHAT_STORAGE_KEY);
    if (!raw) return emptyStorage();
    const parsed = JSON.parse(raw) as {
      version?: number;
      turns?: unknown;
      conversations?: unknown;
      activeConversationId?: unknown;
    };

    if (parsed.version === 2 && Array.isArray(parsed.conversations)) {
      const conversations = parsed.conversations
        .map(sanitizeConversation)
        .filter((conversation): conversation is Conversation => conversation !== null);
      if (conversations.length === 0) return emptyStorage();
      const activeId =
        typeof parsed.activeConversationId === 'string' &&
        conversations.some((conversation) => conversation.id === parsed.activeConversationId)
          ? parsed.activeConversationId
          : conversations[0].id;
      return { version: 2, activeConversationId: activeId, conversations };
    }

    // v1 migration: a single flat thread becomes the first conversation. Read-only
    // until the first successful v2 write, so a crash mid-migration never loses v1.
    if (parsed.version === 1 && Array.isArray(parsed.turns)) {
      const turns = parsed.turns.filter(isChatTurn);
      if (turns.length === 0) return emptyStorage();
      const conversation: Conversation = {
        id: newId(),
        title: deriveTitle(turns),
        createdAt: turns[0].timestamp,
        updatedAt: turns[turns.length - 1].timestamp,
        turns,
      };
      return { version: 2, activeConversationId: conversation.id, conversations: [conversation] };
    }

    reportUnreadableStorage(raw);
    return emptyStorage();
  } catch {
    if (raw) reportUnreadableStorage(raw);
    return emptyStorage();
  }
}

export function useChat(): UseChatResult {
  const [storage, setStorage] = useState<ChatStorageV2>(loadStorage);
  const [pending, setPending] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
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
      if (incoming && incoming.length > 0) {
        setStorage((prev) => mergeConversations(prev, incoming, true));
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

  useEffect(() => {
    // Skip writes while a stream is in flight — the assistant turn's content mutates
    // on every delta, and persisting each one would thrash localStorage. `pagehide`
    // below covers a reload or navigation that lands mid-stream.
    if (pending) return;
    const outcome = persistStorage(storage);
    setPersistError(outcome === 'failed');
    setPersistPartial(outcome === 'partial');
    if (!chatStoreReady || !chatStoreAvailable()) return;
    let current = true;
    // Mirrors persistStorage: a lone empty conversation clears the store rather than
    // being saved, so it can't come back next to a new one after a reload.
    void (isOnlyEmpty(storage) ? clearChatStore() : saveChatStore(storage)).then((saved) => {
      // The full history is safe in IndexedDB: a localStorage quota miss loses nothing.
      if (saved && current) {
        setPersistError(false);
        setPersistPartial(false);
      }
    });
    return () => {
      current = false;
    };
  }, [storage, pending, chatStoreReady]);

  useEffect(() => {
    // A reload, a link followed in this tab, or a closed tab during a 1-2 minute answer
    // used to lose the question and everything streamed so far.
    // The answer being streamed is cut off by the unload, so it is saved as stopped:
    // saved as-is it read as finished after the reload, and went back as history.
    const flush = () => {
      const turnId = streamingTurnIdRef.current;
      const current = storageRef.current;
      const snapshot =
        turnId === null
          ? current
          : {
              ...current,
              conversations: current.conversations.map((conversation) => ({
                ...conversation,
                turns: conversation.turns.map((turn) =>
                  turn.id === turnId ? { ...turn, stopped: true } : turn,
                ),
              })),
            };
      persistStorage(snapshot);
      // Best effort: the page may be gone before this commits; localStorage has it.
      if (chatStoreAvailable()) void saveChatStore(snapshot);
    };
    window.addEventListener('pagehide', flush);
    return () => window.removeEventListener('pagehide', flush);
  }, []);

  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.key !== CHAT_STORAGE_KEY || !event.newValue) return;
      try {
        const incoming = parseConversations(JSON.parse(event.newValue));
        if (!incoming) return;
        setStorage((prev) => mergeConversations(prev, incoming));
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

  // Apply a turns-updater to the active conversation. `touch` bumps its updatedAt; a
  // streamed delta doesn't, so the conversation list (sorted by it) doesn't re-render
  // on every frame of an answer.
  const setActiveTurns = useCallback((update: (turns: ChatTurn[]) => ChatTurn[], touch = true) => {
    setStorage((prev) => {
      const targetId = prev.activeConversationId ?? prev.conversations[0]?.id ?? null;
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

    setActiveTurns((prev) => [...prev, makeTurn('user', question)]);
    setPending(true);
    // Seeded before the first server frame arrives so there is never a gap with a
    // pending request and no label at all.
    setStage(DEFAULT_STAGE_LABEL);

    // The assistant turn is created lazily on the first `sources`/`delta` event so a
    // mid-stream failure before any event arrives renders as a clean error turn.
    const assistantTurnId = newId();
    streamingTurnIdRef.current = assistantTurnId;
    const updateAssistantTurn = (update: (turn: ChatTurn) => ChatTurn, touch = true) => {
      setActiveTurns((prev) => {
        const existing = prev.find((turn) => turn.id === assistantTurnId);
        if (!existing) {
          return [...prev, update({ ...makeTurn('assistant', ''), id: assistantTurnId })];
        }
        return prev.map((turn) => (turn.id === assistantTurnId ? update(turn) : turn));
      }, touch);
    };

    // Deltas are coalesced into one state update per animation frame. Each update
    // re-renders the streaming message, whose markdown is re-parsed over the whole
    // answer so far — per token that is quadratic in answer length on a fast provider.
    let bufferedDelta = '';
    let frameId: number | null = null;
    const flushDelta = () => {
      if (frameId !== null) {
        cancelAnimationFrame(frameId);
        frameId = null;
      }
      if (!bufferedDelta) return;
      const text = bufferedDelta;
      bufferedDelta = '';
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
            bufferedDelta += text;
            if (frameId === null) frameId = requestAnimationFrame(flushDelta);
          },
          onDone: (answer, sources, timings, traceId, cached) => {
            flushDelta();
            updateAssistantTurn((turn) => ({
              ...turn,
              content: answer,
              sources,
              timings,
              traceId,
              ...(cached ? { cached: true } : {}),
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
        setActiveTurns((prev) =>
          prev
            .filter((turn) => turn.id !== assistantTurnId || turn.content !== '')
            .map((turn) => (turn.id === assistantTurnId ? { ...turn, stopped: true } : turn)),
        );
      } else {
        const message = err instanceof ApiClientError ? err.message : 'Something went wrong.';
        // Whatever streamed before the failure is a fragment. Left unmarked it read as a
        // finished answer: it offered Regenerate and feedback, and went back to the
        // model as history on the next question.
        // (An assistant turn with no text yet is kept as it was: it carries the trace
        // link to the partial server-side trace.)
        setActiveTurns((prev) => [
          ...prev.map((turn) =>
            turn.id === assistantTurnId && turn.content !== '' ? { ...turn, incomplete: true } : turn,
          ),
          { ...makeTurn('error', message), question },
        ]);
      }
    } finally {
      flushDelta();
      streamingTurnIdRef.current = null;
      inflightRef.current = null;
      setPending(false);
      setStage(null);
    }
    // Only refs and stable setters are closed over — see activeConversationRef above.
  }, [setActiveTurns]);

  const ask = useCallback(
    (
      question: string,
      provider?: LlmProvider,
      overrides?: QuestionOverrides,
      filenames?: string[],
      options?: QuestionOptions,
    ) =>
      runAsk(
        question,
        buildHistory(activeConversationRef.current?.turns ?? []),
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
        conversations: [fork, ...[...prev.conversations].sort(byRecentUse)].slice(
          0,
          MAX_CONVERSATIONS,
        ),
      }));
      // runAsk's updates target the active conversation through state updaters, which
      // run after the fork above — so the question lands in the fork.
      await runAsk(question, buildHistory(prior), provider, overrides, filenames, options);
    },
    [runAsk],
  );

  const importConversation = useCallback((raw: unknown, title?: string) => {
    const candidate = Array.isArray(raw) ? { turns: raw } : raw;
    if (!candidate || typeof candidate !== 'object') return false;
    const record = candidate as Partial<Conversation>;
    if (!Array.isArray(record.turns)) return false;
    // Fresh turn ids: importing the same file twice must not produce clashing keys.
    const turns = record.turns.filter(isChatTurn).map((turn) => ({ ...turn, id: newId() }));
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
      conversations: [conversation, ...[...prev.conversations].sort(byRecentUse)].slice(
        0,
        MAX_CONVERSATIONS,
      ),
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
              }
            : conversation,
        ),
      }));
    },
    [],
  );

  const keepInScopes = useCallback((keep: (filename: string) => boolean) => {
    setStorage((prev) => {
      let changed = false;
      const conversations = prev.conversations.map((conversation) => {
        const filenames = conversation.scope?.filenames ?? [];
        const kept = filenames.filter(keep);
        if (kept.length === filenames.length) return conversation;
        changed = true;
        return { ...conversation, scope: { tags: conversation.scope?.tags ?? [], filenames: kept } };
      });
      return changed ? { ...prev, conversations } : prev;
    });
  }, []);

  const pruneScopes = useCallback(
    (indexedFilenames: string[]) => {
      const indexed = new Set(indexedFilenames);
      keepInScopes((name) => indexed.has(name));
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

  const clear = useCallback(() => {
    if (pendingRef.current) return;
    setStorage((prev) => ({
      ...prev,
      conversations: prev.conversations.map((conversation) =>
        conversation.id === prev.activeConversationId
          ? // The title came from the first question, which is gone now.
            { ...conversation, turns: [], title: 'New chat', updatedAt: Date.now() }
          : conversation,
      ),
    }));
  }, []);

  const newConversation = useCallback(() => {
    if (pendingRef.current) return;
    setStorage((prev) => {
      // Reuse an already-empty active conversation instead of stacking blanks.
      const active = prev.conversations.find((c) => c.id === prev.activeConversationId);
      if (active && active.turns.length === 0) return prev;
      const conversation = emptyConversation();
      // Past the cap, the least recently *used* conversation goes — not the oldest
      // created, which dropped a conversation still in daily use.
      const conversations = [conversation, ...[...prev.conversations].sort(byRecentUse)].slice(
        0,
        MAX_CONVERSATIONS,
      );
      return { ...prev, activeConversationId: conversation.id, conversations };
    });
  }, []);

  const switchConversation = useCallback((id: string) => {
    if (pendingRef.current) return;
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
        conversation.id === id ? { ...conversation, title: trimmed } : conversation,
      ),
    }));
  }, []);

  const deleteConversation = useCallback((id: string) => {
    if (pendingRef.current) return;
    setStorage((prev) => {
      const remaining = prev.conversations.filter((conversation) => conversation.id !== id);
      if (remaining.length === 0) return emptyStorage();
      const activeId =
        prev.activeConversationId === id
          ? [...remaining].sort(byRecentUse)[0].id
          : prev.activeConversationId;
      return { ...prev, activeConversationId: activeId, conversations: remaining };
    });
  }, []);

  const setTurnFeedback = useCallback((turnId: string, rating: FeedbackRating) => {
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
    }));
    const prev = summariesRef.current;
    const same =
      prev.length === next.length &&
      prev.every(
        (summary, index) =>
          summary.id === next[index].id &&
          summary.title === next[index].title &&
          summary.updatedAt === next[index].updatedAt &&
          summary.turnCount === next[index].turnCount,
      );
    if (same) return prev;
    summariesRef.current = next;
    return next;
  }, [conversations]);

  return {
    turns,
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
    setTurnFeedback,
  };
}
