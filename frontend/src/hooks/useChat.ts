import { useEffect, useRef, useState } from 'react';
import { ApiClientError, api } from '../api/client';
import type { HistoryMessage, LlmProvider, QuestionOverrides } from '../api/types';
import type { ChatTurn } from '../components/ChatMessage';

// Mirrors the server's own REQUEST_HISTORY_MAX_MESSAGES cap (api/schemas.py) — kept
// in sync so a client-side trim never silently disagrees with what the server would
// accept anyway. The char budget is a client-only practical guard against crowding
// a small local model's context window; the server has its own count-based backstop.
const HISTORY_MAX_MESSAGES = 12;
const HISTORY_CHAR_BUDGET = 8000;

/** Build bounded conversation history from prior turns: only completed user/assistant
 * turns (error turns carry no useful context and are excluded), most recent first
 * up to HISTORY_MAX_MESSAGES, then trimmed to a total char budget (dropping the
 * oldest first), restored to chronological order. */
export function buildHistory(turns: ChatTurn[]): HistoryMessage[] {
  const candidates = turns
    .filter((turn): turn is ChatTurn & { role: 'user' | 'assistant' } =>
      (turn.role === 'user' || turn.role === 'assistant') && turn.content.trim().length > 0,
    )
    .slice(-HISTORY_MAX_MESSAGES);

  const withinBudget: HistoryMessage[] = [];
  let charsUsed = 0;
  for (let i = candidates.length - 1; i >= 0; i -= 1) {
    const turn = candidates[i];
    charsUsed += turn.content.length;
    if (charsUsed > HISTORY_CHAR_BUDGET && withinBudget.length > 0) break;
    withinBudget.unshift({ role: turn.role, content: turn.content });
  }
  return withinBudget;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  turns: ChatTurn[];
}

export interface ConversationSummary {
  id: string;
  title: string;
  updatedAt: number;
}

export interface UseChatResult {
  turns: ChatTurn[];
  pending: boolean;
  conversations: ConversationSummary[];
  activeConversationId: string | null;
  ask: (
    question: string,
    provider?: LlmProvider,
    overrides?: QuestionOverrides,
    filenames?: string[],
  ) => Promise<void>;
  cancel: () => void;
  clear: () => void;
  newConversation: () => void;
  switchConversation: (id: string) => void;
  renameConversation: (id: string, title: string) => void;
  deleteConversation: (id: string) => void;
}

function makeTurn(role: ChatTurn['role'], content: string, sources: ChatTurn['sources'] = []): ChatTurn {
  return {
    id: crypto.randomUUID(),
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

interface ChatStorageV2 {
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
  return { id: crypto.randomUUID(), title: 'New chat', createdAt: now, updatedAt: now, turns: [] };
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
  };
}

function loadStorage(): ChatStorageV2 {
  try {
    const raw = localStorage.getItem(CHAT_STORAGE_KEY);
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
        id: crypto.randomUUID(),
        title: deriveTitle(turns),
        createdAt: turns[0].timestamp,
        updatedAt: turns[turns.length - 1].timestamp,
        turns,
      };
      return { version: 2, activeConversationId: conversation.id, conversations: [conversation] };
    }

    return emptyStorage();
  } catch {
    return emptyStorage();
  }
}

export function useChat(): UseChatResult {
  const [storage, setStorage] = useState<ChatStorageV2>(loadStorage);
  const [pending, setPending] = useState(false);

  const { conversations, activeConversationId } = storage;
  const activeConversation =
    conversations.find((conversation) => conversation.id === activeConversationId) ??
    conversations[0];
  const turns = activeConversation?.turns ?? [];

  useEffect(() => {
    // Skip writes while a stream is in flight — the assistant turn's content mutates
    // on every delta, and persisting each one would thrash localStorage.
    if (pending) return;
    try {
      const onlyEmpty =
        conversations.length === 1 && conversations[0].turns.length === 0;
      if (onlyEmpty) {
        localStorage.removeItem(CHAT_STORAGE_KEY);
      } else {
        localStorage.setItem(
          CHAT_STORAGE_KEY,
          JSON.stringify({
            version: CHAT_STORAGE_VERSION,
            activeConversationId,
            conversations: conversations.map((conversation) => ({
              ...conversation,
              turns: conversation.turns.slice(-CHAT_STORAGE_MAX_TURNS),
            })),
          }),
        );
      }
    } catch {
      // Persistence is best-effort; the in-session state above is unaffected.
    }
  }, [storage, pending, conversations, activeConversationId]);

  // Tracks the in-flight question's controller: doubles as a reentrancy guard and lets
  // unmount cancel a hung request.
  const inflightRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      inflightRef.current?.abort();
    };
  }, []);

  // Apply a turns-updater to the active conversation, bumping its updatedAt.
  const setActiveTurns = (update: (turns: ChatTurn[]) => ChatTurn[]) => {
    setStorage((prev) => {
      const targetId = prev.activeConversationId ?? prev.conversations[0]?.id ?? null;
      return {
        ...prev,
        conversations: prev.conversations.map((conversation) =>
          conversation.id === targetId
            ? {
                ...conversation,
                turns: update(conversation.turns),
                title:
                  conversation.title === 'New chat'
                    ? deriveTitle(update(conversation.turns))
                    : conversation.title,
                updatedAt: Date.now(),
              }
            : conversation,
        ),
      };
    });
  };

  const ask = async (
    question: string,
    provider?: LlmProvider,
    overrides?: QuestionOverrides,
    filenames?: string[],
  ) => {
    if (inflightRef.current) return;
    const controller = new AbortController();
    inflightRef.current = controller;

    const history = buildHistory(activeConversation?.turns ?? []);

    setActiveTurns((prev) => [...prev, makeTurn('user', question)]);
    setPending(true);

    // The assistant turn is created lazily on the first `sources`/`delta` event so a
    // mid-stream failure before any event arrives renders as a clean error turn.
    const assistantTurnId = crypto.randomUUID();
    const updateAssistantTurn = (update: (turn: ChatTurn) => ChatTurn) => {
      setActiveTurns((prev) => {
        const existing = prev.find((turn) => turn.id === assistantTurnId);
        if (!existing) {
          return [...prev, update({ ...makeTurn('assistant', ''), id: assistantTurnId })];
        }
        return prev.map((turn) => (turn.id === assistantTurnId ? update(turn) : turn));
      });
    };

    try {
      await api.askQuestionStream(
        question,
        provider,
        overrides,
        filenames,
        history,
        {
          onSources: (sources, traceId) => {
            updateAssistantTurn((turn) => ({ ...turn, sources, traceId }));
          },
          onDelta: (text) => {
            updateAssistantTurn((turn) => ({ ...turn, content: turn.content + text }));
          },
          onDone: (answer, sources, timings, traceId) => {
            updateAssistantTurn((turn) => ({ ...turn, content: answer, sources, timings, traceId }));
          },
        },
        controller.signal,
      );
    } catch (err) {
      const message = err instanceof ApiClientError ? err.message : 'Something went wrong.';
      setActiveTurns((prev) => [...prev, { ...makeTurn('error', message), question }]);
    } finally {
      inflightRef.current = null;
      setPending(false);
    }
  };

  const cancel = () => {
    inflightRef.current?.abort();
  };

  const clear = () => {
    if (pending) return;
    setActiveTurns(() => []);
  };

  const newConversation = () => {
    if (pending) return;
    setStorage((prev) => {
      // Reuse an already-empty active conversation instead of stacking blanks.
      const active = prev.conversations.find((c) => c.id === prev.activeConversationId);
      if (active && active.turns.length === 0) return prev;
      const conversation = emptyConversation();
      const conversations = [conversation, ...prev.conversations].slice(0, MAX_CONVERSATIONS);
      return { ...prev, activeConversationId: conversation.id, conversations };
    });
  };

  const switchConversation = (id: string) => {
    if (pending) return;
    setStorage((prev) =>
      prev.conversations.some((conversation) => conversation.id === id)
        ? { ...prev, activeConversationId: id }
        : prev,
    );
  };

  const renameConversation = (id: string, title: string) => {
    const trimmed = title.replace(/\s+/g, ' ').trim().slice(0, TITLE_MAX_CHARS);
    if (!trimmed) return;
    setStorage((prev) => ({
      ...prev,
      conversations: prev.conversations.map((conversation) =>
        conversation.id === id ? { ...conversation, title: trimmed } : conversation,
      ),
    }));
  };

  const deleteConversation = (id: string) => {
    if (pending) return;
    setStorage((prev) => {
      const remaining = prev.conversations.filter((conversation) => conversation.id !== id);
      if (remaining.length === 0) return emptyStorage();
      const activeId =
        prev.activeConversationId === id ? remaining[0].id : prev.activeConversationId;
      return { ...prev, activeConversationId: activeId, conversations: remaining };
    });
  };

  const summaries: ConversationSummary[] = [...conversations]
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .map((conversation) => ({
      id: conversation.id,
      title: conversation.title,
      updatedAt: conversation.updatedAt,
    }));

  return {
    turns,
    pending,
    conversations: summaries,
    activeConversationId: activeConversation?.id ?? null,
    ask,
    cancel,
    clear,
    newConversation,
    switchConversation,
    renameConversation,
    deleteConversation,
  };
}
