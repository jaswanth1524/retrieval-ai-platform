import { useEffect, useRef, useState } from 'react';
import type { Dispatch, MutableRefObject, SetStateAction } from 'react';
import { chatStoreAvailable, clearChatStore, readChatStore, saveChatStore } from './chatStore';
import {
  CHAT_STORAGE_KEY,
  LOCAL_STORAGE_MAX_CONVERSATIONS,
  forChatStore,
  isOnlyEmpty,
  mergeConversations,
  parseConversations,
  parseDeleted,
  persistStorage,
} from './chatPersistence';
import type { ChatStorageV2, Conversation } from './chatPersistence';

// How long IndexedDB writes are held back so a burst of changes is written once.
const CHAT_STORE_SAVE_DELAY_MS = 300;

/** The streaming state ``useChat`` owns that saving reads: the answer being streamed,
 *  where it goes, and its text not yet applied. */
export interface StreamingRefs {
  storageRef: MutableRefObject<ChatStorageV2>;
  streamingTurnIdRef: MutableRefObject<string | null>;
  streamingConversationIdRef: MutableRefObject<string | null>;
  bufferedDeltaRef: MutableRefObject<string>;
}

/** Saving and restoring the chat history (``useChat``'s storage): localStorage's
 *  compact copy on every change, IndexedDB's full one coalesced (and never before it
 *  was read), a snapshot of an answer cut off by a reload, and other tabs' changes
 *  merged in. */
export function useChatPersistence(
  storage: ChatStorageV2,
  setStorage: Dispatch<SetStateAction<ChatStorageV2>>,
  pending: boolean,
  { storageRef, streamingTurnIdRef, streamingConversationIdRef, bufferedDeltaRef }: StreamingRefs,
): { persistError: boolean; persistPartial: boolean; historyReady: boolean } {
  // True when the last persist attempt failed even after the reduced-payload retry
  // below — the session is memory-only from that point on until a write succeeds.
  const [persistError, setPersistError] = useState(false);
  // True when the reduced-payload retry *succeeded*: the active conversation is saved
  // but the others aren't. Distinct from persistError because "some of this survives a
  // reload" and "none of this survives a reload" need different warnings.
  const [persistPartial, setPersistPartial] = useState(false);
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
    void (
      isOnlyEmpty(pendingStorage) ? clearChatStore() : saveChatStore(forChatStore(pendingStorage))
    ).then(
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
  // Set once the stored history was read (or found empty). After a failed read nothing
  // is written there this session: the full history may still be in it, and the next
  // save would have replaced it with localStorage's 20 most recent conversations.
  const chatStoreWritableRef = useRef(false);
  const [chatStoreUnreadable, setChatStoreUnreadable] = useState(false);

  useEffect(() => {
    if (chatStoreReady) return;
    let cancelled = false;
    void readChatStore().then((read) => {
      if (cancelled) return;
      if (read.status === 'failed') {
        setChatStoreUnreadable(true);
        setChatStoreReady(true);
        return;
      }
      const raw = read.status === 'data' ? read.value : null;
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
      chatStoreWritableRef.current = true;
      setChatStoreReady(true);
    });
    return () => {
      cancelled = true;
    };
    // Once, on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
    // Not before the stored history was read: this would replace it with this copy.
    if (chatStoreAvailable() && chatStoreWritableRef.current) {
      void saveChatStore(forChatStore(snapshot));
    }
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
    // Without IndexedDB, localStorage's copy is all there is, and it holds only the
    // most recent conversations.
    setPersistPartial(
      outcome === 'partial' ||
        chatStoreUnreadable ||
        (outcome === 'saved' &&
          !chatStoreAvailable() &&
          storage.conversations.length > LOCAL_STORAGE_MAX_CONVERSATIONS),
    );
    if (!chatStoreReady || !chatStoreAvailable() || !chatStoreWritableRef.current) return;
    // Coalesced: every scope click, rename or rating used to rewrite the whole history
    // in IndexedDB at once; a burst of changes now costs one write. The timer is not
    // cancelled on unmount (the write must still land), and a hidden tab or pagehide
    // flushes it early.
    pendingChatStoreRef.current = storage;
    window.clearTimeout(chatStoreTimerRef.current);
    chatStoreTimerRef.current = window.setTimeout(flushChatStore, CHAT_STORE_SAVE_DELAY_MS);
    // persistStreamSnapshot and flushChatStore only read refs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storage, pending, chatStoreReady, chatStoreUnreadable]);

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
    // Both stable (a state setter and a ref): registered once, as before.
  }, [setStorage, streamingConversationIdRef]);

  return {
    persistError,
    persistPartial,
    // Not after a failed read: what is loaded is only localStorage's partial copy, and
    // App prunes drafts against it (a draft of a conversation only IndexedDB holds).
    historyReady: chatStoreReady && !chatStoreUnreadable,
  };
}
