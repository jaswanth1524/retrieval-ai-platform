import { lazy, startTransition, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ApiClientError, api, apiErrorMessage } from './api/client';
import type {
  AccessLevel,
  ApiStatus,
  CitationResponse,
  FeedbackItemResponse,
  LlmProvider,
  PublicConfigResponse,
  QuestionOverrides,
  TraceSummaryResponse,
} from './api/types';
import ChatHeader from './components/ChatHeader';
import type { ChatMode } from './components/ChatHeader';
import ChatThread from './components/ChatThread';
import type { FeedbackPayload } from './components/ChatMessage';
import Composer from './components/Composer';
import { NEW_CONVERSATION_DRAFT, pruneDrafts } from './utils/drafts';
import { QUESTION_INPUT_ID } from './utils/id';
import ContextPanel from './components/ContextPanel';
import ConversationList from './components/ConversationList';
import CorpusPanel, { type DocumentMeta } from './components/CorpusPanel';
import IconRail from './components/IconRail';
import LazyChunkBoundary from './components/LazyChunkBoundary';
import { loadMarkdown } from './components/markdownLoader';
import type { RailPanel } from './components/IconRail';
import type { InspectorTab } from './components/Inspector';
import SourcePreview from './components/SourcePreview';
import ToastRow from './components/ToastRow';
import SearchPanel from './components/SearchPanel';
import TracesPanel from './components/TracesPanel';
import FeedbackPanel from './components/FeedbackPanel';
import { MAX_PINNED_CONVERSATIONS } from './hooks/chatPersistence';
import { useChat } from './hooks/useChat';
import { useStableHandler } from './hooks/useStableHandler';
import { prefetchCitationContent } from './utils/contentPrefetch';
import { useCommands } from './hooks/useCommands';
import { useCorpus } from './hooks/useCorpus';
import { useResponsiveLayout } from './hooks/useResponsiveLayout';
import { useToasts } from './hooks/useToasts';
import { useWindowFileDrop } from './hooks/useWindowFileDrop';
import { NOTIFY_STORAGE_KEY, useBackgroundNotice } from './hooks/useBackgroundNotice';
import { chatToJson, chatToMarkdown, downloadFile } from './utils/exportChat';
import { EMPTY_OVERRIDES } from './utils/overrides';
import { initialProvider } from './utils/providers';
import { readStored, readStoredJson, writeStored } from './utils/safeStorage';
import {
  acceptedExtensions,
  DEFAULT_MAX_UPLOAD_BYTES,
  validateUploads,
} from './utils/uploadValidation';
import { sliceWhole } from './utils/text';

// Dialogs and the inspector render only on demand, so they load on demand too: the
// first paint (thread + composer) no longer waits on their code.
const CommandPalette = lazy(() => import('./components/CommandPalette'));
const DocumentViewer = lazy(() => import('./components/DocumentViewer'));
const Inspector = lazy(() => import('./components/Inspector'));
const SettingsModal = lazy(() => import('./components/SettingsModal'));

const ADVANCED_OPTIONS_STORAGE_KEY = 'docrag-advanced-options';
const MODE_STORAGE_KEY = 'docrag-mode';
const PROVIDER_STORAGE_KEY = 'docrag-provider';
// Also read by main.tsx before the first render.
const THEME_STORAGE_KEY = 'docrag-theme';

function loadPersistedOverrides(): QuestionOverrides {
  // Storage denied/corrupt — fall back to defaults for this session.
  const parsed = readStoredJson<Partial<QuestionOverrides>>(ADVANCED_OPTIONS_STORAGE_KEY);
  if (!parsed) return EMPTY_OVERRIDES;
  return {
    rerankTopK: parsed.rerankTopK ?? null,
    maxContextChunks: parsed.maxContextChunks ?? null,
    llmTemperature: parsed.llmTemperature ?? null,
  };
}

/** Drop saved overrides the server would now reject.
 *
 * Overrides are saved in the browser, limits live on the server: an operator lowering
 * FUSED_TOP_N below a saved "Rerank top K" made every question fail with a 422 — and
 * the slider, clamped by the browser to the new max, couldn't show why. */
function clampOverrides(
  overrides: QuestionOverrides,
  config: PublicConfigResponse,
): QuestionOverrides {
  const within = (value: number | null, max: number, min = 1) =>
    value !== null && (value < min || value > max) ? null : value;
  const rerankTopK = within(overrides.rerankTopK, config.rerank_top_k_limit);
  const maxContextChunks = within(overrides.maxContextChunks, config.max_context_chunks_limit);
  return {
    rerankTopK,
    // More context chunks than reranked ones is a 422 on every question (the server's
    // QuestionRequest rule), and the values were saved, so it outlived a reload.
    maxContextChunks:
      rerankTopK !== null && maxContextChunks !== null
        ? Math.min(maxContextChunks, rerankTopK)
        : maxContextChunks,
    llmTemperature: within(overrides.llmTemperature, config.llm_temperature_max, 0),
  };
}

function loadPersistedMode(): ChatMode {
  return readStored(MODE_STORAGE_KEY) === 'engineer' ? 'engineer' : 'reader';
}

// The browser's own PDF viewer shows an original in the document viewer; past this a
// download serves better than holding the whole file in memory as a blob.
const MAX_PREVIEW_BYTES = 50 * 1024 * 1024;

// How long a pointer or focus rests on a citation before its viewer page is fetched.
const CITATION_PREFETCH_DWELL_MS = 150;
// /health while the page is up: how often, and how often once it has failed.
const HEALTH_POLL_MS = 30_000;
const HEALTH_RETRY_MS = 5_000;
// A failed document listing is retried from this delay, doubling up to BOOT_RETRY_MAX_MS.
const LISTING_RETRY_BASE_MS = 5000;
// Backoff between boot attempts while the API is unreachable: 2s, 4s, 8s ... 30s.
const BOOT_RETRY_BASE_MS = 2000;
const BOOT_RETRY_MAX_MS = 30_000;

function canPreviewOriginal(
  filename: string,
  originalsStored: boolean,
  meta: DocumentMeta | undefined,
): boolean {
  return (
    originalsStored &&
    filename.toLowerCase().endsWith('.pdf') &&
    Boolean(meta?.reindexable) &&
    (meta?.bytes ?? 0) <= MAX_PREVIEW_BYTES
  );
}

// Shared empty values: a fresh [] per render would re-render the memoized child.
const NO_HIDDEN_PANELS: RailPanel[] = [];
const HIDDEN_FEEDBACK_PANEL: RailPanel[] = ['feedback'];
const NO_SOURCES: CitationResponse[] = [];

const PANEL_META: Record<RailPanel, { title: string; actionLabel: string }> = {
  chat: { title: 'Conversations', actionLabel: 'New' },
  corpus: { title: 'Corpus', actionLabel: 'Upload' },
  search: { title: 'Search passages', actionLabel: 'Clear' },
  traces: { title: 'Traces', actionLabel: 'Refresh' },
  feedback: { title: 'Feedback', actionLabel: 'Refresh' },
};

function scopeLabel(
  indexedFilenames: string[],
  selectedFilenames: string[],
  selectedTags: string[] = [],
): string {
  const tagPart = selectedTags.length > 0 ? ` · #${selectedTags.join(' #')}` : '';
  if (selectedFilenames.length === 0) {
    return selectedTags.length > 0
      ? `Tagged${tagPart}`
      : `All ${indexedFilenames.length} documents`;
  }
  if (selectedFilenames.length === 1) return `${selectedFilenames[0]}${tagPart}`;
  return `${selectedFilenames.length} documents${tagPart}`;
}

function App() {
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking');
  // The first corpus listing has come back (or failed): until then "no documents"
  // only means "not loaded yet", and must not tell anyone to upload one.
  const [corpusChecked, setCorpusChecked] = useState(false);
  // The first listing failed with something other than a 401 (a 500, a timeout on a
  // large corpus): unknown, not empty. It used to read as "no documents" and disabled
  // the composer under "Upload a document to start." until a reload.
  const [listingFailed, setListingFailed] = useState(false);
  const [config, setConfig] = useState<PublicConfigResponse | null>(null);
  const [selectedProvider, setSelectedProvider] = useState<LlmProvider>('ollama');
  const [advancedOptions, setAdvancedOptions] = useState<QuestionOverrides>(loadPersistedOverrides);
  const [theme, setTheme] = useState<'dark' | 'light'>(
    () => (document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'),
  );
  const [mode, setModeState] = useState<ChatMode>(loadPersistedMode);
  const [rail, setRail] = useState<RailPanel>('chat');
  const [searchOpened, setSearchOpened] = useState(false);
  if (rail === 'search' && !searchOpened) setSearchOpened(true);
  const [searchResetKey, setSearchResetKey] = useState(0);
  const [traces, setTraces] = useState<TraceSummaryResponse[] | null>(null);
  const [tracesError, setTracesError] = useState<string | null>(null);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  // Set once the user opens or closes the inspector themselves. The width-driven
  // auto-collapse must not override a deliberate choice — otherwise crossing a
  // breakpoint silently undoes what the user just did.
  const [inspectorForced, setInspectorForced] = useState(false);
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>('sources');
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const {
    toasts,
    push: pushToast,
    dismiss: dismissToast,
    hold: holdToast,
    release: releaseToast,
  } = useToasts();
  const [exportingBackup, setExportingBackup] = useState(false);
  // Optimistic until /access answers: a full-key (or keyless) server is the common case.
  const [access, setAccess] = useState<AccessLevel>('full');
  const readOnly = access === 'read';
  const [restoring, setRestoring] = useState(false);
  const [notifyOnAnswer, setNotifyOnAnswer] = useState(
    () => readStored(NOTIFY_STORAGE_KEY) === 'true',
  );
  const restoreInputRef = useRef<HTMLInputElement>(null);
  const exportingRef = useRef(false);
  const { tooNarrowForInspector, roomyEnoughForInspector, panelCollapsed } = useResponsiveLayout();
  // Narrow screens only: whether the context panel's drawer is showing. On wider screens
  // the panel is a grid column and this is ignored.
  const [panelDrawerOpen, setPanelDrawerOpen] = useState(false);
  // Which turn the inspector is describing. null follows the newest assistant turn,
  // so the panel keeps up with the conversation on its own; clicking a trace row pins
  // it to that specific turn instead.
  const [pinnedTraceId, setPinnedTraceId] = useState<string | null>(null);
  const [hoveredCitation, setHoveredCitation] = useState<CitationResponse | null>(null);
  // Source document viewer: which document/chunk a clicked citation opens (null = closed).
  // chunkId null: opened from the corpus panel's Read, at the document's first passage.
  // highlights: the passages a citation rests on, marked in the cited chunk.
  const [sourceView, setSourceView] = useState<{
    filename: string;
    chunkId: string | null;
    highlights?: string[];
  } | null>(null);
  const corpusBrowseInputRef = useRef<HTMLInputElement>(null);
  const {
    turns,
    pending,
    streamingConversationId,
    askedTurns,
    stage,
    ask,
    cancel,
    clear: clearTurns,
    conversations,
    activeConversationId,
    newConversation: createConversation,
    switchConversation: selectConversation,
    renameConversation,
    deleteConversation: removeConversation,
    setPinned,
    searchConversations,
    persistError,
    persistPartial,
    setTurnFeedback,
    scope,
    setScope,
    pruneScopes,
    removeFromScopes,
    editAndResend,
    importConversation,
    historyReady,
  } = useChat();
  // A deleted conversation's unsent draft goes with it, whichever tab deleted it. After
  // the composer has saved it on the switch away, hence an effect on the list. Not
  // before the full history has loaded: a conversation only IndexedDB holds would
  // look deleted, and lose its draft.
  useEffect(() => {
    if (!historyReady) return;
    pruneDrafts(new Set(conversations.map((conversation) => conversation.id)));
  }, [conversations, historyReady]);
  // An answer can stream into one conversation while another is open. The open one shows
  // its progress (and offers Stop) only when it is the one being answered.
  const streamingHere = pending && streamingConversationId === activeConversationId;
  const {
    uploads,
    indexedFilenames,
    documents,
    availableTags,
    chunkTotal,
    refreshDocuments,
    handleUpload,
    handleRetryUpload,
    handleDismissUpload,
    handleCancelUpload,
    handleReindex,
    handleReindexMany,
    handleReindexAllStale,
    handleDeleteDocument,
    handleSetTags,
    trackJobs,
  } = useCorpus({
    onAuthFailure: (err) => noteAuthFailure(err),
    onAuthRestored: () => setApiStatus((prev) => (prev === 'unauthorized' ? 'ok' : prev)),
    pushToast,
    pruneScopes,
    removeFromScopes,
  });
  // The search scope belongs to the active conversation (see useChat).
  const selectedFilenames = scope.filenames;
  const selectedTags = scope.tags;
  const setSelectedFilenames = useCallback(
    (filenames: string[]) => setScope((prev) => ({ ...prev, filenames })),
    [setScope],
  );
  const setSelectedTags = useCallback(
    (tags: string[]) => setScope((prev) => ({ ...prev, tags })),
    [setScope],
  );

  // A pinned trace belongs to one conversation's turns, so anything that replaces the
  // turns array has to release it. Otherwise inspectedTurn's lookup finds nothing, the
  // Retrieval/Trace tabs fall back to the stale pinnedTraceId and keep rendering the
  // previous conversation's trace, while the Sources tab (which reads the live turn)
  // correctly goes empty — one inspector showing two different questions. The hovered
  // citation goes too: its button unmounts without a mouseleave/blur (a switch from the
  // keyboard or palette), which left its preview on screen over the next conversation.
  // Stable (useChat's actions are), so the memoized conversation list skips re-rendering
  // on every streamed token.
  const clear = useCallback(() => {
    setPinnedTraceId(null);
    setHoveredCitation(null);
    clearTurns();
  }, [clearTurns]);
  const newConversation = useCallback(() => {
    setPinnedTraceId(null);
    setHoveredCitation(null);
    createConversation();
  }, [createConversation]);
  const switchConversation = useCallback(
    (id: string) => {
      setPinnedTraceId(null);
      setHoveredCitation(null);
      // A transition: a long conversation mounts up to 100 rendered answers, and the
      // list's highlight and the drawer closing shouldn't wait for all of them.
      startTransition(() => selectConversation(id));
    },
    [selectConversation],
  );
  const deleteConversation = useCallback(
    (id: string) => {
      setPinnedTraceId(null);
      setHoveredCitation(null);
      removeConversation(id);
    },
    [removeConversation],
  );
  // A conversation search hit names the turn it matched: the thread scrolls to it.
  const [focusTurn, setFocusTurn] = useState<{ turnId: string } | null>(null);
  const switchConversationFromList = useCallback(
    (id: string, turnId?: string) => {
      switchConversation(id);
      setFocusTurn(turnId ? { turnId } : null);
      setPanelDrawerOpen(false);
    },
    [switchConversation],
  );
  const pinConversation = useCallback(
    (id: string, pinned: boolean) => {
      if (!setPinned(id, pinned)) {
        pushToast({
          tone: 'warn',
          title: 'Too many pinned conversations',
          body: `Unpin one first — at most ${MAX_PINNED_CONVERSATIONS} stay pinned.`,
        });
      }
    },
    [setPinned, pushToast],
  );

  const updateAdvancedOptions = (next: QuestionOverrides) => {
    setAdvancedOptions(next);
    writeStored(ADVANCED_OPTIONS_STORAGE_KEY, JSON.stringify(next));
  };

  const setMode = useCallback((next: ChatMode) => {
    setModeState(next);
    writeStored(MODE_STORAGE_KEY, next);
  }, []);

  // Until a theme is picked the page follows the OS — live, not only at load: switching
  // the OS to dark at sunset left an open tab light. A pick made in another tab applies
  // here too (each tab used to keep its own).
  useEffect(() => {
    const apply = (next: 'dark' | 'light') => {
      setTheme(next);
      document.documentElement.dataset.theme = next;
    };
    const media = window.matchMedia?.('(prefers-color-scheme: light)');
    const onSystemChange = (event: MediaQueryListEvent) => {
      if (readStored(THEME_STORAGE_KEY) === null) apply(event.matches ? 'light' : 'dark');
    };
    const onStorage = (event: StorageEvent) => {
      if (event.key !== THEME_STORAGE_KEY) return;
      if (event.newValue === 'light' || event.newValue === 'dark') apply(event.newValue);
      else if (event.newValue === null && media) apply(media.matches ? 'light' : 'dark');
    };
    media?.addEventListener?.('change', onSystemChange);
    window.addEventListener('storage', onStorage);
    return () => {
      media?.removeEventListener?.('change', onSystemChange);
      window.removeEventListener('storage', onStorage);
    };
  }, []);

  const toggleTheme = useStableHandler(() => {
    const next = theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    document.documentElement.dataset.theme = next;
    writeStored(THEME_STORAGE_KEY, next);
  });

  /** Note a 401 so it isn't mistaken for an empty corpus.
   *
   * /health and /config are unauthenticated, so a server with API_KEY set answers both
   * and the app looks healthy — while every /documents call 401s. Swallowing that left
   * indexedFilenames empty, which disabled the composer under the hint "Upload a
   * document to start.", advice that would itself have 401'd. Returns true when it
   * handled the error, so callers keep their own fallback for everything else.
   */
  const noteAuthFailure = useStableHandler((err: unknown): boolean => {
    if (err instanceof ApiClientError && err.statusCode === 401) {
      setApiStatus('unauthorized');
      return true;
    }
    // A 403 means a read-only key (API_READ_KEY): hide what it can't do from now on.
    // The error itself is still shown by the caller.
    if (err instanceof ApiClientError && err.statusCode === 403) setAccess('read');
    return false;
  });

  /** Ask the server which key this browser holds. An older server without /access
   *  (404) is treated as full access; a 403 on any write corrects that later. */
  const refreshAccess = async (signal?: AbortSignal) => {
    try {
      setAccess((await api.access(signal)).access);
    } catch (err) {
      if (err instanceof ApiClientError && err.statusCode === 404) setAccess('full');
      else noteAuthFailure(err);
    }
  };

  // Bumped to boot again: by the backoff below after a failed /config, or by the
  // banner's "Retry now". The API may just be starting (compose boots it after Qdrant),
  // and a one-shot boot left the page on "start the API and refresh" for good.
  const [bootAttempt, setBootAttempt] = useState(0);
  const retryBoot = useCallback(() => setBootAttempt((attempt) => attempt + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;

    // /config, /access and the corpus listing go out together: each round trip used to
    // wait for the one before it (a /health probe first), and the composer said "Upload
    // a document" until the listing, the last of them, came back.
    async function boot() {
      const listing = refreshDocuments(controller.signal)
        .then(() => {
          if (!cancelled) setListingFailed(false);
        })
        .catch((err) => {
          // Anything other than a 401 leaves the corpus unknown; the effect below
          // retries the listing until it answers.
          if (!cancelled && !noteAuthFailure(err)) setListingFailed(true);
        })
        .finally(() => {
          if (!cancelled) setCorpusChecked(true);
        });
      void refreshAccess(controller.signal);
      try {
        const configResult = await api.config(controller.signal);
        if (cancelled) return;
        setConfig(configResult);
        setAdvancedOptions((prev) => {
          const clamped = clampOverrides(prev, configResult);
          const changed =
            clamped.rerankTopK !== prev.rerankTopK ||
            clamped.maxContextChunks !== prev.maxContextChunks ||
            clamped.llmTemperature !== prev.llmTemperature;
          if (!changed) return prev;
          writeStored(ADVANCED_OPTIONS_STORAGE_KEY, JSON.stringify(clamped));
          return clamped;
        });
        // A provider picked earlier in this browser wins while it can answer; otherwise
        // the server's default, then any other that can (never stuck on an Ollama that
        // is down while OpenAI is configured). The remembered choice stays stored.
        const startOn = initialProvider(readStored(PROVIDER_STORAGE_KEY), configResult);
        if (startOn) setSelectedProvider(startOn);
        // The listing or /access may have answered 401 first; that stands.
        setApiStatus((prev) => (prev === 'unauthorized' ? prev : 'ok'));
      } catch {
        if (cancelled) return;
        setApiStatus('error');
        retryTimer = setTimeout(
          () => setBootAttempt((attempt) => attempt + 1),
          Math.min(BOOT_RETRY_BASE_MS * 2 ** bootAttempt, BOOT_RETRY_MAX_MS),
        );
      }
      await listing;
    }

    void boot();
    return () => {
      cancelled = true;
      clearTimeout(retryTimer);
      // Aborting releases the in-flight request instead of just ignoring its result.
      controller.abort();
    };
    // Boots once, and again per retry. refreshDocuments reads only setters and useChat's
    // stable pruneScopes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bootAttempt]);

  // The markdown renderer is its own chunk; fetch it once the first paint is done, so
  // it is there before the first answer is.
  useEffect(() => {
    const preload = () => void loadMarkdown();
    if (typeof window.requestIdleCallback === 'function') {
      const handle = window.requestIdleCallback(preload, { timeout: 2000 });
      return () => window.cancelIdleCallback(handle);
    }
    const timer = setTimeout(preload, 200);
    return () => clearTimeout(timer);
  }, []);

  const [feedbackItems, setFeedbackItems] = useState<FeedbackItemResponse[] | null>(null);
  const [feedbackError, setFeedbackError] = useState<string | null>(null);
  // One request per list at a time: a Refresh aborts the load still in flight, whose
  // late answer used to overwrite the newer one.
  const feedbackRequestRef = useRef<AbortController | null>(null);
  const loadFeedback = () => {
    feedbackRequestRef.current?.abort();
    const controller = new AbortController();
    feedbackRequestRef.current = controller;
    const { signal } = controller;
    setFeedbackItems(null);
    setFeedbackError(null);
    api
      .listFeedback(200, signal)
      .then((response) => {
        if (!signal.aborted) setFeedbackItems(response.feedback);
      })
      .catch((err) => {
        if (signal.aborted) return;
        noteAuthFailure(err);
        setFeedbackError(apiErrorMessage(err, 'Could not load feedback.'));
      });
  };

  useEffect(() => {
    if (rail !== 'feedback') return;
    loadFeedback();
    return () => feedbackRequestRef.current?.abort();
    // loadFeedback only calls setters, refs and the API.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rail]);

  // The feedback panel needs feedback on and the full key (GET /feedback is operator data).
  const feedbackPanelAvailable = Boolean(config?.feedback_enabled) && !readOnly;
  useEffect(() => {
    if (rail === 'feedback' && !feedbackPanelAvailable) setRail('chat');
  }, [rail, feedbackPanelAvailable]);

  const tracesRequestRef = useRef<AbortController | null>(null);
  const loadTraces = () => {
    tracesRequestRef.current?.abort();
    const controller = new AbortController();
    tracesRequestRef.current = controller;
    const { signal } = controller;
    setTraces(null);
    setTracesError(null);
    api
      .listTraces(signal)
      .then((response) => {
        if (!signal.aborted) setTraces(response.traces);
      })
      .catch((err) => {
        if (signal.aborted) return;
        // A 401 is the key banner's to explain, like every other list.
        noteAuthFailure(err);
        setTracesError(apiErrorMessage(err, 'Could not load traces.'));
      });
  };

  useEffect(() => {
    if (rail !== 'traces') return;
    loadTraces();
    return () => tracesRequestRef.current?.abort();
    // loadTraces only calls setters, refs and the API.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rail]);

  // A trace row is a debugging artifact, so opening one implies Engineer mode — the
  // Trace tab does not exist in Reader and the click would otherwise appear to do
  // nothing. Pins the inspector to that trace rather than the newest turn.
  const handleSelectTrace = useStableHandler((traceId: string) => {
    setPinnedTraceId(traceId);
    setMode('engineer');
    setInspectorTab('trace');
    setInspectorOpen(true);
    // Opening a trace is an explicit request to see the inspector, so it outranks the
    // width rule the same way a manual toggle does.
    setInspectorForced(true);
    // On a narrow screen the inspector opens as a drawer; the panel drawer the trace was
    // picked from would otherwise sit on top of it.
    setPanelDrawerOpen(false);
  });

  // Widening past the breakpoint turns the drawer back into a column; forget that it was
  // open so narrowing again later doesn't pop it up unasked.
  useEffect(() => {
    if (!panelCollapsed) setPanelDrawerOpen(false);
  }, [panelCollapsed]);

  // Rail click: on narrow screens it also opens the drawer, and clicking the panel that
  // is already showing closes it again (the rail is the drawer's toggle).
  const handleRailSelect = useStableHandler((panel: RailPanel) => {
    if (panelCollapsed) setPanelDrawerOpen((open) => !(open && panel === rail));
    setRail(panel);
  });

  // Width-driven open/close, deferring to the user once they have taken a position —
  // except that a side panel opened on a wide window closes when the window narrows:
  // below that width it is a modal drawer, which took focus and trapped Tab for an
  // inspector nobody had just asked for.
  const inspectorOpenedNarrowRef = useRef(false);
  useEffect(() => {
    if (inspectorOpen) inspectorOpenedNarrowRef.current = tooNarrowForInspector;
    // Only when it opens: a resize while open is the effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inspectorOpen]);
  useEffect(() => {
    if (inspectorForced) {
      if (tooNarrowForInspector && !inspectorOpenedNarrowRef.current) setInspectorOpen(false);
      return;
    }
    if (tooNarrowForInspector) setInspectorOpen(false);
    else if (roomyEnoughForInspector) setInspectorOpen(true);
  }, [inspectorForced, tooNarrowForInspector, roomyEnoughForInspector]);

  const apiReachable = apiStatus === 'ok';

  // Files dropped anywhere outside the corpus panel's own dropzone upload directly
  // (that dropzone stages them for review instead).
  const fileDragActive = useWindowFileDrop(apiReachable && !readOnly, (files) => {
    const { accepted, rejected } = validateUploads(
      files,
      config?.max_upload_bytes ?? DEFAULT_MAX_UPLOAD_BYTES,
      acceptedExtensions(config?.supported_extensions),
    );
    if (rejected.length > 0) {
      pushToast({
        tone: 'warn',
        title: `${rejected.length} file${rejected.length === 1 ? '' : 's'} not uploaded`,
        body: rejected.map((item) => item.message).join(' '),
      });
    }
    if (accepted.length > 0) {
      setRail('corpus');
      void handleUpload(accepted);
    }
  });
  // The server can go away after boot (a restart, a laptop's network dropping). The
  // footer said "api ok" for the rest of the session, and the first sign was a failed
  // question. /health is polled while the page is visible — every 30 s, every 5 s once
  // it has failed — and the browser's offline/online events are taken at their word.
  const [connectionLost, setConnectionLost] = useState(false);
  const connectionLostRef = useRef(false);
  connectionLostRef.current = connectionLost;
  const booted = apiStatus === 'ok' || apiStatus === 'unauthorized';
  useEffect(() => {
    if (!booted) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const schedule = () => {
      clearTimeout(timer);
      timer = setTimeout(
        () => void check(),
        connectionLostRef.current ? HEALTH_RETRY_MS : HEALTH_POLL_MS,
      );
    };
    const markLost = (lost: boolean) => {
      // The ref too, at once: the next check is scheduled before React re-renders.
      connectionLostRef.current = lost;
      setConnectionLost(lost);
    };
    const check = async () => {
      if (document.visibilityState !== 'hidden') {
        try {
          await api.health();
          if (!cancelled) markLost(false);
        } catch {
          if (!cancelled) markLost(true);
        }
      }
      if (!cancelled) schedule();
    };
    const goneOffline = () => {
      markLost(true);
      schedule();
    };
    const backOnline = () => void check();
    window.addEventListener('offline', goneOffline);
    window.addEventListener('online', backOnline);
    schedule();
    return () => {
      cancelled = true;
      clearTimeout(timer);
      window.removeEventListener('offline', goneOffline);
      window.removeEventListener('online', backOnline);
    };
  }, [booted]);

  // Retry a listing that failed (not a 401) while the page is open, backing off.
  useEffect(() => {
    if (!listingFailed) return;
    let cancelled = false;
    let delay = LISTING_RETRY_BASE_MS;
    let timer: ReturnType<typeof setTimeout>;
    const attempt = () => {
      timer = setTimeout(() => {
        refreshDocuments()
          .then(() => {
            if (!cancelled) setListingFailed(false);
          })
          .catch((err) => {
            if (cancelled) return;
            if (noteAuthFailure(err)) {
              setListingFailed(false);
              return;
            }
            delay = Math.min(delay * 2, BOOT_RETRY_MAX_MS);
            attempt();
          });
      }, delay);
    };
    attempt();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [listingFailed, refreshDocuments, noteAuthFailure]);
  const noDocs = corpusChecked && !listingFailed && indexedFilenames.length === 0;
  const booting = apiStatus === 'checking' || !corpusChecked;
  // Auth takes precedence over noDocs: when the corpus list 401s, "no documents" is a
  // symptom, and telling the user to upload one would send them at a call that 401s too.
  const composerHint =
    apiStatus === 'unauthorized'
      ? 'This server requires an API key — add one in Settings.'
      : connectionLost
        ? 'Can’t reach the server right now — checking again every few seconds.'
        : listingFailed
          ? 'Couldn’t load the document list — retrying. You can still ask.'
          : noDocs
            ? 'Upload a document to start.'
            : booting
              ? 'Connecting…'
              : undefined;
  const activeConversation = conversations.find((c) => c.id === activeConversationId);
  const streamingTitle = conversations.find((c) => c.id === streamingConversationId)?.title;
  const streamingElsewhereHint = `Still answering in “${streamingTitle ?? 'another conversation'}” — you can type, and send once it finishes.`;
  // From the conversation asked in, which needn't be the one open by the time it ends.
  const lastQuestion = [...askedTurns].reverse().find((turn) => turn.role === 'user')?.content;
  const lastTurn = askedTurns[askedTurns.length - 1];
  // A question stopped before the server sent anything leaves no answer turn at all: the
  // question itself is the last turn, and it read as "Answer ready".
  const lastOutcome =
    lastTurn?.role === 'error'
      ? 'failed'
      : lastTurn?.stopped || lastTurn?.role === 'user'
        ? 'stopped'
        : 'answered';
  useBackgroundNotice(pending, notifyOnAnswer, lastQuestion, lastOutcome);
  // The selected provider's model. Each answer records the one it was asked of
  // (ChatTurn.model); this labels answers saved before that, and the composer.
  const currentModelLabel = config
    ? selectedProvider === 'ollama'
      ? config.llm_model
      : selectedProvider === 'openai_compatible'
        ? config.openai_compatible_model || 'OpenAI-compatible'
        : config.openai_model
    : undefined;

  // The turn the inspector describes: the one pinned by a trace-row click, else the
  // most recent assistant turn so the panel tracks the conversation without a click.
  // A pin this conversation doesn't hold describes no turn here: falling back to the
  // latest answer showed that answer's trace instead of the row that was clicked.
  const inspectedTurn = pinnedTraceId
    ? turns.find((turn) => turn.traceId === pinnedTraceId)
    : [...turns].reverse().find((turn) => turn.role === 'assistant');
  // A pinned trace from the trace browser may belong to a turn this conversation never
  // held (another session, or one since cleared) — fall back to the id itself so the
  // Retrieval and Trace tabs can still fetch and render it.
  const inspectedTraceId = inspectedTurn?.traceId ?? pinnedTraceId;
  // The server sends trace_id in the `sources` SSE event, before generation even
  // starts (so a mid-stream failure can still be linked to a trace) — but only writes
  // the trace itself once the turn reaches its terminal event, which is also the
  // moment `timings` gets set (see useChat's onDone handler). Fetching in that gap
  // 404s; a trace pinned from the trace browser is never mid-stream, so it defaults
  // ready. Found via a real ~19s Ollama generation — a mocked instant SSE stream can't
  // reproduce the gap this guards against.
  // Only the in-flight turn can be in that gap. A turn that ended without `done`
  // (cancelled, or failed mid-stream) never gets timings, but the server still wrote an
  // error trace for it — gating on timings alone left its Inspector stuck forever.
  const inspectedTurnInFlight =
    streamingHere && inspectedTurn !== undefined && inspectedTurn.id === turns[turns.length - 1]?.id;
  const inspectedTraceReady = inspectedTurn
    ? inspectedTurn.timings !== null || !inspectedTurnInFlight
    : true;

  // Shared by both the input box and a Retry click on a failed turn — retry always
  // uses the CURRENT provider/overrides/scope, not whatever was selected when the
  // original question failed. No selection = search the whole corpus (undefined).
  // useCallback (not a plain const) because this is passed down as ChatMessage's
  // onRetry prop through ChatThread — an unstable reference here would defeat
  // React.memo(ChatMessage) and reintroduce a full-list re-render on every SSE delta.
  const scopeFilenames = selectedFilenames.length > 0 ? selectedFilenames : undefined;
  const currentScopeLabel = scopeLabel(indexedFilenames, selectedFilenames, selectedTags);
  const askQuestion = useCallback(
    (question: string) =>
      ask(question, selectedProvider, advancedOptions, scopeFilenames, {
        tags: selectedTags,
        modelLabel: currentModelLabel,
      }),
    [ask, selectedProvider, advancedOptions, scopeFilenames, selectedTags, currentModelLabel],
  );
  // Regenerate must produce a new answer, not the cached copy of the one on screen.
  const regenerateQuestion = useCallback(
    (question: string) =>
      ask(question, selectedProvider, advancedOptions, scopeFilenames, {
        tags: selectedTags,
        bypassCache: true,
        modelLabel: currentModelLabel,
      }),
    [ask, selectedProvider, advancedOptions, scopeFilenames, selectedTags, currentModelLabel],
  );
  const editQuestion = useCallback(
    (turnId: string, question: string) => {
      setPinnedTraceId(null);
      return editAndResend(turnId, question, selectedProvider, advancedOptions, scopeFilenames, {
        tags: selectedTags,
        modelLabel: currentModelLabel,
      });
    },
    [editAndResend, selectedProvider, advancedOptions, scopeFilenames, selectedTags, currentModelLabel],
  );

  // Same reasoning as askQuestion above — passed to both ChatThread and Inspector.
  const handleOpenSource = useCallback(
    (filename: string, chunkId: string, highlights?: string[]) =>
      setSourceView({ filename, chunkId, highlights }),
    [],
  );
  const handleReadDocument = useCallback((filename: string) => {
    setSourceView({ filename, chunkId: null });
    // On a narrow screen the panel is a drawer over the viewer's opener.
    setPanelDrawerOpen(false);
  }, []);
  // A citation's viewer page is fetched once the pointer (or focus) rests on it a
  // moment: tabbing through an answer used to request a 61-chunk page per tab stop.
  const prefetchTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(prefetchTimerRef.current), []);
  const handleCitationLeave = useCallback(() => {
    clearTimeout(prefetchTimerRef.current);
    setHoveredCitation(null);
  }, []);
  // Escape dismisses the preview (WCAG 1.4.13: content shown on hover or focus can be
  // dismissed without moving the pointer or focus).
  useEffect(() => {
    if (!hoveredCitation) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setHoveredCitation(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [hoveredCitation]);
  // A pointer or focus on a citation usually precedes a click on it: the viewer's code
  // and its first page start loading now, so the click opens it without waiting.
  const handleCitationHover = useCallback((citation: CitationResponse) => {
    setHoveredCitation(citation);
    clearTimeout(prefetchTimerRef.current);
    prefetchTimerRef.current = setTimeout(() => {
      void import('./components/DocumentViewer');
      prefetchCitationContent(citation.filename, citation.chunk_id);
    }, CITATION_PREFETCH_DWELL_MS);
  }, []);

  // Shown at once (ChatMessage's feedbackGiven); taken back if it couldn't be sent, so
  // the buttons don't claim a rating the server never recorded and it can be retried.
  const handleFeedback = useCallback((payload: FeedbackPayload) => {
    setTurnFeedback(payload.turnId, payload.rating);
    void api
      .submitFeedback({
        trace_id: payload.traceId,
        question: payload.question,
        answer_excerpt: sliceWhole(payload.answerExcerpt, 2000),
        cited_filenames: payload.citedFilenames,
        rating: payload.rating,
        citation_source_number: null,
      })
      .catch((err) => {
        setTurnFeedback(payload.turnId, undefined);
        if (noteAuthFailure(err)) return;
        pushToast({
          tone: 'warn',
          title: "Couldn't send your rating",
          body: apiErrorMessage(err, 'Try again in a moment.'),
        });
      });
    // noteAuthFailure only calls state setters; pushToast is stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setTurnFeedback, pushToast]);

  // Persistence problems arrive as state flags, not events, so they are surfaced with
  // stable ids — a re-render must refresh the same notice rather than stack duplicates.
  useEffect(() => {
    if (persistError) {
      pushToast({
        id: 'persist-error',
        tone: 'bad',
        title: "Chat history couldn't be saved",
        body: "Storage may be full — this session won't be here after a reload.",
        sticky: true,
      });
    } else if (persistPartial) {
      pushToast({
        id: 'persist-partial',
        tone: 'warn',
        title: 'Chat history was only partly saved',
        body: 'Browser storage is full or unavailable. Export anything you need to keep.',
        sticky: true,
      });
    }
    // Sticky, so they must be withdrawn once a later save fully succeeds.
    if (!persistError) dismissToast('persist-error');
    if (!persistPartial) dismissToast('persist-partial');
  }, [persistError, persistPartial, pushToast, dismissToast]);

  // Read when clicked or exported rather than closed over: the header's Export and the
  // palette keep one identity while an answer streams in.
  const turnsRef = useRef(turns);
  turnsRef.current = turns;
  const conversationTitleRef = useRef(activeConversation?.title);
  conversationTitleRef.current = activeConversation?.title;

  const exportMarkdown = useCallback(() => {
    if (turnsRef.current.length === 0) return;
    downloadFile(
      `docrag-${conversationTitleRef.current ?? 'chat'}.md`.replace(/[^\w.-]+/g, '-'),
      'text/markdown',
      chatToMarkdown(turnsRef.current),
    );
  }, []);

  const exportJson = useCallback(() => {
    if (turnsRef.current.length === 0) return;
    downloadFile(
      `docrag-${conversationTitleRef.current ?? 'chat'}.json`.replace(/[^\w.-]+/g, '-'),
      'application/json',
      chatToJson(turnsRef.current),
    );
  }, []);

  const openPalette = useCallback(() => setPaletteOpen(true), []);
  // Settings needs /config to render. Opened before it arrived, the flag sat set with
  // nothing shown — every ⌘ shortcut then thought a dialog was open — and the modal
  // popped up by itself once the config came in.
  const configLoaded = config !== null;
  const openSettings = useCallback(() => {
    if (configLoaded) setSettingsOpen(true);
  }, [configLoaded]);
  const closeInspector = useCallback(() => setInspectorOpen(false), []);
  const closePanelDrawer = useCallback(() => setPanelDrawerOpen(false), []);
  const changeProvider = useCallback((provider: LlmProvider) => {
    setSelectedProvider(provider);
    writeStored(PROVIDER_STORAGE_KEY, provider);
  }, []);
  const inspectorOpenRef = useRef(inspectorOpen);
  inspectorOpenRef.current = inspectorOpen;
  const toggleInspector = useCallback(() => {
    // Reopening on a new turn should follow the conversation again rather than
    // resurface whatever trace row was last clicked.
    if (!inspectorOpenRef.current) setPinnedTraceId(null);
    setInspectorForced(true);
    setInspectorOpen((prev) => !prev);
  }, []);

  const importInputRef = useRef<HTMLInputElement>(null);

  const exportControllerRef = useRef<AbortController | null>(null);
  useEffect(() => () => exportControllerRef.current?.abort(), []);

  // Starts a backup download, or cancels the one running (the palette command toggles).
  const downloadBackup = () => {
    if (exportingRef.current) {
      exportControllerRef.current?.abort();
      return;
    }
    exportingRef.current = true;
    setExportingBackup(true);
    const controller = new AbortController();
    exportControllerRef.current = controller;
    pushToast({ tone: 'info', title: 'Preparing backup…', body: 'The download starts when it is ready.' });
    api
      .exportCorpus(controller.signal)
      .then(({ blob, filename }) => downloadFile(filename, 'application/zip', blob))
      .catch((err) => {
        if (controller.signal.aborted) {
          pushToast({ tone: 'info', title: 'Backup cancelled' });
          return;
        }
        if (noteAuthFailure(err)) return;
        pushToast({
          tone: 'bad',
          title: 'Backup failed',
          body: apiErrorMessage(err, 'Could not build the backup.'),
        });
      })
      .finally(() => {
        exportingRef.current = false;
        exportControllerRef.current = null;
        setExportingBackup(false);
      });
  };

  const startUpload = useStableHandler(() => {
    setRail('corpus');
    // On a narrow screen the panel is a closed drawer: the picked files were staged
    // out of sight, still waiting on an Upload click nobody could see.
    if (panelCollapsed) setPanelDrawerOpen(true);
    // The panel has to render before its hidden file input can be clicked.
    requestAnimationFrame(() => corpusBrowseInputRef.current?.click());
  });
  // Example questions for an empty conversation: about the newest documents, which the
  // user most likely means to ask about. A click puts one in the composer to edit.
  const suggestions = useMemo(() => {
    const newest = [...indexedFilenames]
      .sort((a, b) => (documents[b]?.uploadedAt ?? 0) - (documents[a]?.uploadedAt ?? 0))
      .slice(0, 3);
    return newest.map((filename) => `What are the key points of ${filename}?`);
  }, [indexedFilenames, documents]);
  const [prefill, setPrefill] = useState<{ text: string; nonce: number } | null>(null);
  const suggestQuestion = useCallback(
    (text: string) => setPrefill((prev) => ({ text, nonce: (prev?.nonce ?? 0) + 1 })),
    [],
  );

  const commands = useCommands(
    {
      pending: streamingHere,
      configLoaded: config !== null,
      inspectorOpen,
      mode,
      theme,
      hasTurns: turns.length > 0,
      conversations,
      activeConversationId,
      readOnly,
      restoring,
      exportingBackup,
    },
    {
      newConversation: () => {
        setRail('chat');
        newConversation();
      },
      upload: startUpload,
      openSettings,
      toggleInspector,
      setMode,
      toggleTheme,
      pinConversation,
      exportMarkdown,
      exportJson,
      searchPassages: () => {
        setRail('search');
        if (panelCollapsed) setPanelDrawerOpen(true);
      },
      browseTraces: () => {
        setRail('traces');
        if (panelCollapsed) setPanelDrawerOpen(true);
      },
      restoreBackup: () => restoreInputRef.current?.click(),
      downloadBackup,
      importConversation: () => importInputRef.current?.click(),
      clear,
      switchConversation,
    },
    {
      otherDialogOpen: settingsOpen || sourceView !== null,
      togglePalette: () => setPaletteOpen((prev) => !prev),
      closePalette: () => setPaletteOpen(false),
    },
  );

  const handleRestoreFile = async (file: File) => {
    setRestoring(true);
    pushToast({ tone: 'info', title: 'Restoring backup…', body: file.name });
    let result;
    try {
      result = await api.importBackup(file);
    } catch (err) {
      if (!noteAuthFailure(err)) {
        pushToast({
          tone: 'bad',
          title: 'Restore failed',
          body: apiErrorMessage(err, 'Could not read the backup.'),
        });
      }
      return;
    } finally {
      setRestoring(false);
    }
    const notes = [
      result.skipped.length > 0 && `${result.skipped.length} already indexed (kept as is)`,
      result.deferred.length > 0 &&
        `${result.deferred.length} waiting for room in the queue — restore the same file again later`,
      result.missing_originals.length > 0 &&
        `${result.missing_originals.length} had no stored original in the backup`,
      result.rejected.length > 0 && `${result.rejected.length} couldn't be used`,
      result.feedback_imported > 0 && `${result.feedback_imported} ratings restored`,
    ].filter(Boolean);
    pushToast({
      tone: result.deferred.length + result.rejected.length > 0 ? 'warn' : 'good',
      title: `Restoring ${result.jobs.length} document${result.jobs.length === 1 ? '' : 's'}`,
      body: notes.length > 0 ? `${notes.join('; ')}.` : undefined,
    });
    if (result.jobs.length > 0) {
      setRail('corpus');
      await trackJobs(result.jobs);
    }
  };

  const downloadOriginal = useCallback(
    (filename: string) => {
      api
        .getDocumentOriginal(filename)
        .then((blob) => downloadFile(filename, blob.type || 'application/octet-stream', blob))
        .catch((err) => {
          if (noteAuthFailure(err)) return;
          pushToast({
            tone: 'bad',
            title: 'Download failed',
            body: apiErrorMessage(err, 'Could not fetch the original.'),
          });
        });
    },
    // noteAuthFailure only calls state setters; pushToast is stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [pushToast],
  );
  const handleImportFile = async (file: File) => {
    let parsed: unknown;
    try {
      parsed = JSON.parse(await file.text());
    } catch {
      parsed = null;
    }
    const imported = parsed !== null && importConversation(parsed);
    pushToast(
      imported
        ? { tone: 'good', title: 'Conversation imported', body: file.name }
        : {
            tone: 'bad',
            title: "Couldn't import that file",
            body: 'Choose a conversation exported as JSON from DocRAG.',
          },
    );
    if (imported) {
      setPinnedTraceId(null);
      setRail('chat');
    }
  };

  const handlePanelAction = useStableHandler(() => {
    if (rail === 'chat') {
      newConversation();
      setPanelDrawerOpen(false);
    } else if (rail === 'corpus') {
      if (readOnly) void refreshDocuments().catch(noteAuthFailure);
      else corpusBrowseInputRef.current?.click();
    }
    else if (rail === 'search') setSearchResetKey((key) => key + 1);
    else if (rail === 'feedback') loadFeedback();
    else loadTraces();
  });

  return (
    <div className="app-shell">
      {/* First stop for Tab: past the rail and the whole context panel to the question. */}
      <a
        className="app-skip-link"
        href={`#${QUESTION_INPUT_ID}`}
        onClick={(event) => {
          event.preventDefault();
          document.getElementById(QUESTION_INPUT_ID)?.focus();
        }}
      >
        Skip to question
      </a>
      {fileDragActive && (
        <div className="app-drop-overlay" data-testid="app-drop-overlay">
          <p>Drop to upload</p>
        </div>
      )}
      <input
        ref={importInputRef}
        type="file"
        accept=".json,application/json"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = '';
          if (file) void handleImportFile(file);
        }}
        data-testid="import-conversation-input"
      />
      <input
        ref={restoreInputRef}
        type="file"
        accept=".zip,application/zip"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = '';
          if (file) void handleRestoreFile(file);
        }}
        data-testid="restore-backup-input"
      />
      <IconRail
        active={rail}
        onSelect={handleRailSelect}
        theme={theme}
        onToggleTheme={toggleTheme}
        onOpenSettings={openSettings}
        settingsAvailable={configLoaded}
        hidden={feedbackPanelAvailable ? NO_HIDDEN_PANELS : HIDDEN_FEEDBACK_PANEL}
      />
      <ContextPanel
        title={PANEL_META[rail].title}
        actionLabel={rail === 'corpus' && readOnly ? 'Refresh' : PANEL_META[rail].actionLabel}
        onAction={handlePanelAction}
        apiStatus={connectionLost && booted ? 'error' : apiStatus}
        chunkTotal={chunkTotal}
        drawer={panelCollapsed}
        drawerOpen={panelDrawerOpen}
        onCloseDrawer={closePanelDrawer}
      >
        {rail === 'chat' && (
          <ConversationList
            conversations={conversations}
            activeId={activeConversationId}
            onSwitch={switchConversationFromList}
            onRename={renameConversation}
            onDelete={deleteConversation}
            onPin={pinConversation}
            onSearch={searchConversations}
            streamingId={pending ? streamingConversationId : null}
          />
        )}
        {rail === 'corpus' && (
          <CorpusPanel
            filenames={indexedFilenames}
            documents={documents}
            uploads={uploads}
            onUpload={handleUpload}
            onDelete={handleDeleteDocument}
            maxUploadBytes={config?.max_upload_bytes}
            supportedExtensions={config?.supported_extensions}
            disabled={pending}
            browseInputRef={corpusBrowseInputRef}
            onRetryUpload={handleRetryUpload}
            onDismissUpload={handleDismissUpload}
            onCancelUpload={readOnly ? undefined : handleCancelUpload}
            onReindex={config?.raw_documents_enabled && !readOnly ? handleReindex : undefined}
            onReindexMany={config?.raw_documents_enabled && !readOnly ? handleReindexMany : undefined}
            onReindexAllStale={
              config?.raw_documents_enabled && !readOnly ? handleReindexAllStale : undefined
            }
            onSetTags={readOnly ? undefined : handleSetTags}
            readOnly={readOnly}
            onOpenOriginal={config?.raw_documents_enabled ? downloadOriginal : undefined}
            onRead={handleReadDocument}
          />
        )}
        {/* Kept once opened, hidden while another panel shows: its query and results
            used to be lost on every switch to the corpus or traces and back. */}
        {searchOpened && (
          <div className="app-kept-panel" hidden={rail !== 'search'}>
            <SearchPanel
              filenames={selectedFilenames}
              tags={selectedTags}
              onOpenSource={handleOpenSource}
              resetKey={searchResetKey}
              onAuthFailure={noteAuthFailure}
            />
          </div>
        )}
        {rail === 'traces' && (
          <TracesPanel traces={traces} error={tracesError} onSelect={handleSelectTrace} />
        )}
        {rail === 'feedback' && (
          <FeedbackPanel
            feedback={feedbackItems}
            error={feedbackError}
            onSelectTrace={handleSelectTrace}
          />
        )}
      </ContextPanel>
      <main className="app-main">
        {apiStatus === 'error' ? (
          <div className="app-main__unreachable" role="alert">
            <p>
              Cannot reach the DocRAG API &mdash; trying again on its own. Start the API service
              if it isn&rsquo;t running.
            </p>
            <button type="button" onClick={retryBoot} data-testid="api-retry">
              Retry now
            </button>
          </div>
        ) : (
          <>
            <ChatHeader
              key={activeConversationId ?? 'none'}
              title={activeConversation?.title ?? 'New chat'}
              scopeLabel={currentScopeLabel}
              mode={mode}
              onSetMode={setMode}
              hasTurns={turns.length > 0}
              onClear={clear}
              onExport={exportMarkdown}
              disabled={streamingHere}
              onOpenPalette={openPalette}
              inspectorOpen={inspectorOpen}
              onToggleInspector={toggleInspector}
            />
            <ChatThread
              // Per conversation: "a new turn arrived" was judged by the turn count, so
              // switching to a conversation of the same length kept the last one's
              // scroll position and unpinned state.
              // (Prefixed: ChatHeader, a sibling, is keyed by the same id.)
              key={`thread-${activeConversationId ?? 'none'}`}
              turns={turns}
              pending={streamingHere}
              busy={pending}
              stage={streamingHere ? stage : null}
              engineerMode={mode === 'engineer'}
              currentModelLabel={currentModelLabel}
              onRetry={askQuestion}
              onRegenerate={regenerateQuestion}
              onEditQuestion={editQuestion}
              feedbackEnabled={config?.feedback_enabled}
              onFeedback={handleFeedback}
              onOpenSource={handleOpenSource}
              onCitationHover={handleCitationHover}
              onCitationLeave={handleCitationLeave}
              documentCount={indexedFilenames.length}
              documentsLoading={!corpusChecked}
              suggestions={suggestions}
              onSuggest={suggestQuestion}
              onUpload={readOnly ? undefined : startUpload}
              focusTurn={focusTurn}
            />
            {hoveredCitation && <SourcePreview citation={hoveredCitation} />}
          </>
        )}
        {/* Outside the branch above, at a fixed position among main's children: a toast
            pushed while the API is unreachable used to go nowhere, and the live regions
            remounted (and so were not announced) when it came back. */}
        <ToastRow
          toasts={toasts}
          onDismiss={dismissToast}
          onHold={holdToast}
          onRelease={releaseToast}
        />
        {apiStatus !== 'error' && (
          <Composer
            onSubmit={askQuestion}
            // Typing is fine while the app boots; only sending waits for it.
            disabled={apiStatus === 'unauthorized' || noDocs}
            sendHeld={booting || (pending && !streamingHere)}
            hint={composerHint ?? (pending && !streamingHere ? streamingElsewhereHint : undefined)}
            pending={streamingHere}
            onCancel={cancel}
            scopeLabel={currentScopeLabel}
            indexedFilenames={indexedFilenames}
            selectedFilenames={selectedFilenames}
            onSelectedFilenamesChange={setSelectedFilenames}
            availableTags={availableTags}
            selectedTags={selectedTags}
            onSelectedTagsChange={setSelectedTags}
            config={config}
            provider={selectedProvider}
            onProviderChange={changeProvider}
            providerLabel={currentModelLabel ?? selectedProvider}
            draftKey={activeConversationId ?? NEW_CONVERSATION_DRAFT}
            prefill={prefill}
          />
        )}
      </main>
      <LazyChunkBoundary
        onDismiss={() => {
          setInspectorOpen(false);
          setPaletteOpen(false);
          setSettingsOpen(false);
          setSourceView(null);
        }}
      >
        <Suspense fallback={null}>
          {inspectorOpen && apiStatus !== 'error' && (
            <Inspector
              engineerMode={mode === 'engineer'}
              tab={inspectorTab}
              onTabChange={setInspectorTab}
              onClose={closeInspector}
              sources={inspectedTurn?.sources ?? NO_SOURCES}
              traceId={inspectedTraceId ?? null}
              traceReady={inspectedTraceReady}
              onOpenSource={handleOpenSource}
              overlay={tooNarrowForInspector}
            />
          )}
          {paletteOpen && (
            <CommandPalette commands={commands} onClose={() => setPaletteOpen(false)} />
          )}
          {settingsOpen && config && (
            <SettingsModal
              config={config}
              overrides={advancedOptions}
              onOverridesChange={updateAdvancedOptions}
              onClose={() => setSettingsOpen(false)}
              onSaved={() => {
                pushToast({ tone: 'good', title: 'Settings saved' });
                // A newly entered API key only takes effect on the next request; without
                // this refetch the "API key required" state and empty corpus stayed until
                // a manual reload.
                refreshDocuments().catch((err) => {
                  noteAuthFailure(err);
                });
                // A different key may unlock more (or less).
                void refreshAccess();
              }}
              disabled={pending}
              notifyOnAnswer={notifyOnAnswer}
              onNotifyOnAnswerChange={(value) => {
                setNotifyOnAnswer(value);
                writeStored(NOTIFY_STORAGE_KEY, value ? 'true' : null);
              }}
            />
          )}
          {sourceView && (
            <DocumentViewer
              filename={sourceView.filename}
              chunkId={sourceView.chunkId}
              highlights={sourceView.highlights}
              onClose={() => setSourceView(null)}
              onDownloadOriginal={
                config?.raw_documents_enabled && documents[sourceView.filename]?.reindexable
                  ? () => downloadOriginal(sourceView.filename)
                  : undefined
              }
              canPreviewOriginal={canPreviewOriginal(
                sourceView.filename,
                Boolean(config?.raw_documents_enabled),
                documents[sourceView.filename],
              )}
            />
          )}
        </Suspense>
      </LazyChunkBoundary>
    </div>
  );
}

export default App;
