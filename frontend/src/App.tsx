import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { ApiClientError, api } from './api/client';
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
import ContextPanel from './components/ContextPanel';
import ConversationList from './components/ConversationList';
import CorpusPanel from './components/CorpusPanel';
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
import { useChat } from './hooks/useChat';
import { useCommands } from './hooks/useCommands';
import { useCorpus } from './hooks/useCorpus';
import { useResponsiveLayout } from './hooks/useResponsiveLayout';
import { useToasts } from './hooks/useToasts';
import { useWindowFileDrop } from './hooks/useWindowFileDrop';
import { NOTIFY_STORAGE_KEY, useBackgroundNotice } from './hooks/useBackgroundNotice';
import { chatToJson, chatToMarkdown, downloadFile } from './utils/exportChat';
import { EMPTY_OVERRIDES } from './utils/overrides';
import { readStored, readStoredJson, writeStored } from './utils/safeStorage';
import { DEFAULT_MAX_UPLOAD_BYTES, validateUploads } from './utils/uploadValidation';

// Dialogs and the inspector render only on demand, so they load on demand too: the
// first paint (thread + composer) no longer waits on their code.
const CommandPalette = lazy(() => import('./components/CommandPalette'));
const DocumentViewer = lazy(() => import('./components/DocumentViewer'));
const Inspector = lazy(() => import('./components/Inspector'));
const SettingsModal = lazy(() => import('./components/SettingsModal'));

const ADVANCED_OPTIONS_STORAGE_KEY = 'docrag-advanced-options';
const MODE_STORAGE_KEY = 'docrag-mode';
const PROVIDER_STORAGE_KEY = 'docrag-provider';

/** Whether `provider` can be asked on this server: configured, and allowed per request. */
function isProviderUsable(
  provider: string | null,
  config: PublicConfigResponse,
): provider is LlmProvider {
  const allowed = config.allowed_request_providers;
  if (allowed && provider !== null && !allowed.includes(provider)) return false;
  if (provider === 'ollama') return true;
  if (provider === 'openai') return config.openai_available;
  if (provider === 'openai_compatible') return Boolean(config.openai_compatible_available);
  return false;
}
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
  return {
    rerankTopK: within(overrides.rerankTopK, config.rerank_top_k_limit),
    maxContextChunks: within(overrides.maxContextChunks, config.max_context_chunks_limit),
    llmTemperature: within(overrides.llmTemperature, config.llm_temperature_max, 0),
  };
}

function loadPersistedMode(): ChatMode {
  return readStored(MODE_STORAGE_KEY) === 'engineer' ? 'engineer' : 'reader';
}

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
  const [config, setConfig] = useState<PublicConfigResponse | null>(null);
  const [selectedProvider, setSelectedProvider] = useState<LlmProvider>('ollama');
  const [advancedOptions, setAdvancedOptions] = useState<QuestionOverrides>(loadPersistedOverrides);
  const [theme, setTheme] = useState<'dark' | 'light'>(
    () => (document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'),
  );
  const [mode, setModeState] = useState<ChatMode>(loadPersistedMode);
  const [rail, setRail] = useState<RailPanel>('chat');
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
  const { toasts, push: pushToast, dismiss: dismissToast } = useToasts();
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
  const [sourceView, setSourceView] = useState<{ filename: string; chunkId: string } | null>(null);
  const corpusBrowseInputRef = useRef<HTMLInputElement>(null);
  const {
    turns,
    pending,
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
    persistError,
    persistPartial,
    setTurnFeedback,
    scope,
    setScope,
    pruneScopes,
    removeFromScopes,
    editAndResend,
    importConversation,
  } = useChat();
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
  const setSelectedFilenames = (filenames: string[]) =>
    setScope((prev) => ({ ...prev, filenames }));
  const setSelectedTags = (tags: string[]) => setScope((prev) => ({ ...prev, tags }));

  // A pinned trace belongs to one conversation's turns, so anything that replaces the
  // turns array has to release it. Otherwise inspectedTurn's lookup finds nothing, the
  // Retrieval/Trace tabs fall back to the stale pinnedTraceId and keep rendering the
  // previous conversation's trace, while the Sources tab (which reads the live turn)
  // correctly goes empty — one inspector showing two different questions.
  // Stable (useChat's actions are), so the memoized conversation list skips re-rendering
  // on every streamed token.
  const clear = useCallback(() => {
    setPinnedTraceId(null);
    clearTurns();
  }, [clearTurns]);
  const newConversation = useCallback(() => {
    setPinnedTraceId(null);
    createConversation();
  }, [createConversation]);
  const switchConversation = useCallback(
    (id: string) => {
      setPinnedTraceId(null);
      selectConversation(id);
    },
    [selectConversation],
  );
  const deleteConversation = useCallback(
    (id: string) => {
      setPinnedTraceId(null);
      removeConversation(id);
    },
    [removeConversation],
  );
  const switchConversationFromList = useCallback(
    (id: string) => {
      switchConversation(id);
      setPanelDrawerOpen(false);
    },
    [switchConversation],
  );

  const updateAdvancedOptions = (next: QuestionOverrides) => {
    setAdvancedOptions(next);
    writeStored(ADVANCED_OPTIONS_STORAGE_KEY, JSON.stringify(next));
  };

  const setMode = useCallback((next: ChatMode) => {
    setModeState(next);
    writeStored(MODE_STORAGE_KEY, next);
  }, []);

  const toggleTheme = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    document.documentElement.dataset.theme = next;
    writeStored('docrag-theme', next);
  };

  /** Note a 401 so it isn't mistaken for an empty corpus.
   *
   * /health and /config are unauthenticated, so a server with API_KEY set answers both
   * and the app looks healthy — while every /documents call 401s. Swallowing that left
   * indexedFilenames empty, which disabled the composer under the hint "Upload a
   * document to start.", advice that would itself have 401'd. Returns true when it
   * handled the error, so callers keep their own fallback for everything else.
   */
  const noteAuthFailure = (err: unknown): boolean => {
    if (err instanceof ApiClientError && err.statusCode === 401) {
      setApiStatus('unauthorized');
      return true;
    }
    // A 403 means a read-only key (API_READ_KEY): hide what it can't do from now on.
    // The error itself is still shown by the caller.
    if (err instanceof ApiClientError && err.statusCode === 403) setAccess('read');
    return false;
  };

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

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;

    async function checkHealth() {
      try {
        await api.health(controller.signal);
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
        // A provider picked earlier in this browser wins, if the server still offers
        // it; otherwise sync the dropdown to the server's default.
        const remembered = readStored(PROVIDER_STORAGE_KEY);
        if (isProviderUsable(remembered, configResult)) {
          setSelectedProvider(remembered);
        } else if (configResult.llm_provider === 'openai' && configResult.openai_available) {
          setSelectedProvider('openai');
        } else if (
          configResult.llm_provider === 'openai_compatible' &&
          configResult.openai_compatible_available
        ) {
          setSelectedProvider('openai_compatible');
        } else if (
          // Ollama is the hardcoded default above, but if it's unreachable and OpenAI
          // is configured (and allowed), don't leave the user stuck on one that will 502.
          configResult.llm_provider === 'ollama' &&
          !configResult.ollama_available &&
          isProviderUsable('openai', configResult)
        ) {
          setSelectedProvider('openai');
        }
        setApiStatus('ok');
        void refreshAccess(controller.signal);
        try {
          await refreshDocuments(controller.signal);
        } catch (err) {
          if (cancelled) return;
          // Anything other than a 401 is non-fatal — the corpus panel just stays empty
          // until the next refresh.
          noteAuthFailure(err);
        }
      } catch {
        if (cancelled) return;
        setApiStatus('error');
      }
    }

    void checkHealth();
    return () => {
      cancelled = true;
      // Aborting releases the in-flight request instead of just ignoring its result.
      controller.abort();
    };
    // Boot once. refreshDocuments reads only setters and useChat's stable pruneScopes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
  const loadFeedback = (signal?: AbortSignal) => {
    setFeedbackItems(null);
    setFeedbackError(null);
    api
      .listFeedback(200, signal)
      .then((response) => setFeedbackItems(response.feedback))
      .catch((err) => {
        if (signal?.aborted) return;
        noteAuthFailure(err);
        setFeedbackError(err instanceof ApiClientError ? err.message : 'Could not load feedback.');
      });
  };

  useEffect(() => {
    if (rail !== 'feedback') return;
    const controller = new AbortController();
    loadFeedback(controller.signal);
    return () => controller.abort();
    // loadFeedback only calls setters and the API.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rail]);

  // The feedback panel needs feedback on and the full key (GET /feedback is operator data).
  const feedbackPanelAvailable = Boolean(config?.feedback_enabled) && !readOnly;
  useEffect(() => {
    if (rail === 'feedback' && !feedbackPanelAvailable) setRail('chat');
  }, [rail, feedbackPanelAvailable]);

  const loadTraces = (signal?: AbortSignal) => {
    setTraces(null);
    setTracesError(null);
    api
      .listTraces(signal)
      .then((response) => setTraces(response.traces))
      .catch((err) => {
        if (signal?.aborted) return;
        setTracesError(err instanceof ApiClientError ? err.message : 'Could not load traces.');
      });
  };

  useEffect(() => {
    if (rail !== 'traces') return;
    const controller = new AbortController();
    loadTraces(controller.signal);
    return () => controller.abort();
  }, [rail]);

  // A trace row is a debugging artifact, so opening one implies Engineer mode — the
  // Trace tab does not exist in Reader and the click would otherwise appear to do
  // nothing. Pins the inspector to that trace rather than the newest turn.
  const handleSelectTrace = (traceId: string) => {
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
  };

  // Widening past the breakpoint turns the drawer back into a column; forget that it was
  // open so narrowing again later doesn't pop it up unasked.
  useEffect(() => {
    if (!panelCollapsed) setPanelDrawerOpen(false);
  }, [panelCollapsed]);

  // Rail click: on narrow screens it also opens the drawer, and clicking the panel that
  // is already showing closes it again (the rail is the drawer's toggle).
  const handleRailSelect = (panel: RailPanel) => {
    if (panelCollapsed) setPanelDrawerOpen((open) => !(open && panel === rail));
    setRail(panel);
  };

  // Width-driven open/close, deferring to the user once they have taken a position.
  useEffect(() => {
    if (inspectorForced) return;
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
  const noDocs = indexedFilenames.length === 0;
  // Auth takes precedence over noDocs: when the corpus list 401s, "no documents" is a
  // symptom, and telling the user to upload one would send them at a call that 401s too.
  const composerHint =
    apiStatus === 'unauthorized'
      ? 'This server requires an API key — add one in Settings.'
      : noDocs
        ? 'Upload a document to start.'
        : undefined;
  const activeConversation = conversations.find((c) => c.id === activeConversationId);
  const lastQuestion = [...turns].reverse().find((turn) => turn.role === 'user')?.content;
  const lastTurn = turns[turns.length - 1];
  const lastOutcome =
    lastTurn?.role === 'error' ? 'failed' : lastTurn?.stopped ? 'stopped' : 'answered';
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
    pending && inspectedTurn !== undefined && inspectedTurn.id === turns[turns.length - 1]?.id;
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
    (filename: string, chunkId: string) => setSourceView({ filename, chunkId }),
    [],
  );
  const handleCitationLeave = useCallback(() => setHoveredCitation(null), []);

  // Shown at once (ChatMessage's feedbackGiven); taken back if it couldn't be sent, so
  // the buttons don't claim a rating the server never recorded and it can be retried.
  const handleFeedback = useCallback((payload: FeedbackPayload) => {
    setTurnFeedback(payload.turnId, payload.rating);
    void api
      .submitFeedback({
        trace_id: payload.traceId,
        question: payload.question,
        answer_excerpt: payload.answerExcerpt.slice(0, 2000),
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
          body: err instanceof ApiClientError ? err.message : 'Try again in a moment.',
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
        title: 'Only this conversation was saved',
        body: 'Browser storage is full. Export anything you need to keep.',
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

  const downloadBackup = () => {
    if (exportingRef.current) return;
    exportingRef.current = true;
    setExportingBackup(true);
    pushToast({ tone: 'info', title: 'Preparing backup…', body: 'The download starts when it is ready.' });
    api
      .exportCorpus()
      .then(({ blob, filename }) => downloadFile(filename, 'application/zip', blob))
      .catch((err) => {
        if (noteAuthFailure(err)) return;
        pushToast({
          tone: 'bad',
          title: 'Backup failed',
          body: err instanceof ApiClientError ? err.message : 'Could not build the backup.',
        });
      })
      .finally(() => {
        exportingRef.current = false;
        setExportingBackup(false);
      });
  };

  const commands = useCommands(
    {
      pending,
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
      upload: () => {
        setRail('corpus');
        // On a narrow screen the panel is a closed drawer: the picked files were staged
        // out of sight, still waiting on an Upload click nobody could see.
        if (panelCollapsed) setPanelDrawerOpen(true);
        // The panel has to render before its hidden file input can be clicked.
        requestAnimationFrame(() => corpusBrowseInputRef.current?.click());
      },
      openSettings: () => setSettingsOpen(true),
      toggleInspector,
      setMode,
      toggleTheme,
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
          body: err instanceof ApiClientError ? err.message : 'Could not read the backup.',
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
            body: err instanceof ApiClientError ? err.message : 'Could not fetch the original.',
          });
        });
    },
    // noteAuthFailure only calls state setters; pushToast is stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [pushToast],
  );
  const handleImportFile = async (file: File) => {
    if (pending) {
      pushToast({ tone: 'warn', title: 'Wait for the answer to finish', body: 'Then import the conversation.' });
      return;
    }
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

  const handlePanelAction = () => {
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
  };

  return (
    <div className="app-shell">
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
        onOpenSettings={() => setSettingsOpen(true)}
        hidden={feedbackPanelAvailable ? [] : ['feedback']}
      />
      <ContextPanel
        title={PANEL_META[rail].title}
        actionLabel={rail === 'corpus' && readOnly ? 'Refresh' : PANEL_META[rail].actionLabel}
        onAction={handlePanelAction}
        apiStatus={apiStatus}
        chunkTotal={chunkTotal}
        drawer={panelCollapsed}
        drawerOpen={panelDrawerOpen}
        onCloseDrawer={() => setPanelDrawerOpen(false)}
      >
        {rail === 'chat' && (
          <ConversationList
            conversations={conversations}
            activeId={activeConversationId}
            onSwitch={switchConversationFromList}
            onRename={renameConversation}
            onDelete={deleteConversation}
            disabled={pending}
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
            disabled={pending}
            browseInputRef={corpusBrowseInputRef}
            onRetryUpload={handleRetryUpload}
            onDismissUpload={handleDismissUpload}
            onCancelUpload={readOnly ? undefined : handleCancelUpload}
            onReindex={config?.raw_documents_enabled && !readOnly ? handleReindex : undefined}
            onReindexAllStale={
              config?.raw_documents_enabled && !readOnly ? handleReindexAllStale : undefined
            }
            onSetTags={readOnly ? undefined : handleSetTags}
            readOnly={readOnly}
            onOpenOriginal={config?.raw_documents_enabled ? downloadOriginal : undefined}
          />
        )}
        {rail === 'search' && (
          <SearchPanel
            filenames={selectedFilenames}
            tags={selectedTags}
            onOpenSource={handleOpenSource}
            resetKey={searchResetKey}
            onAuthFailure={noteAuthFailure}
          />
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
            Cannot reach the DocRAG API &mdash; start the API service and refresh.
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
              disabled={pending}
              onOpenPalette={openPalette}
              inspectorOpen={inspectorOpen}
              onToggleInspector={toggleInspector}
            />
            <ChatThread
              turns={turns}
              pending={pending}
              stage={stage}
              engineerMode={mode === 'engineer'}
              currentModelLabel={currentModelLabel}
              onRetry={askQuestion}
              onRegenerate={regenerateQuestion}
              onEditQuestion={editQuestion}
              feedbackEnabled={config?.feedback_enabled}
              onFeedback={handleFeedback}
              onOpenSource={handleOpenSource}
              onCitationHover={setHoveredCitation}
              onCitationLeave={handleCitationLeave}
              documentCount={indexedFilenames.length}
            />
            {hoveredCitation && <SourcePreview citation={hoveredCitation} />}
            <ToastRow toasts={toasts} onDismiss={dismissToast} />
            <Composer
              onSubmit={askQuestion}
              disabled={!apiReachable || noDocs}
              hint={composerHint}
              pending={pending}
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
              onProviderChange={(provider) => {
                setSelectedProvider(provider);
                writeStored(PROVIDER_STORAGE_KEY, provider);
              }}
              providerLabel={currentModelLabel ?? selectedProvider}
            />
          </>
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
              onClose={() => setInspectorOpen(false)}
              sources={inspectedTurn?.sources ?? []}
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
              onClose={() => setSourceView(null)}
              onDownloadOriginal={
                config?.raw_documents_enabled && documents[sourceView.filename]?.reindexable
                  ? () => downloadOriginal(sourceView.filename)
                  : undefined
              }
            />
          )}
        </Suspense>
      </LazyChunkBoundary>
    </div>
  );
}

export default App;
