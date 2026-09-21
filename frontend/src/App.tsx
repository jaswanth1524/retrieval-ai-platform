import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ApiClientError, api } from './api/client';
import type {
  ApiStatus,
  CitationResponse,
  LlmProvider,
  PublicConfigResponse,
  QuestionOverrides,
  TraceSummaryResponse,
} from './api/types';
import ChatHeader from './components/ChatHeader';
import type { ChatMode } from './components/ChatHeader';
import ChatThread from './components/ChatThread';
import type { FeedbackPayload } from './components/ChatMessage';
import CommandPalette from './components/CommandPalette';
import type { Command } from './components/CommandPalette';
import Composer from './components/Composer';
import ContextPanel from './components/ContextPanel';
import ConversationList from './components/ConversationList';
import CorpusPanel from './components/CorpusPanel';
import type { UploadItem } from './components/CorpusPanel';
import DocumentViewer from './components/DocumentViewer';
import IconRail from './components/IconRail';
import type { RailPanel } from './components/IconRail';
import Inspector from './components/Inspector';
import type { InspectorTab } from './components/Inspector';
import SettingsModal from './components/SettingsModal';
import SourcePreview from './components/SourcePreview';
import ToastRow from './components/ToastRow';
import TracesPanel from './components/TracesPanel';
import { useChat } from './hooks/useChat';
import { useResponsiveLayout } from './hooks/useResponsiveLayout';
import { useToasts } from './hooks/useToasts';
import { chatToJson, chatToMarkdown, downloadFile } from './utils/exportChat';

const ADVANCED_OPTIONS_STORAGE_KEY = 'docrag-advanced-options';
const MODE_STORAGE_KEY = 'docrag-mode';
const EMPTY_OVERRIDES: QuestionOverrides = {
  rerankTopK: null,
  maxContextChunks: null,
  llmTemperature: null,
};

function loadPersistedOverrides(): QuestionOverrides {
  try {
    const raw = localStorage.getItem(ADVANCED_OPTIONS_STORAGE_KEY);
    if (!raw) return EMPTY_OVERRIDES;
    const parsed = JSON.parse(raw) as Partial<QuestionOverrides>;
    return {
      rerankTopK: parsed.rerankTopK ?? null,
      maxContextChunks: parsed.maxContextChunks ?? null,
      llmTemperature: parsed.llmTemperature ?? null,
    };
  } catch {
    // Storage denied/corrupt — fall back to defaults for this session.
    return EMPTY_OVERRIDES;
  }
}

function loadPersistedMode(): ChatMode {
  try {
    return localStorage.getItem(MODE_STORAGE_KEY) === 'engineer' ? 'engineer' : 'reader';
  } catch {
    return 'reader';
  }
}

const PANEL_META: Record<RailPanel, { title: string; actionLabel: string }> = {
  chat: { title: 'Conversations', actionLabel: 'New' },
  corpus: { title: 'Corpus', actionLabel: 'Upload' },
  traces: { title: 'Traces', actionLabel: 'Refresh' },
};

function scopeLabel(indexedFilenames: string[], selectedFilenames: string[]): string {
  if (selectedFilenames.length === 0) return `All ${indexedFilenames.length} documents`;
  if (selectedFilenames.length === 1) return selectedFilenames[0];
  return `${selectedFilenames.length} documents`;
}

function App() {
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking');
  const [config, setConfig] = useState<PublicConfigResponse | null>(null);
  const [uploads, setUploads] = useState<UploadItem[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<LlmProvider>('ollama');
  const [advancedOptions, setAdvancedOptions] = useState<QuestionOverrides>(loadPersistedOverrides);
  const [selectedFilenames, setSelectedFilenames] = useState<string[]>([]);
  // The full corpus currently indexed in Qdrant (across all sessions). Both the
  // corpus-management panel and the search-scope filter operate over this one list,
  // so any indexed document is scopable regardless of which session uploaded it.
  const [indexedFilenames, setIndexedFilenames] = useState<string[]>([]);
  const [chunkCounts, setChunkCounts] = useState<Record<string, number>>({});
  const [pageCounts, setPageCounts] = useState<Record<string, number>>({});
  const [byteSizes, setByteSizes] = useState<Record<string, number>>({});
  const [uploadedAts, setUploadedAts] = useState<Record<string, number>>({});
  const [theme, setTheme] = useState<'dark' | 'light'>(
    () => (document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'),
  );
  const [mode, setModeState] = useState<ChatMode>(loadPersistedMode);
  const [rail, setRail] = useState<RailPanel>('chat');
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
  const { tooNarrowForInspector, roomyEnoughForInspector } = useResponsiveLayout();
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
  } = useChat();

  // A pinned trace belongs to one conversation's turns, so anything that replaces the
  // turns array has to release it. Otherwise inspectedTurn's lookup finds nothing, the
  // Retrieval/Trace tabs fall back to the stale pinnedTraceId and keep rendering the
  // previous conversation's trace, while the Sources tab (which reads the live turn)
  // correctly goes empty — one inspector showing two different questions.
  const clear = () => {
    setPinnedTraceId(null);
    clearTurns();
  };
  const newConversation = () => {
    setPinnedTraceId(null);
    createConversation();
  };
  const switchConversation = (id: string) => {
    setPinnedTraceId(null);
    selectConversation(id);
  };
  const deleteConversation = (id: string) => {
    setPinnedTraceId(null);
    removeConversation(id);
  };

  const updateAdvancedOptions = (next: QuestionOverrides) => {
    setAdvancedOptions(next);
    try {
      localStorage.setItem(ADVANCED_OPTIONS_STORAGE_KEY, JSON.stringify(next));
    } catch {
      // Persistence is best-effort; the in-session value above already applies.
    }
  };

  const setMode = (next: ChatMode) => {
    setModeState(next);
    try {
      localStorage.setItem(MODE_STORAGE_KEY, next);
    } catch {
      // Persistence is best-effort; the in-session value above already applies.
    }
  };

  const toggleTheme = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    document.documentElement.dataset.theme = next;
    // localStorage can throw in some private-browsing/storage-denied contexts —
    // the theme should still flip for this session even if persistence fails.
    try {
      localStorage.setItem('docrag-theme', next);
    } catch {
      // Persistence is best-effort; the visible toggle above already succeeded.
    }
  };

  const refreshDocuments = async (signal?: AbortSignal) => {
    const documents = await api.listDocuments(signal);
    setIndexedFilenames(documents.filenames);
    setChunkCounts(documents.chunk_counts ?? {});
    setPageCounts(documents.page_counts ?? {});
    setByteSizes(documents.byte_sizes ?? {});
    setUploadedAts(documents.uploaded_ats ?? {});
    // A key that used to be rejected now works — clear the banner.
    setApiStatus((prev) => (prev === 'unauthorized' ? 'ok' : prev));
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
    return false;
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
        // Sync the dropdown to the server's default provider once config loads.
        if (configResult.llm_provider === 'openai' && configResult.openai_available) {
          setSelectedProvider('openai');
        }
        // Ollama is the hardcoded default above, but if it's unreachable and OpenAI
        // is configured, don't leave the user stuck on a provider that will 502.
        if (
          configResult.llm_provider === 'ollama' &&
          !configResult.ollama_available &&
          configResult.openai_available
        ) {
          setSelectedProvider('openai');
        }
        setApiStatus('ok');
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
  }, []);

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
  };

  // Width-driven open/close, deferring to the user once they have taken a position.
  useEffect(() => {
    if (inspectorForced) return;
    if (tooNarrowForInspector) setInspectorOpen(false);
    else if (roomyEnoughForInspector) setInspectorOpen(true);
  }, [inspectorForced, tooNarrowForInspector, roomyEnoughForInspector]);

  const uploadOne = async (file: File, id: string) => {
    try {
      const accepted = await api.uploadDocument(file);
      const status = await api.pollDocumentJob(accepted.job_id, (jobStatus) => {
        setUploads((prev) =>
          prev.map((item) =>
            item.id === id
              ? {
                  ...item,
                  progress: {
                    state: jobStatus.state,
                    chunksDone: jobStatus.chunks_done,
                    chunksTotal: jobStatus.chunks_total,
                  },
                }
              : item,
          ),
        );
      });
      if (status.state === 'failed') {
        setUploads((prev) =>
          prev.map((item) =>
            item.id === id
              ? { ...item, status: 'error', error: status.error ?? 'Ingestion failed.' }
              : item,
          ),
        );
        return;
      }
      setUploads((prev) =>
        prev.map((item) =>
          item.id === id ? { ...item, status: 'success', result: status.result ?? undefined } : item,
        ),
      );
      const filename = status.result?.filename;
      if (filename) {
        // Dedupe by name — re-uploading the same filename replaces its chunks
        // server-side, so the corpus list should not grow a second entry for it.
        setIndexedFilenames((prev) => (prev.includes(filename) ? prev : [...prev, filename]));
      }
    } catch (err) {
      noteAuthFailure(err);
      setUploads((prev) =>
        prev.map((item) =>
          item.id === id
            ? { ...item, status: 'error', error: err instanceof ApiClientError ? err.message : 'Upload failed.' }
            : item,
        ),
      );
    }
  };

  const handleUpload = async (files: File[]) => {
    const newItems: UploadItem[] = files.map((file) => ({
      id: crypto.randomUUID(),
      filename: file.name,
      status: 'uploading',
    }));
    setUploads((prev) => [...prev, ...newItems]);
    await Promise.allSettled(files.map((file, index) => uploadOne(file, newItems[index].id)));
    // Once for the whole batch, not once per file. Chunk counts live server-side and
    // GET /documents scrolls the entire collection to compute them, so refreshing
    // inside uploadOne meant a 20-file drop triggered 20 full-collection scans.
    try {
      await refreshDocuments();
    } catch (err) {
      // Non-fatal — counts just stay stale until the next refresh.
      noteAuthFailure(err);
    }
  };

  const handleDeleteDocument = async (filename: string) => {
    await api.deleteDocument(filename);
    // A deleted document can never remain in the corpus or the active search scope.
    setIndexedFilenames((prev) => prev.filter((name) => name !== filename));
    setSelectedFilenames((prev) => prev.filter((name) => name !== filename));
    setChunkCounts((prev) => {
      const next = { ...prev };
      delete next[filename];
      return next;
    });
  };

  const apiReachable = apiStatus === 'ok';
  const noDocs = indexedFilenames.length === 0;
  // Auth takes precedence over noDocs: when the corpus list 401s, "no documents" is a
  // symptom, and telling the user to upload one would send them at a call that 401s too.
  const composerHint =
    apiStatus === 'unauthorized'
      ? 'This server requires an API key — add one in Settings.'
      : noDocs
        ? 'Upload a document to start.'
        : undefined;
  const chunkTotal = Object.values(chunkCounts).reduce((sum, n) => sum + n, 0);
  const activeConversation = conversations.find((c) => c.id === activeConversationId);
  // The engineer meta line's model label is the CURRENTLY selected provider's model,
  // not necessarily the one that answered an older turn — the app doesn't record a
  // per-turn model, and adding that is a backend change out of scope for the redesign.
  const currentModelLabel = config
    ? selectedProvider === 'ollama'
      ? config.llm_model
      : config.openai_model
    : undefined;

  // The turn the inspector describes: the one pinned by a trace-row click, else the
  // most recent assistant turn so the panel tracks the conversation without a click.
  const inspectedTurn =
    (pinnedTraceId ? turns.find((turn) => turn.traceId === pinnedTraceId) : undefined) ??
    [...turns].reverse().find((turn) => turn.role === 'assistant');
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
  const inspectedTraceReady = inspectedTurn ? inspectedTurn.timings !== null : true;

  // Shared by both the input box and a Retry click on a failed turn — retry always
  // uses the CURRENT provider/overrides/scope, not whatever was selected when the
  // original question failed. No selection = search the whole corpus (undefined).
  // useCallback (not a plain const) because this is passed down as ChatMessage's
  // onRetry prop through ChatThread — an unstable reference here would defeat
  // React.memo(ChatMessage) and reintroduce a full-list re-render on every SSE delta.
  const askQuestion = useCallback(
    (question: string) =>
      ask(
        question,
        selectedProvider,
        advancedOptions,
        selectedFilenames.length > 0 ? selectedFilenames : undefined,
      ),
    [ask, selectedProvider, advancedOptions, selectedFilenames],
  );

  // Same reasoning as askQuestion above — passed to both ChatThread and Inspector.
  const handleOpenSource = useCallback(
    (filename: string, chunkId: string) => setSourceView({ filename, chunkId }),
    [],
  );
  const handleCitationLeave = useCallback(() => setHoveredCitation(null), []);

  // Fire-and-forget: a rating is low-stakes feedback, not an action the user needs
  // confirmed or retried on failure — ChatMessage already shows the pick optimistically
  // (see its feedbackGiven state) before this even resolves.
  const handleFeedback = useCallback((payload: FeedbackPayload) => {
    void api
      .submitFeedback({
        trace_id: payload.traceId,
        question: payload.question,
        answer_excerpt: payload.answerExcerpt.slice(0, 2000),
        cited_filenames: payload.citedFilenames,
        rating: payload.rating,
        citation_source_number: null,
      })
      .catch(() => {
        // Best-effort — nothing in the UI depends on this succeeding.
      });
  }, []);

  // Persistence problems arrive as state flags, not events, so they are surfaced with
  // stable ids — a re-render must refresh the same notice rather than stack duplicates.
  useEffect(() => {
    if (persistError) {
      pushToast({
        id: 'persist-error',
        tone: 'bad',
        title: "Chat history couldn't be saved",
        body: "Storage may be full — this session won't be here after a reload.",
      });
    } else if (persistPartial) {
      pushToast({
        id: 'persist-partial',
        tone: 'warn',
        title: 'Only this conversation was saved',
        body: 'Browser storage is full. Export anything you need to keep.',
      });
    }
  }, [persistError, persistPartial, pushToast]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        setPaletteOpen((prev) => !prev);
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, []);

  const exportMarkdown = useCallback(() => {
    if (turns.length === 0) return;
    downloadFile(
      `docrag-${activeConversation?.title ?? 'chat'}.md`.replace(/[^\w.-]+/g, '-'),
      'text/markdown',
      chatToMarkdown(turns),
    );
  }, [turns, activeConversation]);

  const exportJson = useCallback(() => {
    if (turns.length === 0) return;
    downloadFile(
      `docrag-${activeConversation?.title ?? 'chat'}.json`.replace(/[^\w.-]+/g, '-'),
      'application/json',
      chatToJson(turns),
    );
  }, [turns, activeConversation]);

  const commands: Command[] = useMemo(
    () => [
      {
        id: 'new-conversation',
        glyph: '＋',
        label: 'New conversation',
        shortcut: '⌘N',
        run: () => {
          setRail('chat');
          newConversation();
        },
        disabled: pending,
      },
      {
        id: 'upload',
        glyph: '↑',
        label: 'Upload a document',
        shortcut: '⌘U',
        run: () => {
          setRail('corpus');
          // The panel has to render before its hidden file input can be clicked.
          requestAnimationFrame(() => corpusBrowseInputRef.current?.click());
        },
      },
      {
        id: 'settings',
        glyph: '⚙',
        label: 'Open settings',
        shortcut: '⌘,',
        run: () => setSettingsOpen(true),
        disabled: !config,
      },
      {
        id: 'toggle-inspector',
        glyph: '◧',
        label: inspectorOpen ? 'Hide inspector' : 'Show inspector',
        run: () => {
          setInspectorForced(true);
          setInspectorOpen((prev) => !prev);
        },
      },
      {
        id: 'toggle-mode',
        glyph: '◑',
        label: mode === 'engineer' ? 'Switch to Reader mode' : 'Switch to Engineer mode',
        run: () => setMode(mode === 'engineer' ? 'reader' : 'engineer'),
      },
      {
        id: 'toggle-theme',
        glyph: theme === 'dark' ? '☀' : '☾',
        label: theme === 'dark' ? 'Use light theme' : 'Use dark theme',
        shortcut: '⌘J',
        run: toggleTheme,
      },
      {
        id: 'export',
        glyph: '⇩',
        label: 'Export conversation as Markdown',
        run: exportMarkdown,
        disabled: turns.length === 0,
      },
      {
        id: 'export-json',
        glyph: '⇩',
        label: 'Export conversation as JSON',
        run: exportJson,
        disabled: turns.length === 0,
      },
      {
        id: 'traces',
        glyph: '◔',
        label: 'Browse traces',
        run: () => setRail('traces'),
      },
      {
        id: 'clear',
        glyph: '✕',
        label: 'Clear this conversation',
        run: clear,
        disabled: pending || turns.length === 0,
      },
    ],
    // toggleTheme/clear/newConversation are stable enough for a menu rebuilt on open.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [pending, config, inspectorOpen, mode, theme, turns.length, exportMarkdown, exportJson],
  );

  const handlePanelAction = () => {
    if (rail === 'chat') newConversation();
    else if (rail === 'corpus') corpusBrowseInputRef.current?.click();
    else loadTraces();
  };

  return (
    <div className="app-shell">
      <IconRail
        active={rail}
        onSelect={setRail}
        theme={theme}
        onToggleTheme={toggleTheme}
        onOpenSettings={() => setSettingsOpen(true)}
      />
      <ContextPanel
        title={PANEL_META[rail].title}
        actionLabel={PANEL_META[rail].actionLabel}
        onAction={handlePanelAction}
        apiStatus={apiStatus}
        chunkTotal={chunkTotal}
      >
        {rail === 'chat' && (
          <ConversationList
            conversations={conversations}
            activeId={activeConversationId}
            onSwitch={switchConversation}
            onRename={renameConversation}
            onDelete={deleteConversation}
            disabled={pending}
          />
        )}
        {rail === 'corpus' && (
          <CorpusPanel
            filenames={indexedFilenames}
            chunkCounts={chunkCounts}
            pageCounts={pageCounts}
            byteSizes={byteSizes}
            uploadedAts={uploadedAts}
            uploads={uploads}
            onUpload={handleUpload}
            onDelete={handleDeleteDocument}
            maxUploadBytes={config?.max_upload_bytes}
            disabled={pending}
            browseInputRef={corpusBrowseInputRef}
          />
        )}
        {rail === 'traces' && (
          <TracesPanel traces={traces} error={tracesError} onSelect={handleSelectTrace} />
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
              title={activeConversation?.title ?? 'New chat'}
              scopeLabel={scopeLabel(indexedFilenames, selectedFilenames)}
              mode={mode}
              onSetMode={setMode}
              turns={turns}
              onClear={clear}
              disabled={pending}
              onOpenPalette={() => setPaletteOpen(true)}
              inspectorOpen={inspectorOpen}
              onToggleInspector={() => {
                // Reopening on a new turn should follow the conversation again rather
                // than resurface whatever trace row was last clicked.
                if (!inspectorOpen) setPinnedTraceId(null);
                setInspectorForced(true);
                setInspectorOpen((prev) => !prev);
              }}
            />
            <ChatThread
              turns={turns}
              pending={pending}
              stage={stage}
              engineerMode={mode === 'engineer'}
              currentModelLabel={currentModelLabel}
              onRetry={askQuestion}
              feedbackEnabled={config?.feedback_enabled}
              onFeedback={handleFeedback}
              onOpenSource={handleOpenSource}
              onCitationHover={setHoveredCitation}
              onCitationLeave={handleCitationLeave}
            />
            {hoveredCitation && <SourcePreview citation={hoveredCitation} />}
            <ToastRow toasts={toasts} onDismiss={dismissToast} />
            <Composer
              onSubmit={askQuestion}
              disabled={!apiReachable || pending || noDocs}
              hint={composerHint}
              pending={pending}
              onCancel={cancel}
              scopeLabel={scopeLabel(indexedFilenames, selectedFilenames)}
              indexedFilenames={indexedFilenames}
              selectedFilenames={selectedFilenames}
              onSelectedFilenamesChange={setSelectedFilenames}
              config={config}
              provider={selectedProvider}
              onProviderChange={setSelectedProvider}
              providerLabel={currentModelLabel ?? selectedProvider}
            />
          </>
        )}
      </main>
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
          onSaved={() => pushToast({ tone: 'good', title: 'Settings saved' })}
          disabled={pending}
        />
      )}
      {sourceView && (
        <DocumentViewer
          filename={sourceView.filename}
          chunkId={sourceView.chunkId}
          onClose={() => setSourceView(null)}
        />
      )}
    </div>
  );
}

export default App;
