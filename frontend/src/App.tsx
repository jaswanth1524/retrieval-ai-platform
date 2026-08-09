import { useEffect, useRef, useState } from 'react';
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
import Composer from './components/Composer';
import ConfigPanel from './components/ConfigPanel';
import ContextPanel from './components/ContextPanel';
import ConversationList from './components/ConversationList';
import CorpusPanel from './components/CorpusPanel';
import type { UploadItem } from './components/CorpusPanel';
import DocumentViewer from './components/DocumentViewer';
import IconRail from './components/IconRail';
import type { RailPanel } from './components/IconRail';
import SourcePreview from './components/SourcePreview';
import TracesPanel from './components/TracesPanel';
import { useChat } from './hooks/useChat';

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
  const [theme, setTheme] = useState<'dark' | 'light'>(
    () => (document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'),
  );
  const [mode, setModeState] = useState<ChatMode>(loadPersistedMode);
  const [rail, setRail] = useState<RailPanel>('chat');
  const [traces, setTraces] = useState<TraceSummaryResponse[] | null>(null);
  const [tracesError, setTracesError] = useState<string | null>(null);
  // Plumbing for stage 3's inspector — nothing renders on this yet.
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [hoveredCitation, setHoveredCitation] = useState<CitationResponse | null>(null);
  // Source document viewer: which document/chunk a clicked citation opens (null = closed).
  const [sourceView, setSourceView] = useState<{ filename: string; chunkId: string } | null>(null);
  const corpusBrowseInputRef = useRef<HTMLInputElement>(null);
  const {
    turns,
    pending,
    ask,
    cancel,
    clear,
    conversations,
    activeConversationId,
    newConversation,
    switchConversation,
    renameConversation,
    deleteConversation,
    persistError,
    persistPartial,
  } = useChat();

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
        } catch {
          // Non-fatal — the corpus panel just stays empty until the next refresh.
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

  // Opens the inspector on the Trace tab in stage 3; the inspector doesn't exist yet.
  const handleSelectTrace = (traceId: string) => {
    void traceId;
  };

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
      // Chunk counts live server-side; a targeted refetch keeps the footer total and
      // this document's card accurate without threading the count through the job poll.
      try {
        await refreshDocuments();
      } catch {
        // Non-fatal — counts just stay stale until the next refresh.
      }
    } catch (err) {
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

  // Shared by both the input box and a Retry click on a failed turn — retry always
  // uses the CURRENT provider/overrides/scope, not whatever was selected when the
  // original question failed. No selection = search the whole corpus (undefined).
  const askQuestion = (question: string) =>
    ask(
      question,
      selectedProvider,
      advancedOptions,
      selectedFilenames.length > 0 ? selectedFilenames : undefined,
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
        onOpenSettings={() => {
          /* settings modal lands in stage 4 */
        }}
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
              onOpenPalette={() => {
                /* command palette lands in stage 4 */
              }}
              inspectorOpen={inspectorOpen}
              onToggleInspector={() => setInspectorOpen((prev) => !prev)}
            />
            {persistError && (
              <div className="app-main__persist-warning" role="alert" data-testid="persist-error-banner">
                Chat history couldn&apos;t be saved to this browser (storage may be full) &mdash;
                this session won&apos;t be there after a reload.
              </div>
            )}
            {!persistError && persistPartial && (
              <div
                className="app-main__persist-warning"
                role="alert"
                data-testid="persist-partial-banner"
              >
                Browser storage is full, so only this conversation was saved &mdash; your other
                conversations won&apos;t be there after a reload. Export anything you need to keep.
              </div>
            )}
            {config && (
              // Collapsed by default: ConfigPanel is a temporary placement pending
              // stage 4's settings modal, and its full metrics grid is tall enough to
              // squeeze the chat thread down to a sliver if left open in normal flow.
              <details className="app-main__config-disclosure">
                <summary>Advanced settings</summary>
                <ConfigPanel
                  config={config}
                  overrides={advancedOptions}
                  onOverridesChange={updateAdvancedOptions}
                  disabled={pending}
                />
              </details>
            )}
            <ChatThread
              turns={turns}
              pending={pending}
              engineerMode={mode === 'engineer'}
              currentModelLabel={currentModelLabel}
              onRetry={askQuestion}
              onOpenSource={(filename, chunkId) => setSourceView({ filename, chunkId })}
              onCitationHover={setHoveredCitation}
              onCitationLeave={() => setHoveredCitation(null)}
            />
            {hoveredCitation && <SourcePreview citation={hoveredCitation} />}
            <Composer
              onSubmit={askQuestion}
              disabled={!apiReachable || pending || noDocs}
              hint={noDocs ? 'Upload a document to start.' : undefined}
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
