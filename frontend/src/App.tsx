import { useEffect, useState } from 'react';
import { ApiClientError, api } from './api/client';
import type { LlmProvider, PublicConfigResponse, QuestionOverrides } from './api/types';
import ChatThread from './components/ChatThread';
import ChatToolbar from './components/ChatToolbar';
import DocumentViewer from './components/DocumentViewer';
import ProviderSelector from './components/ProviderSelector';
import TraceBrowser from './components/TraceBrowser';
import QuestionInput from './components/QuestionInput';
import Sidebar from './components/Sidebar';
import type { ApiStatus } from './components/StatusBadge';
import type { UploadItem } from './components/UploadPanel';
import { useChat } from './hooks/useChat';

const ADVANCED_OPTIONS_STORAGE_KEY = 'docrag-advanced-options';
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

function App() {
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking');
  const [apiStatusMessage, setApiStatusMessage] = useState<string | undefined>();
  const [config, setConfig] = useState<PublicConfigResponse | null>(null);
  const [uploads, setUploads] = useState<UploadItem[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<LlmProvider>('ollama');
  const [advancedOptions, setAdvancedOptions] = useState<QuestionOverrides>(loadPersistedOverrides);
  const [selectedFilenames, setSelectedFilenames] = useState<string[]>([]);
  // The full corpus currently indexed in Qdrant (across all sessions). Both the
  // corpus-management panel and the search-scope filter operate over this one list,
  // so any indexed document is scopable regardless of which session uploaded it.
  const [indexedFilenames, setIndexedFilenames] = useState<string[]>([]);
  const [theme, setTheme] = useState<'dark' | 'light'>(
    () => (document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'),
  );
  // Sidebar becomes an off-canvas overlay below the responsive breakpoint (see
  // global.css); closed by default so it never covers the chat on first paint.
  const [sidebarOpen, setSidebarOpen] = useState(false);
  // Source document viewer: which document/chunk a clicked citation opens (null = closed).
  const [sourceView, setSourceView] = useState<{ filename: string; chunkId: string } | null>(null);
  const [traceBrowserOpen, setTraceBrowserOpen] = useState(false);
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
          const documents = await api.listDocuments(controller.signal);
          if (!cancelled) setIndexedFilenames(documents.filenames);
        } catch {
          // Non-fatal — the corpus panel just stays empty until the next refresh.
        }
      } catch (err) {
        if (cancelled) return;
        setApiStatus('error');
        setApiStatusMessage(err instanceof ApiClientError ? err.message : 'Unknown error');
      }
    }

    void checkHealth();
    return () => {
      cancelled = true;
      // Previously only the `cancelled` flag guarded setState — the underlying
      // fetch kept running to completion regardless. Aborting it here actually
      // releases the in-flight request instead of just ignoring its result.
      controller.abort();
    };
  }, []);

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
  };

  const apiReachable = apiStatus === 'ok';
  const noDocs = indexedFilenames.length === 0;

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

  return (
    <div className="app-shell">
      {sidebarOpen && (
        <div
          className="app-shell__backdrop"
          data-testid="sidebar-backdrop"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <Sidebar
        apiStatus={apiStatus}
        apiStatusMessage={apiStatusMessage}
        config={config}
        onUpload={handleUpload}
        uploads={uploads}
        theme={theme}
        onToggleTheme={toggleTheme}
        overrides={advancedOptions}
        onOverridesChange={updateAdvancedOptions}
        overridesDisabled={pending}
        selectedFilenames={selectedFilenames}
        onSelectedFilenamesChange={setSelectedFilenames}
        indexedFilenames={indexedFilenames}
        onDeleteDocument={handleDeleteDocument}
        conversations={conversations}
        activeConversationId={activeConversationId}
        onNewConversation={newConversation}
        onSwitchConversation={switchConversation}
        onRenameConversation={renameConversation}
        onDeleteConversation={deleteConversation}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <main className="app-main">
        {apiStatus === 'error' ? (
          <div className="app-main__unreachable" role="alert">
            Cannot reach the DocRAG API &mdash; start the API service and refresh.
          </div>
        ) : (
          <>
            <button
              type="button"
              className="app-main__sidebar-toggle"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open sidebar"
              data-testid="sidebar-open"
            >
              &#9776;
            </button>
            {config && (
              <ProviderSelector
                config={config}
                value={selectedProvider}
                onChange={setSelectedProvider}
                disabled={pending}
              />
            )}
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
            <div className="app-main__toolbar-row">
              <ChatToolbar turns={turns} onClear={clear} disabled={pending} />
              <button
                type="button"
                className="app-main__traces-button"
                onClick={() => setTraceBrowserOpen(true)}
                data-testid="open-trace-browser"
              >
                Traces
              </button>
            </div>
            <ChatThread
              turns={turns}
              pending={pending}
              onCancel={cancel}
              onRetry={askQuestion}
              onOpenSource={(filename, chunkId) => setSourceView({ filename, chunkId })}
            />
            <QuestionInput
              onSubmit={askQuestion}
              disabled={!apiReachable || pending || noDocs}
              hint={noDocs ? 'Upload a document to start.' : undefined}
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
      {traceBrowserOpen && <TraceBrowser onClose={() => setTraceBrowserOpen(false)} />}
    </div>
  );
}

export default App;
