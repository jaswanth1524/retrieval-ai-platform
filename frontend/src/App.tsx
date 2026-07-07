import { useEffect, useState } from 'react';
import { ApiClientError, api } from './api/client';
import type { LlmProvider, PublicConfigResponse } from './api/types';
import ChatThread from './components/ChatThread';
import ProviderSelector from './components/ProviderSelector';
import QuestionInput from './components/QuestionInput';
import Sidebar from './components/Sidebar';
import type { ApiStatus } from './components/StatusBadge';
import type { UploadState } from './components/UploadPanel';
import { useChat } from './hooks/useChat';

function App() {
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking');
  const [apiStatusMessage, setApiStatusMessage] = useState<string | undefined>();
  const [config, setConfig] = useState<PublicConfigResponse | null>(null);
  const [uploadState, setUploadState] = useState<UploadState>({ status: 'idle' });
  const [selectedProvider, setSelectedProvider] = useState<LlmProvider>('ollama');
  const [theme, setTheme] = useState<'dark' | 'light'>(
    () => (document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'),
  );
  const { turns, pending, ask, cancel } = useChat();

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
        setApiStatus('ok');
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

  const handleUpload = async (file: File) => {
    setUploadState({ status: 'uploading' });
    try {
      const result = await api.uploadDocument(file);
      setUploadState({ status: 'success', result });
    } catch (err) {
      setUploadState({
        status: 'error',
        error: err instanceof ApiClientError ? err.message : 'Upload failed.',
      });
    }
  };

  const apiReachable = apiStatus === 'ok';

  return (
    <div className="app-shell">
      <Sidebar
        apiStatus={apiStatus}
        apiStatusMessage={apiStatusMessage}
        config={config}
        onUpload={handleUpload}
        uploadState={uploadState}
        onFileSelected={() => setUploadState({ status: 'idle' })}
        theme={theme}
        onToggleTheme={toggleTheme}
      />
      <main className="app-main">
        {apiStatus === 'error' ? (
          <div className="app-main__unreachable" role="alert">
            Cannot reach the DocRAG API &mdash; start the API service and refresh.
          </div>
        ) : (
          <>
            {config && (
              <ProviderSelector
                config={config}
                value={selectedProvider}
                onChange={setSelectedProvider}
                disabled={pending}
              />
            )}
            <ChatThread turns={turns} pending={pending} onCancel={cancel} />
            <QuestionInput
              onSubmit={(question) => ask(question, selectedProvider)}
              disabled={!apiReachable || pending}
            />
          </>
        )}
      </main>
    </div>
  );
}

export default App;
