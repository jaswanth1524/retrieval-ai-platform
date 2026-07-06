import { useEffect, useState } from 'react';
import { ApiClientError, api } from './api/client';
import type { PublicConfigResponse } from './api/types';
import ChatThread from './components/ChatThread';
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
  const { turns, pending, ask } = useChat();

  useEffect(() => {
    let cancelled = false;

    async function checkHealth() {
      try {
        await api.health();
        const configResult = await api.config();
        if (cancelled) return;
        setConfig(configResult);
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
      />
      <main className="app-main">
        {apiStatus === 'error' ? (
          <div className="app-main__unreachable">
            Cannot reach the DocRAG API &mdash; start the API service and refresh.
          </div>
        ) : (
          <>
            <ChatThread turns={turns} pending={pending} />
            <QuestionInput onSubmit={ask} disabled={!apiReachable || pending} />
          </>
        )}
      </main>
    </div>
  );
}

export default App;
