import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { CHAT_STORAGE_KEY } from '../hooks/chatPersistence';
import { clearChatStore, loadChatStore } from '../hooks/chatStore';
import { downloadFile } from '../utils/exportChat';
import './RootErrorBoundary.css';

interface RootErrorBoundaryState {
  failed: boolean;
}

/** The last resort when rendering the app throws.
 *
 * Saved history is restored on every load, so one unreadable conversation used to
 * crash the page every time it opened, with nothing on screen to recover from. This
 * offers to save the history first, then clear it.
 */
class RootErrorBoundary extends Component<{ children: ReactNode }, RootErrorBoundaryState> {
  state: RootErrorBoundaryState = { failed: false };

  static getDerivedStateFromError(): RootErrorBoundaryState {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('DocRAG failed to render.', error, info.componentStack);
  }

  private exportHistory = async () => {
    // IndexedDB holds the full history; localStorage only the compact copy.
    const full = await loadChatStore();
    let compact: string | null = null;
    try {
      compact = localStorage.getItem(CHAT_STORAGE_KEY);
    } catch {
      compact = null;
    }
    const content = full !== null ? JSON.stringify(full, null, 2) : (compact ?? '{}');
    downloadFile('docrag-history-backup.json', 'application/json', content);
  };

  private clearHistory = async () => {
    try {
      localStorage.removeItem(CHAT_STORAGE_KEY);
    } catch {
      // Nothing more to try; IndexedDB below is the copy that matters.
    }
    await clearChatStore();
    window.location.reload();
  };

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <main className="root-error" role="alert" data-testid="root-error">
        <h1>Something went wrong</h1>
        <p>
          DocRAG could not show this page. If it keeps happening, the saved chat history may
          be the cause: download a copy of it, then clear it and reload.
        </p>
        <div className="root-error__actions">
          <button type="button" onClick={() => window.location.reload()}>
            Reload
          </button>
          <button type="button" onClick={() => void this.exportHistory()} data-testid="root-error-export">
            Download history
          </button>
          <button type="button" onClick={() => void this.clearHistory()} data-testid="root-error-clear">
            Clear history and reload
          </button>
        </div>
      </main>
    );
  }
}

export default RootErrorBoundary;
