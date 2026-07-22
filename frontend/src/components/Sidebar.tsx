import type { PublicConfigResponse, QuestionOverrides } from '../api/types';
import type { ConversationSummary } from '../hooks/useChat';
import ConfigPanel from './ConfigPanel';
import ConversationList from './ConversationList';
import CorpusPanel from './CorpusPanel';
import DocumentFilter from './DocumentFilter';
import StatusBadge, { type ApiStatus } from './StatusBadge';
import UploadPanel, { type UploadItem } from './UploadPanel';
import './Sidebar.css';

interface SidebarProps {
  apiStatus: ApiStatus;
  apiStatusMessage?: string;
  config: PublicConfigResponse | null;
  onUpload: (files: File[]) => Promise<void>;
  uploads: UploadItem[];
  theme: 'dark' | 'light';
  onToggleTheme: () => void;
  overrides: QuestionOverrides;
  onOverridesChange: (value: QuestionOverrides) => void;
  overridesDisabled?: boolean;
  selectedFilenames: string[];
  onSelectedFilenamesChange: (filenames: string[]) => void;
  indexedFilenames: string[];
  onDeleteDocument: (filename: string) => Promise<void>;
  conversations: ConversationSummary[];
  activeConversationId: string | null;
  onNewConversation: () => void;
  onSwitchConversation: (id: string) => void;
  onRenameConversation: (id: string, title: string) => void;
  onDeleteConversation: (id: string) => void;
  // Off-canvas overlay state used only below the responsive breakpoint (see
  // global.css) — the sidebar is always visible above it regardless of `open`.
  open?: boolean;
  onClose?: () => void;
}

function Sidebar({
  apiStatus,
  apiStatusMessage,
  config,
  onUpload,
  uploads,
  theme,
  onToggleTheme,
  overrides,
  onOverridesChange,
  overridesDisabled,
  selectedFilenames,
  onSelectedFilenamesChange,
  indexedFilenames,
  onDeleteDocument,
  conversations,
  activeConversationId,
  onNewConversation,
  onSwitchConversation,
  onRenameConversation,
  onDeleteConversation,
  open,
  onClose,
}: SidebarProps) {
  return (
    <aside className={`sidebar${open ? ' sidebar--open' : ''}`}>
      <div className="sidebar__header">
        <div className="sidebar__brand">
          <span className="sidebar__logo">D</span>
          <h1 className="sidebar__title">DocRAG</h1>
        </div>
        <button
          type="button"
          className="sidebar__theme-toggle"
          onClick={onToggleTheme}
          aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          data-testid="theme-toggle"
        >
          {theme === 'dark' ? '☀' : '☾'}
        </button>
        {onClose && (
          <button
            type="button"
            className="sidebar__close"
            onClick={onClose}
            aria-label="Close sidebar"
            data-testid="sidebar-close"
          >
            &times;
          </button>
        )}
      </div>
      <StatusBadge status={apiStatus} message={apiStatusMessage} />
      <ConversationList
        conversations={conversations}
        activeId={activeConversationId}
        onNew={onNewConversation}
        onSwitch={onSwitchConversation}
        onRename={onRenameConversation}
        onDelete={onDeleteConversation}
        disabled={overridesDisabled}
      />
      <UploadPanel onUpload={onUpload} uploads={uploads} maxUploadBytes={config?.max_upload_bytes} />
      <CorpusPanel
        filenames={indexedFilenames}
        onDelete={onDeleteDocument}
        disabled={overridesDisabled}
      />
      <DocumentFilter
        filenames={indexedFilenames}
        selected={selectedFilenames}
        onChange={onSelectedFilenamesChange}
        disabled={overridesDisabled}
      />
      {config && (
        <ConfigPanel
          config={config}
          overrides={overrides}
          onOverridesChange={onOverridesChange}
          disabled={overridesDisabled}
        />
      )}
    </aside>
  );
}

export default Sidebar;
