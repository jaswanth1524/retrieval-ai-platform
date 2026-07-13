import type { PublicConfigResponse, QuestionOverrides } from '../api/types';
import ConfigPanel from './ConfigPanel';
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
  sessionFilenames: string[];
  selectedFilenames: string[];
  onSelectedFilenamesChange: (filenames: string[]) => void;
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
  sessionFilenames,
  selectedFilenames,
  onSelectedFilenamesChange,
}: SidebarProps) {
  return (
    <aside className="sidebar">
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
      </div>
      <StatusBadge status={apiStatus} message={apiStatusMessage} />
      <UploadPanel onUpload={onUpload} uploads={uploads} maxUploadBytes={config?.max_upload_bytes} />
      <DocumentFilter
        filenames={sessionFilenames}
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
