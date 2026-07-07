import type { PublicConfigResponse } from '../api/types';
import ConfigPanel from './ConfigPanel';
import StatusBadge, { type ApiStatus } from './StatusBadge';
import UploadPanel, { type UploadState } from './UploadPanel';
import './Sidebar.css';

interface SidebarProps {
  apiStatus: ApiStatus;
  apiStatusMessage?: string;
  config: PublicConfigResponse | null;
  onUpload: (file: File) => Promise<void>;
  uploadState: UploadState;
}

function Sidebar({ apiStatus, apiStatusMessage, config, onUpload, uploadState }: SidebarProps) {
  return (
    <aside className="sidebar">
      <h1 className="sidebar__title">DocRAG</h1>
      <StatusBadge status={apiStatus} message={apiStatusMessage} />
      <UploadPanel onUpload={onUpload} state={uploadState} maxUploadBytes={config?.max_upload_bytes} />
      {config && <ConfigPanel config={config} />}
    </aside>
  );
}

export default Sidebar;
