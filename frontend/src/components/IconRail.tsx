import { memo } from 'react';
import './IconRail.css';

export type RailPanel = 'chat' | 'corpus' | 'search' | 'traces' | 'feedback';

interface RailItem {
  key: RailPanel;
  glyph: string;
  label: string;
}

const RAIL_ITEMS: RailItem[] = [
  { key: 'chat', glyph: '☰', label: 'Conversations' },
  { key: 'corpus', glyph: '▤', label: 'Corpus' },
  { key: 'search', glyph: '⌕', label: 'Search passages' },
  { key: 'traces', glyph: '◔', label: 'Traces' },
  { key: 'feedback', glyph: '★', label: 'Feedback' },
];

interface IconRailProps {
  active: RailPanel;
  onSelect: (panel: RailPanel) => void;
  theme: 'dark' | 'light';
  onToggleTheme: () => void;
  onOpenSettings: () => void;
  /** False until the server's config has loaded: Settings has nothing to show yet. */
  settingsAvailable?: boolean;
  /** Rail entries to leave out (the feedback panel when feedback is off or read-only). */
  hidden?: RailPanel[];
}

function IconRail({
  active,
  onSelect,
  theme,
  onToggleTheme,
  onOpenSettings,
  settingsAvailable = true,
  hidden = [],
}: IconRailProps) {
  return (
    <nav className="icon-rail" aria-label="Primary">
      <div className="icon-rail__logo-slot">
        <span className="icon-rail__logo" role="img" aria-label="DocRAG">
          DR
        </span>
      </div>
      {RAIL_ITEMS.filter((item) => !hidden.includes(item.key)).map((item) => (
        <button
          key={item.key}
          type="button"
          title={item.label}
          aria-label={item.label}
          aria-pressed={active === item.key}
          className={`icon-rail__button${active === item.key ? ' icon-rail__button--active' : ''}`}
          onClick={() => onSelect(item.key)}
          data-testid={`rail-${item.key}`}
        >
          <span aria-hidden="true">{item.glyph}</span>
        </button>
      ))}
      <div className="icon-rail__spacer" />
      <button
        type="button"
        title="Toggle theme"
        aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
        className="icon-rail__button"
        onClick={onToggleTheme}
        data-testid="theme-toggle"
      >
        <span aria-hidden="true">{theme === 'dark' ? '☾' : '☀'}</span>
      </button>
      <button
        type="button"
        title={settingsAvailable ? 'Settings' : 'Settings — available once the server answers'}
        aria-label="Settings"
        className="icon-rail__button"
        onClick={onOpenSettings}
        disabled={!settingsAvailable}
        data-testid="open-settings"
      >
        <span aria-hidden="true">⚙</span>
      </button>
    </nav>
  );
}

// Memoized: App re-renders on every streamed frame, and none of this depends on it.
export default memo(IconRail);
