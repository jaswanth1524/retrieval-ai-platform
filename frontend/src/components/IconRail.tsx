import { memo } from 'react';
import './IconRail.css';

export type RailPanel =
  | 'chat'
  | 'saved'
  | 'review'
  | 'corpus'
  | 'search'
  | 'traces'
  | 'feedback'
  | 'unanswered';

interface RailItem {
  key: RailPanel;
  glyph: string;
  label: string;
}

const RAIL_ITEMS: RailItem[] = [
  { key: 'chat', glyph: '☰', label: 'Conversations' },
  { key: 'saved', glyph: '☆', label: 'Saved answers' },
  { key: 'review', glyph: '⚑', label: 'Needs review' },
  { key: 'corpus', glyph: '▤', label: 'Corpus' },
  { key: 'search', glyph: '⌕', label: 'Search passages' },
  { key: 'traces', glyph: '◔', label: 'Traces' },
  { key: 'feedback', glyph: '★', label: 'Feedback' },
  { key: 'unanswered', glyph: '⁇', label: 'Asked questions' },
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
  /** A count shown on an entry (answers still to review). */
  badges?: Partial<Record<RailPanel, number>>;
}

function IconRail({
  active,
  onSelect,
  theme,
  onToggleTheme,
  onOpenSettings,
  settingsAvailable = true,
  hidden = [],
  badges = {},
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
          aria-label={badges[item.key] ? `${item.label} (${badges[item.key]})` : item.label}
          aria-pressed={active === item.key}
          className={`icon-rail__button${active === item.key ? ' icon-rail__button--active' : ''}`}
          onClick={() => onSelect(item.key)}
          data-testid={`rail-${item.key}`}
        >
          <span aria-hidden="true">{item.glyph}</span>
          {badges[item.key] ? (
            <span className="icon-rail__badge" aria-hidden="true" data-testid={`rail-badge-${item.key}`}>
              {Math.min(badges[item.key] ?? 0, 99)}
            </span>
          ) : null}
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
