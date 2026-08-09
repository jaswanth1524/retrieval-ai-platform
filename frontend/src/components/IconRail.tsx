import './IconRail.css';

export type RailPanel = 'chat' | 'corpus' | 'traces';

interface RailItem {
  key: RailPanel;
  glyph: string;
  label: string;
}

const RAIL_ITEMS: RailItem[] = [
  { key: 'chat', glyph: '☰', label: 'Conversations' },
  { key: 'corpus', glyph: '▤', label: 'Corpus' },
  { key: 'traces', glyph: '◔', label: 'Traces' },
];

interface IconRailProps {
  active: RailPanel;
  onSelect: (panel: RailPanel) => void;
  theme: 'dark' | 'light';
  onToggleTheme: () => void;
  onOpenSettings: () => void;
}

function IconRail({ active, onSelect, theme, onToggleTheme, onOpenSettings }: IconRailProps) {
  return (
    <nav className="icon-rail" aria-label="Primary">
      <div className="icon-rail__logo-slot">
        <span className="icon-rail__logo" role="img" aria-label="DocRAG">
          DR
        </span>
      </div>
      {RAIL_ITEMS.map((item) => (
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
        title="Settings"
        aria-label="Settings"
        className="icon-rail__button"
        onClick={onOpenSettings}
        data-testid="open-settings"
      >
        <span aria-hidden="true">⚙</span>
      </button>
    </nav>
  );
}

export default IconRail;
