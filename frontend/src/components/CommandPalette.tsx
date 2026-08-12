import { useEffect, useMemo, useRef, useState } from 'react';
import { useDialog } from '../hooks/useDialog';
import './CommandPalette.css';

export interface Command {
  id: string;
  label: string;
  glyph: string;
  shortcut?: string;
  run: () => void;
  /** Commands that cannot act right now are listed but not runnable. */
  disabled?: boolean;
}

interface CommandPaletteProps {
  commands: Command[];
  onClose: () => void;
}

function CommandPalette({ commands, onClose }: CommandPaletteProps) {
  const dialogRef = useDialog(true, onClose);
  const inputRef = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState('');
  const [highlighted, setHighlighted] = useState(0);

  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return commands;
    return commands.filter((command) => command.label.toLowerCase().includes(needle));
  }, [commands, query]);

  // Typing shrinks the list, so an index from the previous query can point past its end.
  useEffect(() => {
    setHighlighted(0);
  }, [query]);

  // The search input owns focus, not the first result — useDialog focuses the first
  // focusable element, which is this input, and typing must go straight to it.
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const run = (command: Command | undefined) => {
    if (!command || command.disabled) return;
    // Close first: several commands move focus themselves (opening settings, switching
    // panels), and useDialog's focus restore would otherwise fight them.
    onClose();
    command.run();
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setHighlighted((prev) => (matches.length === 0 ? 0 : (prev + 1) % matches.length));
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      setHighlighted((prev) =>
        matches.length === 0 ? 0 : (prev - 1 + matches.length) % matches.length,
      );
    } else if (event.key === 'Enter') {
      event.preventDefault();
      run(matches[highlighted]);
    }
  };

  return (
    <div
      className="palette__backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
      data-testid="palette-backdrop"
    >
      <div
        className="palette"
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        ref={dialogRef}
        onKeyDown={handleKeyDown}
        data-testid="command-palette"
      >
        <div className="palette__search">
          <span className="palette__search-glyph" aria-hidden="true">
            ⌕
          </span>
          <input
            ref={inputRef}
            className="palette__input"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search or run a command"
            aria-label="Search commands"
            // The listbox is rendered below; announce the active option to screen readers.
            role="combobox"
            aria-expanded="true"
            aria-controls="palette-results"
            aria-activedescendant={matches[highlighted] ? `palette-cmd-${matches[highlighted].id}` : undefined}
            data-testid="palette-input"
          />
          <span className="palette__esc mono">esc</span>
        </div>

        <ul className="palette__results" id="palette-results" role="listbox" aria-label="Commands">
          {matches.length === 0 && (
            <li className="palette__empty" data-testid="palette-empty">
              No matching command.
            </li>
          )}
          {matches.map((command, index) => (
            <li key={command.id} role="none">
              <button
                type="button"
                id={`palette-cmd-${command.id}`}
                role="option"
                aria-selected={index === highlighted}
                className={`palette__row${index === highlighted ? ' palette__row--active' : ''}`}
                disabled={command.disabled}
                onMouseEnter={() => setHighlighted(index)}
                onClick={() => run(command)}
                data-testid="palette-row"
              >
                <span className="palette__glyph" aria-hidden="true">
                  {command.glyph}
                </span>
                <span className="palette__label">{command.label}</span>
                {command.shortcut && (
                  <span className="palette__shortcut mono">{command.shortcut}</span>
                )}
              </button>
            </li>
          ))}
        </ul>

        <div className="palette__footer mono">
          <span>↑↓ navigate</span>
          <span>⏎ run</span>
          <span>⌘K toggle</span>
        </div>
      </div>
    </div>
  );
}

export default CommandPalette;
