import type { Command } from './CommandPalette';
import { useDialog } from '../hooks/useDialog';
import { PLAIN_KEYS } from '../hooks/useCommands';
import { shortcutLabel } from '../utils/platform';
import './SummaryDialog.css';
import './ShortcutsDialog.css';

interface ShortcutsDialogProps {
  commands: Command[];
  onClose: () => void;
}

/** Every keyboard shortcut: the palette commands that have one, and the plain keys. */
export default function ShortcutsDialog({ commands, onClose }: ShortcutsDialogProps) {
  const dialogRef = useDialog(true, onClose);
  const bound = commands.filter((command) => command.shortcut && command.shortcut !== '?');
  return (
    <div
      className="summary-dialog__backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        className="summary-dialog shortcuts-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
        ref={dialogRef}
        data-testid="shortcuts-dialog"
      >
        <h2 className="summary-dialog__title">Keyboard shortcuts</h2>
        <table className="shortcuts-dialog__table">
          <tbody>
            <tr>
              <td className="mono">{shortcutLabel('⌘K')}</td>
              <td>Command palette</td>
            </tr>
            {bound.map((command) => (
              <tr key={command.id}>
                <td className="mono">{shortcutLabel(command.shortcut ?? '')}</td>
                <td>{command.label}</td>
              </tr>
            ))}
            {PLAIN_KEYS.map(([keys, what]) => (
              <tr key={keys}>
                <td className="mono">{keys}</td>
                <td>{what}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="summary-dialog__actions">
          <button type="button" className="summary-dialog__primary" onClick={onClose} autoFocus>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
