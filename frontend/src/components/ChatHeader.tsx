import { useState } from 'react';
import type { ChatTurn } from './ChatMessage';
import { chatToMarkdown, downloadFile } from '../utils/exportChat';
import './ChatHeader.css';

export type ChatMode = 'reader' | 'engineer';

interface ChatHeaderProps {
  title: string;
  scopeLabel: string;
  mode: ChatMode;
  onSetMode: (mode: ChatMode) => void;
  turns: ChatTurn[];
  onClear: () => void;
  disabled?: boolean;
  onOpenPalette: () => void;
  inspectorOpen: boolean;
  onToggleInspector: () => void;
}

function ChatHeader({
  title,
  scopeLabel,
  mode,
  onSetMode,
  turns,
  onClear,
  disabled,
  onOpenPalette,
  inspectorOpen,
  onToggleInspector,
}: ChatHeaderProps) {
  const [confirmingClear, setConfirmingClear] = useState(false);

  // Markdown only: the header has one Export button and always passed 'markdown', so
  // the JSON branch was unreachable. JSON export lives in the command palette
  // (App.tsx's exportJson), which is the one implementation of it.
  const handleExport = () => {
    downloadFile('docrag-chat.md', 'text/markdown', chatToMarkdown(turns));
  };

  return (
    <header className="chat-header">
      <h1 className="chat-header__title">{title}</h1>
      <span className="chat-header__scope">{scopeLabel}</span>
      {turns.length > 0 && (
        <div className="chat-header__actions">
          <button
            type="button"
            className="chat-header__action"
            onClick={handleExport}
            data-testid="chat-header-export"
          >
            Export
          </button>
          {confirmingClear ? (
            <span className="chat-header__confirm">
              <button
                type="button"
                className="chat-header__confirm-yes"
                onClick={() => {
                  onClear();
                  setConfirmingClear(false);
                }}
                data-testid="chat-header-clear-confirm"
              >
                Confirm
              </button>
              <button
                type="button"
                className="chat-header__confirm-no"
                onClick={() => setConfirmingClear(false)}
              >
                Cancel
              </button>
            </span>
          ) : (
            <button
              type="button"
              className="chat-header__action"
              onClick={() => setConfirmingClear(true)}
              disabled={disabled}
              data-testid="chat-header-clear"
            >
              Clear
            </button>
          )}
        </div>
      )}
      <div className="chat-header__spacer" />
      <div className="chat-header__mode" role="radiogroup" aria-label="Reader or Engineer mode">
        <button
          type="button"
          role="radio"
          aria-checked={mode === 'reader'}
          className={`chat-header__mode-btn${mode === 'reader' ? ' chat-header__mode-btn--active' : ''}`}
          onClick={() => onSetMode('reader')}
          data-testid="mode-reader"
        >
          Reader
        </button>
        <button
          type="button"
          role="radio"
          aria-checked={mode === 'engineer'}
          className={`chat-header__mode-btn${mode === 'engineer' ? ' chat-header__mode-btn--active' : ''}`}
          onClick={() => onSetMode('engineer')}
          data-testid="mode-engineer"
        >
          Engineer
        </button>
      </div>
      <button type="button" className="chat-header__palette" onClick={onOpenPalette} data-testid="open-palette">
        <span aria-hidden="true">⌕</span>
        <span className="chat-header__palette-label">Search or run a command</span>
        <kbd className="chat-header__kbd">⌘K</kbd>
      </button>
      <button
        type="button"
        className={`chat-header__inspector-toggle${inspectorOpen ? ' chat-header__inspector-toggle--active' : ''}`}
        onClick={onToggleInspector}
        title="Toggle inspector"
        aria-label="Toggle inspector"
        aria-pressed={inspectorOpen}
        data-testid="toggle-inspector"
      >
        <span aria-hidden="true">◧</span>
      </button>
    </header>
  );
}

export default ChatHeader;
