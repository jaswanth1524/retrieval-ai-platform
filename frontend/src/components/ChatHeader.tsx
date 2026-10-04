import { memo, useState } from 'react';
import type { KeyboardEvent } from 'react';
import { shortcutLabel } from '../utils/platform';
import ConfirmInline from './ConfirmInline';
import './ChatHeader.css';

export type ChatMode = 'reader' | 'engineer';

interface ChatHeaderProps {
  title: string;
  scopeLabel: string;
  mode: ChatMode;
  onSetMode: (mode: ChatMode) => void;
  /** Export and Clear only show once the conversation has something in it. A boolean,
   *  not the turns: those change on every streamed token, and this header doesn't. */
  hasTurns: boolean;
  onClear: () => void;
  // Markdown export, named after the conversation (App.tsx's exportMarkdown — the same
  // implementation the command palette runs).
  onExport: () => void;
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
  hasTurns,
  onClear,
  onExport,
  disabled,
  onOpenPalette,
  inspectorOpen,
  onToggleInspector,
}: ChatHeaderProps) {
  // App keys this component by conversation, so a pending confirm can't carry over to
  // a different conversation.
  const [confirmingClear, setConfirmingClear] = useState(false);

  // Arrow keys move between the two modes, as in any radio group; only the checked one
  // is in the tab order.
  const handleModeKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    const next: ChatMode = mode === 'reader' ? 'engineer' : 'reader';
    onSetMode(next);
    event.currentTarget.querySelector<HTMLButtonElement>(`[data-mode="${next}"]`)?.focus();
  };

  return (
    <header className="chat-header">
      <h1 className="chat-header__title">{title}</h1>
      <span className="chat-header__scope">{scopeLabel}</span>
      {hasTurns && (
        <div className="chat-header__actions">
          <button
            type="button"
            className="chat-header__action"
            onClick={onExport}
            data-testid="chat-header-export"
          >
            Export
          </button>
          {confirmingClear ? (
            <ConfirmInline
              tone="main"
              className="chat-header__confirm"
              confirmLabel="Confirm"
              onCancel={() => setConfirmingClear(false)}
              onConfirm={() => {
                onClear();
                setConfirmingClear(false);
              }}
              // useChat ignores a clear while a question is in flight.
              disabled={disabled}
              confirmTestId="chat-header-clear-confirm"
            />
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
      <div
        className="chat-header__mode"
        role="radiogroup"
        aria-label="Reader or Engineer mode"
        onKeyDown={handleModeKeyDown}
      >
        <button
          type="button"
          role="radio"
          aria-checked={mode === 'reader'}
          tabIndex={mode === 'reader' ? 0 : -1}
          data-mode="reader"
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
          tabIndex={mode === 'engineer' ? 0 : -1}
          data-mode="engineer"
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
        <kbd className="chat-header__kbd">{shortcutLabel('⌘K')}</kbd>
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

// Memoized: App re-renders on every streamed token; every prop here is stable then.
export default memo(ChatHeader);
