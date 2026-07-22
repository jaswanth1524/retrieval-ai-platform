import { useState } from 'react';
import type { ChatTurn } from './ChatMessage';
import { chatToJson, chatToMarkdown, downloadFile } from '../utils/exportChat';
import './ChatToolbar.css';

interface ChatToolbarProps {
  turns: ChatTurn[];
  onClear: () => void;
  disabled?: boolean;
}

function ChatToolbar({ turns, onClear, disabled }: ChatToolbarProps) {
  const [confirmingClear, setConfirmingClear] = useState(false);

  if (turns.length === 0) return null;

  const handleExport = (format: 'markdown' | 'json') => {
    if (format === 'markdown') {
      downloadFile('docrag-chat.md', 'text/markdown', chatToMarkdown(turns));
    } else {
      downloadFile('docrag-chat.json', 'application/json', chatToJson(turns));
    }
  };

  const handleClear = () => {
    onClear();
    setConfirmingClear(false);
  };

  return (
    <div className="chat-toolbar" data-testid="chat-toolbar">
      <button
        type="button"
        className="chat-toolbar__button"
        onClick={() => handleExport('markdown')}
        data-testid="chat-toolbar-export-markdown"
      >
        Export Markdown
      </button>
      <button
        type="button"
        className="chat-toolbar__button"
        onClick={() => handleExport('json')}
        data-testid="chat-toolbar-export-json"
      >
        Export JSON
      </button>
      {confirmingClear ? (
        <span className="chat-toolbar__confirm">
          Clear chat?
          <button
            type="button"
            className="chat-toolbar__confirm-yes"
            onClick={handleClear}
            data-testid="chat-toolbar-clear-confirm"
          >
            Confirm
          </button>
          <button
            type="button"
            className="chat-toolbar__confirm-no"
            onClick={() => setConfirmingClear(false)}
          >
            Cancel
          </button>
        </span>
      ) : (
        <button
          type="button"
          className="chat-toolbar__button chat-toolbar__button--danger"
          onClick={() => setConfirmingClear(true)}
          disabled={disabled}
          data-testid="chat-toolbar-clear"
        >
          Clear chat
        </button>
      )}
    </div>
  );
}

export default ChatToolbar;
