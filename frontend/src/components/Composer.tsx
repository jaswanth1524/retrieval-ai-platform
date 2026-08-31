import { useEffect, useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';
import type { LlmProvider, PublicConfigResponse } from '../api/types';
import './Composer.css';

// Matches the backend's QuestionRequest.question max_length in api/schemas.py.
const MAX_QUESTION_LENGTH = 4000;

interface ComposerProps {
  onSubmit: (question: string) => void;
  disabled: boolean;
  hint?: string;
  pending: boolean;
  onCancel: () => void;
  scopeLabel: string;
  indexedFilenames: string[];
  selectedFilenames: string[];
  onSelectedFilenamesChange: (filenames: string[]) => void;
  config: PublicConfigResponse | null;
  provider: LlmProvider;
  onProviderChange: (provider: LlmProvider) => void;
  providerLabel: string;
}

function Composer({
  onSubmit,
  disabled,
  hint,
  pending,
  onCancel,
  scopeLabel,
  indexedFilenames,
  selectedFilenames,
  onSelectedFilenamesChange,
  config,
  provider,
  onProviderChange,
  providerLabel,
}: ComposerProps) {
  const [value, setValue] = useState('');
  const [openPopover, setOpenPopover] = useState<'scope' | 'provider' | null>(null);
  const [scopeQuery, setScopeQuery] = useState('');
  const wrapRef = useRef<HTMLDivElement>(null);
  const scopeButtonRef = useRef<HTMLButtonElement>(null);
  const providerButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!openPopover) return;
    const handleClick = (event: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(event.target as Node)) {
        setOpenPopover(null);
      }
    };
    const handleKey = (event: globalThis.KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      // Return focus to the control that opened it. Escaping a popover otherwise drops
      // focus to the document body, stranding a keyboard user at the top of the page.
      const opener = openPopover === 'scope' ? scopeButtonRef : providerButtonRef;
      setOpenPopover(null);
      opener.current?.focus();
    };
    document.addEventListener('mousedown', handleClick);
    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('mousedown', handleClick);
      document.removeEventListener('keydown', handleKey);
    };
  }, [openPopover]);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setValue('');
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  const toggleFilename = (filename: string) => {
    onSelectedFilenamesChange(
      selectedFilenames.includes(filename)
        ? selectedFilenames.filter((name) => name !== filename)
        : [...selectedFilenames, filename],
    );
  };

  const openaiEnabled = config?.openai_available ?? false;
  const ollamaEnabled = config?.ollama_available ?? false;

  // A search hiding an already-selected filename would make it permanently
  // unselectable (the checkbox list is the only way to remove it from scope), so a
  // selected filename always stays visible regardless of match.
  const trimmedScopeQuery = scopeQuery.trim().toLowerCase();
  const visibleFilenames =
    trimmedScopeQuery === ''
      ? indexedFilenames
      : indexedFilenames.filter(
          (filename) =>
            filename.toLowerCase().includes(trimmedScopeQuery) ||
            selectedFilenames.includes(filename),
        );

  return (
    <div className="composer" ref={wrapRef}>
      {hint && (
        <p className="composer__hint" data-testid="question-hint">
          {hint}
        </p>
      )}
      <div className="composer__box">
        <textarea
          className="composer__textarea"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your documents…"
          aria-label="Ask a question about your documents"
          disabled={disabled}
          rows={2}
          maxLength={MAX_QUESTION_LENGTH}
          data-testid="question-textarea"
        />
        <div className="composer__controls">
          <div className="composer__popover-anchor">
            <button
              type="button"
              className="composer__chip"
              onClick={() => {
                setOpenPopover((prev) => (prev === 'scope' ? null : 'scope'));
                setScopeQuery('');
              }}
              disabled={indexedFilenames.length === 0}
              aria-haspopup="true"
              aria-expanded={openPopover === 'scope'}
              aria-controls="composer-scope-popover"
              ref={scopeButtonRef}
              data-testid="composer-scope-button"
            >
              <span aria-hidden="true">◎</span> {scopeLabel}
            </button>
            {openPopover === 'scope' && (
              <div
                className="composer__popover"
                id="composer-scope-popover"
                role="group"
                aria-label="Search scope"
                data-testid="composer-scope-popover"
              >
                <label className="composer__popover-item">
                  <input
                    type="checkbox"
                    checked={selectedFilenames.length === 0}
                    onChange={() => onSelectedFilenamesChange([])}
                    data-testid="scope-all"
                  />
                  <span>All documents</span>
                </label>
                {indexedFilenames.length > 5 && (
                  <input
                    type="search"
                    className="composer__popover-search"
                    placeholder="Filter documents…"
                    aria-label="Filter documents"
                    value={scopeQuery}
                    onChange={(event) => setScopeQuery(event.target.value)}
                    data-testid="composer-scope-search"
                  />
                )}
                {visibleFilenames.map((filename) => (
                  <label key={filename} className="composer__popover-item" data-testid="scope-item">
                    <input
                      type="checkbox"
                      checked={selectedFilenames.includes(filename)}
                      onChange={() => toggleFilename(filename)}
                    />
                    <span className="composer__popover-filename mono">{filename}</span>
                  </label>
                ))}
              </div>
            )}
          </div>
          <div className="composer__popover-anchor">
            <button
              type="button"
              className="composer__chip"
              onClick={() => setOpenPopover((prev) => (prev === 'provider' ? null : 'provider'))}
              disabled={!config}
              aria-haspopup="true"
              aria-expanded={openPopover === 'provider'}
              aria-controls="composer-provider-popover"
              ref={providerButtonRef}
              data-testid="composer-provider-button"
            >
              <span aria-hidden="true">⚙</span> {providerLabel}
            </button>
            {openPopover === 'provider' && config && (
              <div
                className="composer__popover"
                id="composer-provider-popover"
                role="radiogroup"
                aria-label="Generation provider"
                data-testid="composer-provider-popover"
              >
                <label className="composer__popover-item">
                  <input
                    type="radio"
                    name="composer-provider"
                    checked={provider === 'ollama'}
                    disabled={!ollamaEnabled}
                    onChange={() => onProviderChange('ollama')}
                    data-testid="provider-ollama"
                  />
                  <span>Ollama &middot; {config.llm_model}</span>
                </label>
                {!ollamaEnabled && <p className="composer__popover-note">Ollama isn&apos;t reachable.</p>}
                <label className="composer__popover-item">
                  <input
                    type="radio"
                    name="composer-provider"
                    checked={provider === 'openai'}
                    disabled={!openaiEnabled}
                    onChange={() => onProviderChange('openai')}
                    data-testid="provider-openai"
                  />
                  <span>OpenAI &middot; {config.openai_model}</span>
                </label>
                {!openaiEnabled && <p className="composer__popover-note">Set OPENAI_API_KEY to enable.</p>}
              </div>
            )}
          </div>
          <span className="composer__spacer" />
          <span className="composer__key-hint">⏎ send &middot; ⇧⏎ newline</span>
          <button
            type="button"
            className="composer__send"
            onClick={pending ? onCancel : submit}
            disabled={pending ? false : disabled || !value.trim()}
            data-testid="question-submit"
          >
            {pending ? 'Stop' : 'Ask'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default Composer;
