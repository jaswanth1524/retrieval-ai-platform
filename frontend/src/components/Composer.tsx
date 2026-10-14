import { memo, useEffect, useMemo, useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';
import type { LlmProvider, PublicConfigResponse } from '../api/types';
import './Composer.css';
import { isImeComposing } from '../utils/keyboard';
import { readDraft, writeDraft } from '../utils/drafts';
import { QUESTION_INPUT_ID } from '../utils/id';
import { MAX_QUESTION_CHARS } from '../utils/limits';
import { isProviderAllowed, isProviderUsable } from '../utils/providers';

// The most tags and files one question may be scoped to (api/schemas.py's
// MAX_TAGS_PER_DOCUMENT and REQUEST_FILENAMES_MAX): one more was a 422 on every
// question and every passage search, with nothing saying why.
const MAX_SCOPE_TAGS = 20;
const MAX_SCOPE_FILENAMES = 500;

interface ComposerProps {
  onSubmit: (question: string) => void;
  disabled: boolean;
  /** The question can be typed but not sent yet (the app is still connecting). */
  sendHeld?: boolean;
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
  /** Every tag in use across the corpus. */
  availableTags?: string[];
  selectedTags?: string[];
  onSelectedTagsChange?: (tags: string[]) => void;
  /** Which conversation's unsent draft the box holds (kept across switches and reloads). */
  draftKey?: string;
  /** Text to put in the box (an example question); a new nonce puts it in again. */
  prefill?: { text: string; nonce: number } | null;
}

// Typing is saved this long after the last keystroke.
const DRAFT_SAVE_DELAY_MS = 300;

function isCoarsePointer(): boolean {
  return typeof window.matchMedia === 'function' && window.matchMedia('(pointer: coarse)').matches;
}

function Composer({
  onSubmit,
  disabled,
  sendHeld = false,
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
  availableTags = [],
  selectedTags = [],
  onSelectedTagsChange,
  draftKey,
  prefill,
}: ComposerProps) {
  const [value, setValue] = useState(() => (draftKey ? readDraft(draftKey) : ''));
  // A paste the length limit cut short: maxLength drops the excess silently, and a
  // long question pasted from elsewhere lost its end without a word.
  const [pasteCut, setPasteCut] = useState(false);
  // Each conversation keeps its own draft: on a switch the text typed so far is saved
  // under the conversation it was typed in, and the next one's draft comes back.
  const valueRef = useRef(value);
  valueRef.current = value;
  useEffect(() => {
    if (!draftKey) return;
    setValue(readDraft(draftKey));
    const save = () => writeDraft(draftKey, valueRef.current);
    // A reload inside the save delay would otherwise lose the last few keystrokes.
    window.addEventListener('pagehide', save);
    return () => {
      window.removeEventListener('pagehide', save);
      save();
    };
  }, [draftKey]);
  useEffect(() => {
    if (!draftKey) return;
    const timer = setTimeout(() => writeDraft(draftKey, value), DRAFT_SAVE_DELAY_MS);
    return () => clearTimeout(timer);
  }, [draftKey, value]);
  // Put in to edit or send, never sent by itself; focused so Enter sends it as is.
  useEffect(() => {
    if (!prefill) return;
    setValue(prefill.text);
    textareaRef.current?.focus();
  }, [prefill]);
  const [openPopover, setOpenPopover] = useState<'scope' | 'provider' | null>(null);
  const [scopeQuery, setScopeQuery] = useState('');
  const wrapRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  // Hand focus back when an answer settles (a click on Stop moved it to the button), so
  // the next question can be typed straight away. Not on touch screens: focusing there
  // opens the on-screen keyboard over the answer that just arrived.
  const wasPendingRef = useRef(pending);
  useEffect(() => {
    if (wasPendingRef.current && !pending && !disabled && !isCoarsePointer()) {
      textareaRef.current?.focus();
    }
    wasPendingRef.current = pending;
  }, [pending, disabled]);
  // "/" anywhere outside a text field jumps to the question box, as in most chat apps.
  useEffect(() => {
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.key !== '/' || event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      if (
        target?.closest('input, textarea, select, [contenteditable="true"], [role="dialog"]') ||
        document.querySelector('[aria-modal="true"]')
      ) {
        return;
      }
      event.preventDefault();
      textareaRef.current?.focus();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);
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

  // The textarea stays editable while an answer streams (it can take a minute or two);
  // only sending is held until it finishes.
  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled || sendHeld || pending) return;
    onSubmit(trimmed);
    setValue('');
    if (draftKey) writeDraft(draftKey, '');
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter while an IME is composing (Japanese, Chinese, Korean) confirms the candidate;
    // treating it as "send" submitted half-typed questions. 229 is Safari's composing code.
    if (isImeComposing(event)) return;
    // Esc in the question box stops the answer being written, like the Stop button.
    if (event.key === 'Escape' && pending && !openPopover) {
      event.preventDefault();
      onCancel();
      return;
    }
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  // Sets for the per-row checks: up to 500 selected files against every listed one was
  // a quadratic scan on each render of the popover.
  const selectedFileSet = useMemo(() => new Set(selectedFilenames), [selectedFilenames]);
  const selectedTagSet = useMemo(() => new Set(selectedTags), [selectedTags]);

  const toggleTag = (tag: string) => {
    onSelectedTagsChange?.(
      selectedTagSet.has(tag)
        ? selectedTags.filter((name) => name !== tag)
        : [...selectedTags, tag],
    );
  };

  const toggleFilename = (filename: string) => {
    onSelectedFilenamesChange(
      selectedFileSet.has(filename)
        ? selectedFilenames.filter((name) => name !== filename)
        : [...selectedFilenames, filename],
    );
  };

  // ALLOWED_REQUEST_PROVIDERS: a provider the server refuses per request is shown
  // disabled rather than offered and turned away with a 400.
  const allowed = (name: string) => !config || isProviderAllowed(name, config);
  const openaiEnabled = config !== null && isProviderUsable('openai', config);
  const ollamaEnabled = config !== null && isProviderUsable('ollama', config);
  const compatibleAllowed = allowed('openai_compatible');

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
            selectedFileSet.has(filename),
        );

  const tagsFull = selectedTags.length >= MAX_SCOPE_TAGS;
  const filenamesFull = selectedFilenames.length >= MAX_SCOPE_FILENAMES;

  return (
    <div className="composer" ref={wrapRef}>
      {hint && (
        <p className="composer__hint" data-testid="question-hint">
          {hint}
        </p>
      )}
      <p className="visually-hidden" role="status" data-testid="question-length-status">
        {pasteCut ? `Pasted text cut to ${MAX_QUESTION_CHARS} characters, the limit for a question.` : ''}
      </p>
      {pasteCut && (
        <p className="composer__hint composer__hint--warn" data-testid="question-length-note" aria-hidden="true">
          Pasted text was cut to {MAX_QUESTION_CHARS.toLocaleString()} characters, the limit for a question.
        </p>
      )}
      <div className="composer__box">
        <textarea
          ref={textareaRef}
          id={QUESTION_INPUT_ID}
          className="composer__textarea"
          value={value}
          onChange={(event) => {
            setValue(event.target.value);
            if (event.target.value.length < MAX_QUESTION_CHARS) setPasteCut(false);
          }}
          onPaste={(event) => {
            const target = event.currentTarget;
            const kept = target.value.length - (target.selectionEnd - target.selectionStart);
            setPasteCut(kept + event.clipboardData.getData('text').length > MAX_QUESTION_CHARS);
          }}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your documents…"
          aria-label="Ask a question about your documents"
          disabled={disabled}
          rows={2}
          maxLength={MAX_QUESTION_CHARS}
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
                    checked={selectedFilenames.length === 0 && selectedTags.length === 0}
                    onChange={() => {
                      onSelectedFilenamesChange([]);
                      onSelectedTagsChange?.([]);
                    }}
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
                {tagsFull && (
                  <p className="composer__popover-note" data-testid="scope-tags-full">
                    Up to {MAX_SCOPE_TAGS} tags per question.
                  </p>
                )}
                {filenamesFull && (
                  <p className="composer__popover-note" data-testid="scope-files-full">
                    Up to {MAX_SCOPE_FILENAMES} documents per question.
                  </p>
                )}
                {availableTags.length > 0 && onSelectedTagsChange && (
                  <div className="composer__popover-tags" role="group" aria-label="Tags">
                    {availableTags.map((tag) => (
                      <label key={tag} className="composer__popover-item" data-testid="scope-tag">
                        <input
                          type="checkbox"
                          checked={selectedTagSet.has(tag)}
                          disabled={tagsFull && !selectedTagSet.has(tag)}
                          onChange={() => toggleTag(tag)}
                        />
                        <span>#{tag}</span>
                      </label>
                    ))}
                  </div>
                )}
                {visibleFilenames.map((filename) => (
                  <label key={filename} className="composer__popover-item" data-testid="scope-item">
                    <input
                      type="checkbox"
                      checked={selectedFileSet.has(filename)}
                      disabled={filenamesFull && !selectedFileSet.has(filename)}
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
                {!ollamaEnabled && (
                  <p className="composer__popover-note">
                    {allowed('ollama') ? "Ollama isn't reachable." : 'Not allowed on this server.'}
                  </p>
                )}
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
                {!openaiEnabled && (
                  <p className="composer__popover-note">
                    {allowed('openai') ? 'Set OPENAI_API_KEY to enable.' : 'Not allowed on this server.'}
                  </p>
                )}
                {config.openai_compatible_available && (
                  <label className="composer__popover-item">
                    <input
                      type="radio"
                      name="composer-provider"
                      checked={provider === 'openai_compatible'}
                      disabled={!compatibleAllowed}
                      onChange={() => onProviderChange('openai_compatible')}
                      data-testid="provider-openai-compatible"
                    />
                    <span>Self-hosted &middot; {config.openai_compatible_model}</span>
                  </label>
                )}
              </div>
            )}
          </div>
          <span className="composer__spacer" />
          <span className="composer__key-hint">⏎ send &middot; ⇧⏎ newline</span>
          <button
            type="button"
            className="composer__send"
            onClick={pending ? onCancel : submit}
            disabled={pending ? false : disabled || sendHeld || !value.trim()}
            data-testid="question-submit"
          >
            {pending ? 'Stop' : 'Ask'}
          </button>
        </div>
      </div>
    </div>
  );
}

// Memoized: App re-renders on every streamed frame, and none of this depends on it.
export default memo(Composer);
