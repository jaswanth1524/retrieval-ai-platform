import { memo, useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';
import { useLeaveGuard } from '../hooks/useLeaveGuard';
import {
  MAX_TEMPLATE_NAME_CHARS,
  type QuestionTemplate,
  deleteTemplate,
  firstPlaceholder,
  readTemplates,
  saveTemplate,
} from '../utils/questionTemplates';
import type { LlmProvider, PublicConfigResponse } from '../api/types';
import type { ScopePreset } from '../utils/scopePresets';
import ScopePopover from './ScopePopover';
import './Composer.css';
import { isImeComposing } from '../utils/keyboard';
import { enterAddsNewline } from '../utils/platform';
import { readDraft, writeDraft } from '../utils/drafts';
import { QUESTION_INPUT_ID } from '../utils/id';
import { MAX_QUESTION_CHARS } from '../utils/limits';
import { singleKeyShortcutsOn } from '../utils/shortcuts';
import { isProviderAllowed, isProviderUsable } from '../utils/providers';

// The most of the window the question box grows to before it scrolls.
const COMPOSER_MAX_HEIGHT_SHARE = 0.4;

interface ComposerProps {
  onSubmit: (question: string) => void;
  disabled: boolean;
  /** The question can be typed but not sent yet (the app is still connecting). */
  sendHeld?: boolean;
  /** The hold is the app starting up: a send asked for meanwhile goes out when it ends
   *  (Enter pressed right after the page loaded was dropped without a word). */
  sendQueueable?: boolean;
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
  /** A page range within the one scoped document; offered while exactly one is. */
  pageRange?: { min?: number; max?: number };
  onPageRangeChange?: (range: { min?: number; max?: number }) => void;
  /** Only documents uploaded since this time (Unix seconds). */
  uploadedAfter?: number;
  onUploadedAfterChange?: (value: number | undefined) => void;
  /** Only documents uploaded up to the end of a day (Unix seconds). */
  uploadedBefore?: number;
  onUploadedBeforeChange?: (value: number | undefined) => void;
  /** Saved scopes of this workspace: picked to set the scope again. */
  scopePresets?: ScopePreset[];
  onApplyScopePreset?: (preset: ScopePreset) => void;
  /** Save the current scope under a name. */
  onSaveScopePreset?: (name: string) => void;
  onDeleteScopePreset?: (name: string) => void;
  /** The conversation's last question: Up in an empty box puts it back, to edit. */
  recallQuestion?: string;
  /** The workspace asked in: a send queued while offline is dropped when it changes
   *  (it went out to the next workspace, against its documents). */
  workspace?: string;
}

// Scoped to more documents than this, the button's label says how many (no chips).
const SCOPE_CHIPS_MAX = 8;

// Typing is saved this long after the last keystroke.
const DRAFT_SAVE_DELAY_MS = 300;

function isCoarsePointer(): boolean {
  return typeof window.matchMedia === 'function' && window.matchMedia('(pointer: coarse)').matches;
}

function Composer({
  onSubmit,
  disabled,
  sendHeld = false,
  sendQueueable = false,
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
  pageRange,
  onPageRangeChange,
  uploadedAfter,
  onUploadedAfterChange,
  uploadedBefore,
  onUploadedBeforeChange,
  scopePresets = [],
  onApplyScopePreset,
  onSaveScopePreset,
  onDeleteScopePreset,
  recallQuestion,
  workspace,
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
    // A send queued during startup belongs to the conversation it was typed in: going
    // out after a switch, it sent the next conversation's draft there instead. Its text
    // stays saved as that conversation's draft.
    setSendQueued(false);
    setValue(readDraft(draftKey));
    const save = () => writeDraft(draftKey, valueRef.current);
    // A reload inside the save delay would otherwise lose the last few keystrokes.
    window.addEventListener('pagehide', save);
    return () => {
      window.removeEventListener('pagehide', save);
      save();
    };
  }, [draftKey]);
  // A send queued for one workspace never goes out in another: the question stays in
  // the box, to send again where it belongs.
  useEffect(() => {
    setSendQueued(false);
  }, [workspace]);
  useEffect(() => {
    if (!draftKey) return;
    const timer = setTimeout(() => writeDraft(draftKey, value), DRAFT_SAVE_DELAY_MS);
    return () => clearTimeout(timer);
  }, [draftKey, value]);
  // Put in to edit or send, never sent by itself; focused so Enter sends it as is.
  useEffect(() => {
    if (!prefill) return;
    // Beside what is already typed, never over it: a quote picked while drafting a
    // question used to wipe the draft.
    setValue((current) =>
      current.trim() ? `${current.replace(/\s+$/, '')}\n${prefill.text}`.slice(0, MAX_QUESTION_CHARS) : prefill.text,
    );
    textareaRef.current?.focus();
  }, [prefill]);
  const [openPopover, setOpenPopover] = useState<'scope' | 'provider' | 'templates' | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  // A removed chip took the focus with it (to <body>): the chip now in its place, or
  // the one before, or the question box, gets it instead.
  const chipsRef = useRef<HTMLUListElement>(null);
  const focusChipRef = useRef<number | null>(null);
  useEffect(() => {
    const index = focusChipRef.current;
    if (index === null) return;
    focusChipRef.current = null;
    const chips = chipsRef.current?.querySelectorAll<HTMLButtonElement>('button') ?? [];
    const next = chips[Math.min(index, chips.length - 1)];
    if (next) next.focus();
    else textareaRef.current?.focus();
  }, [selectedFilenames, selectedTags]);
  // Grows with what is typed (up to a share of the window, then scrolls): two fixed
  // rows hid all but the end of a long, multi-line question.
  useLayoutEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = 'auto';
    const limit = Math.round(window.innerHeight * COMPOSER_MAX_HEIGHT_SHARE);
    element.style.height = `${Math.min(element.scrollHeight, limit)}px`;
    element.style.overflowY = element.scrollHeight > limit ? 'auto' : 'hidden';
  }, [value]);
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
      if (!singleKeyShortcutsOn()) return; // Settings → Keyboard
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
  const templatesButtonRef = useRef<HTMLButtonElement>(null);
  const openerOf = (which: 'scope' | 'provider' | 'templates') =>
    which === 'scope' ? scopeButtonRef : which === 'provider' ? providerButtonRef : templatesButtonRef;
  // Saved questions (this browser's), and the one being saved: its name so far.
  const [templates, setTemplates] = useState<QuestionTemplate[]>(() => readTemplates());
  const [templateName, setTemplateName] = useState('');
  // A template just put in the box: its first {placeholder} is selected once rendered.
  const selectAfterInsertRef = useRef<[number, number] | null>(null);
  useLayoutEffect(() => {
    const range = selectAfterInsertRef.current;
    if (!range) return;
    selectAfterInsertRef.current = null;
    textareaRef.current?.focus();
    textareaRef.current?.setSelectionRange(range[0], range[1]);
  });
  const insertTemplate = (template: QuestionTemplate) => {
    setValue(template.text);
    const placeholder = firstPlaceholder(template.text);
    selectAfterInsertRef.current = placeholder ?? [template.text.length, template.text.length];
    setOpenPopover(null);
  };

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
      const opener = openerOf(openPopover);
      setOpenPopover(null);
      opener.current?.focus();
    };
    // Tabbing out of the popover (and its button) closes it: it stayed open over the
    // thread while focus went on elsewhere. Focus moving, not blur: a click on a
    // checkbox in Safari focuses nothing, and must not close the list.
    const handleFocus = (event: FocusEvent) => {
      const opener = openerOf(openPopover).current;
      const anchor = opener?.closest('.composer__popover-anchor');
      if (anchor && event.target instanceof Node && !anchor.contains(event.target)) {
        setOpenPopover(null);
      }
    };
    document.addEventListener('mousedown', handleClick);
    document.addEventListener('keydown', handleKey);
    document.addEventListener('focusin', handleFocus);
    return () => {
      document.removeEventListener('mousedown', handleClick);
      document.removeEventListener('keydown', handleKey);
      document.removeEventListener('focusin', handleFocus);
    };
  }, [openPopover]);

  // The textarea stays editable while an answer streams (it can take a minute or two);
  // only sending is held until it finishes.
  const [sendQueued, setSendQueued] = useState(false);
  // A question waiting to be sent goes with the page: leaving asks first.
  useLeaveGuard(sendQueued);
  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled || pending) return;
    if (sendHeld) {
      if (sendQueueable) setSendQueued(true);
      return;
    }
    setSendQueued(false);
    onSubmit(trimmed);
    setValue('');
    if (draftKey) writeDraft(draftKey, '');
  };
  // The queued send, once startup is over — or dropped, when it ended unable to send
  // (an API key needed, no documents yet), the question kept in the box.
  const submitRef = useRef(submit);
  submitRef.current = submit;
  useEffect(() => {
    if (!sendQueued || sendHeld) return;
    if (disabled || !value.trim()) setSendQueued(false);
    else submitRef.current();
  }, [sendQueued, sendHeld, disabled, value]);

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
    if (event.key === 'Enter' && !event.shiftKey && !enterAddsNewline()) {
      event.preventDefault();
      submit();
      return;
    }
    // Up in an empty box: the last question back, to fix a typo or ask it a little
    // differently (as a shell recalls the last command).
    if (
      event.key === 'ArrowUp' &&
      !event.shiftKey &&
      !event.altKey &&
      !event.metaKey &&
      !event.ctrlKey &&
      value === '' &&
      recallQuestion
    ) {
      event.preventDefault();
      setValue(recallQuestion);
      const box = event.currentTarget;
      window.requestAnimationFrame(() => box.setSelectionRange(box.value.length, box.value.length));
    }
  };

  // Sets for the per-row checks: up to 500 selected files against every listed one was
  // a quadratic scan on each render of the popover.
  // ALLOWED_REQUEST_PROVIDERS: a provider the server refuses per request is shown
  // disabled rather than offered and turned away with a 400.
  const allowed = (name: string) => !config || isProviderAllowed(name, config);
  const openaiEnabled = config !== null && isProviderUsable('openai', config);
  const ollamaEnabled = config !== null && isProviderUsable('ollama', config);
  const compatibleAllowed = allowed('openai_compatible');
  const anthropicEnabled = config !== null && isProviderUsable('anthropic', config);

  return (
    <div className="composer" ref={wrapRef}>
      {hint && (
        <p className="composer__hint" data-testid="question-hint">
          {hint}
        </p>
      )}
      <p className="visually-hidden" role="status">
        {sendQueued ? 'Your question will be sent as soon as DocRAG is ready.' : ''}
      </p>
      {sendQueued && (
        <p className="composer__hint" data-testid="question-queued" aria-hidden="true">
          Sending as soon as DocRAG is ready…
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
      {(selectedFilenames.length > 0 || selectedTags.length > 0) && (
        <ul
          ref={chipsRef}
          className="composer__scope-chips"
          aria-label="Asking within"
          data-testid="composer-scope-chips"
        >
          {/* Past SCOPE_CHIPS_MAX, a group is one count, not a row of chips each (many
              tags stacked rows above the composer on a phone). */}
          {selectedFilenames.length > SCOPE_CHIPS_MAX ? (
            <li className="composer__scope-chip-text">{selectedFilenames.length} documents</li>
          ) : (
            selectedFilenames.map((name, index) => (
              <li key={`f:${name}`}>
                <button
                  type="button"
                  className="composer__scope-chip"
                  onClick={() => {
                    focusChipRef.current = index;
                    onSelectedFilenamesChange(selectedFilenames.filter((f) => f !== name));
                  }}
                  aria-label={`Stop limiting to ${name}`}
                  title={name}
                >
                  <span className="composer__scope-chip-text">{name.split('/').pop()}</span>
                  <span aria-hidden="true">×</span>
                </button>
              </li>
            ))
          )}
          {selectedTags.length > SCOPE_CHIPS_MAX ? (
            <li className="composer__scope-chip-text">{selectedTags.length} tags</li>
          ) : (
            selectedTags.map((tag, index) => (
              <li key={`t:${tag}`}>
                <button
                  type="button"
                  className="composer__scope-chip"
                  onClick={() => {
                    focusChipRef.current =
                      (selectedFilenames.length > SCOPE_CHIPS_MAX ? 0 : selectedFilenames.length) + index;
                    onSelectedTagsChange?.(selectedTags.filter((t) => t !== tag));
                  }}
                  aria-label={`Stop limiting to the tag ${tag}`}
                >
                  <span className="composer__scope-chip-text">#{tag}</span>
                  <span aria-hidden="true">×</span>
                </button>
              </li>
            ))
          )}
        </ul>
      )}
      <div className="composer__box">
        <textarea
          dir="auto"
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
              <ScopePopover
                indexedFilenames={indexedFilenames}
                selectedFilenames={selectedFilenames}
                onSelectedFilenamesChange={onSelectedFilenamesChange}
                availableTags={availableTags}
                selectedTags={selectedTags}
                onSelectedTagsChange={onSelectedTagsChange}
                pageRange={pageRange}
                onPageRangeChange={onPageRangeChange}
                uploadedAfter={uploadedAfter}
                onUploadedAfterChange={onUploadedAfterChange}
                uploadedBefore={uploadedBefore}
                onUploadedBeforeChange={onUploadedBeforeChange}
                scopePresets={scopePresets}
                onApplyScopePreset={onApplyScopePreset}
                onSaveScopePreset={onSaveScopePreset}
                onDeleteScopePreset={onDeleteScopePreset}
              />
            )}
          </div>
          <div className="composer__popover-anchor">
            <button
              type="button"
              className="composer__chip composer__chip--templates"
              onClick={() => {
                setTemplates(readTemplates());
                setOpenPopover((prev) => (prev === 'templates' ? null : 'templates'));
              }}
              aria-expanded={openPopover === 'templates'}
              aria-controls="composer-templates-popover"
              ref={templatesButtonRef}
              title="Saved questions to start from"
              aria-label="Templates"
              data-testid="composer-templates-button"
            >
              <span aria-hidden="true">☰</span>
              <span className="composer__chip-label"> Templates</span>
            </button>
            {openPopover === 'templates' && (
              <div
                className="composer__popover"
                id="composer-templates-popover"
                role="group"
                aria-label="Question templates"
                data-testid="composer-templates-popover"
              >
                {templates.length === 0 ? (
                  <p className="composer__popover-note">
                    No templates yet. Write a question (with {'{placeholders}'} to fill in) and
                    save it here.
                  </p>
                ) : (
                  <ul className="composer__templates">
                    {templates.map((template) => (
                      <li key={template.name} className="composer__popover-item">
                        <button
                          type="button"
                          className="composer__template"
                          onClick={() => insertTemplate(template)}
                          title={template.text}
                          data-testid="template-insert"
                        >
                          {template.name}
                        </button>
                        <button
                          type="button"
                          className="composer__template-delete"
                          onClick={() => setTemplates(deleteTemplate(template.name))}
                          aria-label={`Delete the template ${template.name}`}
                          data-testid="template-delete"
                        >
                          ×
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
                <form
                  className="composer__template-save"
                  onSubmit={(event) => {
                    event.preventDefault();
                    setTemplates(saveTemplate(templateName, value));
                    setTemplateName('');
                  }}
                >
                  <input
                    value={templateName}
                    onChange={(event) => setTemplateName(event.target.value)}
                    placeholder="Name"
                    maxLength={MAX_TEMPLATE_NAME_CHARS}
                    aria-label="Name for the question in the box"
                    data-testid="template-name"
                  />
                  <button
                    type="submit"
                    disabled={!templateName.trim() || !value.trim()}
                    data-testid="template-save"
                  >
                    Save question
                  </button>
                </form>
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
                {config.anthropic_available && (
                  <label className="composer__popover-item">
                    <input
                      type="radio"
                      name="composer-provider"
                      checked={provider === 'anthropic'}
                      disabled={!anthropicEnabled}
                      onChange={() => onProviderChange('anthropic')}
                      data-testid="provider-anthropic"
                    />
                    <span>Anthropic &middot; {config.anthropic_model}</span>
                  </label>
                )}
                {config.anthropic_available && !anthropicEnabled && (
                  <p className="composer__popover-note">Not allowed on this server.</p>
                )}
              </div>
            )}
          </div>
          <span className="composer__spacer" />
          {!enterAddsNewline() && (
            <span className="composer__key-hint">⏎ send &middot; ⇧⏎ newline</span>
          )}
          <button
            type="button"
            className="composer__send"
            onClick={pending ? onCancel : submit}
            disabled={
              pending ? false : disabled || (sendHeld && !sendQueueable) || sendQueued || !value.trim()
            }
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
