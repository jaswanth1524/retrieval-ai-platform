import { useEffect, useId, useRef, useState } from 'react';
import type { AnswerFormat } from '../../api/types';

export interface ReformatStyle {
  answerFormat?: AnswerFormat;
  answerLanguage?: string;
}

const SHAPES: Array<[AnswerFormat, string]> = [
  ['bullets', 'As bullets'],
  ['table', 'As a table'],
  ['brief', 'Shorter'],
];

/** "Reformat": the same question answered again as bullets, a table, briefly, or in
 *  another language. The current answer is kept as its earlier version. */
export default function ReformatMenu({
  onReformat,
  disabled = false,
}: {
  onReformat: (style: ReformatStyle) => void;
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [language, setLanguage] = useState('');
  const [languageError, setLanguageError] = useState(false);
  const errorId = useId();
  // The control that unmounts takes focus with it: opening puts it on the first
  // choice, closing (a choice, Cancel, Escape) back on the Reformat button.
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const firstChoiceRef = useRef<HTMLButtonElement | null>(null);
  const moveFocus = useRef(false);
  useEffect(() => {
    if (!moveFocus.current) return;
    moveFocus.current = false;
    (open ? firstChoiceRef.current : triggerRef.current)?.focus();
  }, [open]);
  const toggle = (next: boolean) => {
    moveFocus.current = true;
    setOpen(next);
  };
  if (!open) {
    return (
      <button
        ref={triggerRef}
        type="button"
        className="chat-message__regenerate"
        aria-disabled={disabled || undefined}
        aria-expanded={false}
        onClick={() => !disabled && toggle(true)}
        data-testid="chat-message-reformat"
      >
        Reformat
      </button>
    );
  }
  const choose = (style: ReformatStyle) => {
    toggle(false);
    onReformat(style);
  };
  return (
    <span
      className="chat-message__reformat"
      role="group"
      aria-label="Reformat this answer"
      onKeyDown={(event) => {
        if (event.key !== 'Escape') return;
        // Closes the menu, not the dialog or the stream the Escape would reach next.
        event.stopPropagation();
        event.preventDefault();
        toggle(false);
      }}
    >
      {SHAPES.map(([format, label], index) => (
        <button
          key={format}
          ref={index === 0 ? firstChoiceRef : undefined}
          type="button"
          className="chat-message__regenerate"
          aria-disabled={disabled || undefined}
          onClick={() => !disabled && choose({ answerFormat: format })}
          data-testid={`chat-message-reformat-${format}`}
        >
          {label}
        </button>
      ))}
      <form
        className="chat-message__reformat-language"
        onSubmit={(event) => {
          event.preventDefault();
          const name = language.trim();
          if (disabled) return;
          if (/^[\p{L}]+(?: [\p{L}]+)*$/u.test(name)) choose({ answerLanguage: name });
          else setLanguageError(true);
        }}
      >
        <input
          value={language}
          onChange={(event) => {
            setLanguage(event.target.value);
            setLanguageError(false);
          }}
          placeholder="In… (e.g. German)"
          aria-label="Answer again in this language"
          aria-invalid={languageError || undefined}
          aria-describedby={languageError ? errorId : undefined}
          maxLength={40}
          data-testid="chat-message-reformat-language"
        />
        {languageError && (
          <span id={errorId} className="chat-message__reformat-error" role="alert">
            A language name in letters, e.g. Portuguese
          </span>
        )}
      </form>
      <button type="button" className="chat-message__regenerate" onClick={() => toggle(false)}>
        Cancel
      </button>
    </span>
  );
}
