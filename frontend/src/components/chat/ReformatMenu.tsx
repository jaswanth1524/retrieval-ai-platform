import { useState } from 'react';
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
  if (!open) {
    return (
      <button
        type="button"
        className="chat-message__regenerate"
        aria-disabled={disabled || undefined}
        aria-expanded={false}
        onClick={() => !disabled && setOpen(true)}
        data-testid="chat-message-reformat"
      >
        Reformat
      </button>
    );
  }
  const choose = (style: ReformatStyle) => {
    setOpen(false);
    onReformat(style);
  };
  return (
    <span className="chat-message__reformat" role="group" aria-label="Reformat this answer">
      {SHAPES.map(([format, label]) => (
        <button
          key={format}
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
          if (!disabled && /^[\p{L}]+(?: [\p{L}]+)*$/u.test(name)) choose({ answerLanguage: name });
        }}
      >
        <input
          value={language}
          onChange={(event) => setLanguage(event.target.value)}
          placeholder="In… (e.g. German)"
          aria-label="Answer again in this language"
          maxLength={40}
          data-testid="chat-message-reformat-language"
        />
      </form>
      <button type="button" className="chat-message__regenerate" onClick={() => setOpen(false)}>
        Cancel
      </button>
    </span>
  );
}
