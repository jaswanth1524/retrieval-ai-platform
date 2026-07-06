import { useState } from 'react';
import type { KeyboardEvent } from 'react';
import './QuestionInput.css';

interface QuestionInputProps {
  onSubmit: (question: string) => void;
  disabled: boolean;
}

function QuestionInput({ onSubmit, disabled }: QuestionInputProps) {
  const [value, setValue] = useState('');

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

  return (
    <div className="question-input">
      <textarea
        className="question-input__textarea"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask a question about your documents..."
        disabled={disabled}
        rows={2}
        data-testid="question-textarea"
      />
      <button
        type="button"
        className="question-input__submit"
        onClick={submit}
        disabled={disabled || !value.trim()}
        data-testid="question-submit"
      >
        Ask
      </button>
    </div>
  );
}

export default QuestionInput;
