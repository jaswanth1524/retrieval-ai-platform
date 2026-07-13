import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import QuestionInput from '../../src/components/QuestionInput';

describe('QuestionInput', () => {
  it('submits and clears the value on Enter', async () => {
    const onSubmit = vi.fn();
    render(<QuestionInput onSubmit={onSubmit} disabled={false} />);

    const textarea = screen.getByTestId('question-textarea');
    await userEvent.type(textarea, 'What is the refund window?');
    await userEvent.keyboard('{Enter}');

    expect(onSubmit).toHaveBeenCalledWith('What is the refund window?');
    expect(textarea).toHaveValue('');
  });

  it('inserts a newline instead of submitting on Shift+Enter', async () => {
    const onSubmit = vi.fn();
    render(<QuestionInput onSubmit={onSubmit} disabled={false} />);

    const textarea = screen.getByTestId('question-textarea');
    await userEvent.type(textarea, 'line one{Shift>}{Enter}{/Shift}line two');

    expect(onSubmit).not.toHaveBeenCalled();
    expect(textarea).toHaveValue('line one\nline two');
  });

  it('does not submit whitespace-only input', async () => {
    const onSubmit = vi.fn();
    render(<QuestionInput onSubmit={onSubmit} disabled={false} />);

    await userEvent.type(screen.getByTestId('question-textarea'), '   {Enter}');

    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('disables the textarea and submit button when disabled', () => {
    render(<QuestionInput onSubmit={vi.fn()} disabled />);

    expect(screen.getByTestId('question-textarea')).toBeDisabled();
    expect(screen.getByTestId('question-submit')).toBeDisabled();
  });

  it('caps input length to match the backend max_length', () => {
    render(<QuestionInput onSubmit={vi.fn()} disabled={false} />);

    expect(screen.getByTestId('question-textarea')).toHaveAttribute('maxLength', '4000');
  });

  it('exposes an accessible name since the placeholder alone is not one', () => {
    render(<QuestionInput onSubmit={vi.fn()} disabled={false} />);

    expect(
      screen.getByRole('textbox', { name: 'Ask a question about your documents' }),
    ).toBeInTheDocument();
  });

  it('submits via the Ask button', async () => {
    const onSubmit = vi.fn();
    render(<QuestionInput onSubmit={onSubmit} disabled={false} />);

    await userEvent.type(screen.getByTestId('question-textarea'), 'Hello');
    await userEvent.click(screen.getByTestId('question-submit'));

    expect(onSubmit).toHaveBeenCalledWith('Hello');
  });

  it('renders a hint line when provided', () => {
    render(<QuestionInput onSubmit={vi.fn()} disabled hint="Upload a document to start." />);

    expect(screen.getByTestId('question-hint')).toHaveTextContent('Upload a document to start.');
  });

  it('renders no hint line when hint is omitted', () => {
    render(<QuestionInput onSubmit={vi.fn()} disabled={false} />);

    expect(screen.queryByTestId('question-hint')).not.toBeInTheDocument();
  });
});
